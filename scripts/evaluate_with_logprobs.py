#!/usr/bin/env python3
"""Re-run ASR benchmark capturing token and vocabulary log probabilities.

Uses the local sherpa-onnx fork with vocab_log_probs support.
Outputs per-file results incrementally, including tokens, ys_log_probs,
and vocab_log_probs for use in confidence-weighted ROVER experiments.

Usage:
    python scripts/evaluate_with_logprobs.py --model sensevoice-no-itn --dataset figshare-osce
    python scripts/evaluate_with_logprobs.py --model all --dataset all
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MALLEUS_DIR = SCRIPT_DIR.parent

# Audio directories (symlink or configure)
OSSICLES_DIR = Path('/home/grey/dev/graiai/ossicles')
AUDIO_BASE = OSSICLES_DIR / 'assets' / 'audio'
MODELS_BASE = OSSICLES_DIR / 'assets' / 'models'

DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']

# Top 5 models for malleus experiments
MODELS = {
    'parakeet-tdt-0.6b-v2': {
        'type': 'transducer',
        'model_dir': 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8',
    },
    'whisper-distil-v3.5': {
        'type': 'whisper',
        'model_dir': 'sherpa-onnx-whisper-distil-large-v3.5',
    },
    'sensevoice-no-itn': {
        'type': 'sensevoice',
        'model_dir': 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17',
    },
    'qwen3-asr': {
        'type': 'qwen3',
        'model_dir': 'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25',
    },
    'whisper-turbo': {
        'type': 'whisper',
        'model_dir': 'sherpa-onnx-whisper-turbo',
    },
}


def create_recognizer(model_name: str):
    """Create sherpa-onnx recognizer for the given model."""
    import sherpa_onnx

    config = MODELS[model_name]
    model_dir = MODELS_BASE / config['model_dir']

    if config['type'] == 'sensevoice':
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / 'model.int8.onnx'),
            tokens=str(model_dir / 'tokens.txt'),
            use_itn=False,
            num_threads=4,
        )
    elif config['type'] == 'whisper':
        encoder = model_dir / 'encoder.int8.onnx'
        if not encoder.exists():
            encoder = model_dir / 'encoder.onnx'
        decoder = model_dir / 'decoder.int8.onnx'
        if not decoder.exists():
            decoder = model_dir / 'decoder.onnx'
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(model_dir / 'tokens.txt'),
            num_threads=4,
        )
    elif config['type'] == 'transducer':
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / 'encoder.int8.onnx'),
            decoder=str(model_dir / 'decoder.int8.onnx'),
            joiner=str(model_dir / 'joiner.int8.onnx'),
            tokens=str(model_dir / 'tokens.txt'),
            num_threads=4,
        )
    elif config['type'] == 'qwen3':
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            model=str(model_dir / 'model.int8.onnx'),
            tokenizer=str(model_dir / 'tokenizer.json'),
            tokens=str(model_dir / 'tokens.txt'),
            num_threads=4,
        )
    else:
        raise ValueError(f'Unknown model type: {config["type"]}')


def transcribe_with_logprobs(recognizer, audio_path: str) -> dict:
    """Transcribe audio and capture log probabilities."""
    data, sr = sf.read(audio_path, dtype='float32')
    if len(data.shape) > 1:
        data = data[:, 0]

    stream = recognizer.create_stream()
    stream.accept_waveform(sr, data.tolist())
    recognizer.decode_stream(stream)

    result = stream.result
    text = result.text.strip()
    tokens = list(result.tokens)
    ys_log_probs = list(result.ys_log_probs)

    # vocab_log_probs is a list of lists (one per token)
    # Convert to per-token entropy and top-k for storage efficiency
    vocab_stats = []
    for i, vocab_dist in enumerate(result.vocab_log_probs):
        if not vocab_dist:
            vocab_stats.append(None)
            continue
        arr = np.array(vocab_dist, dtype=np.float64)
        # Apply log-softmax (raw logits, not normalized)
        arr = arr - np.max(arr)
        probs = np.exp(arr)
        probs = probs / probs.sum()
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        # Top 5
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = [(int(idx), float(probs[idx])) for idx in top5_idx]
        vocab_stats.append({
            'entropy': round(float(entropy), 4),
            'top5': top5,
        })

    return {
        'text': text,
        'tokens': tokens,
        'ys_log_probs': ys_log_probs,
        'vocab_stats': vocab_stats,
    }


def load_references(dataset: str) -> dict:
    """Load reference transcripts from stapes results."""
    stapes_results = Path('/tmp/stapes/results/on_device') / dataset
    # Use any model's results to get references
    for f in stapes_results.glob('*.json'):
        data = json.loads(f.read_text())
        if isinstance(data, dict) and 'results' in data:
            return {r['file_id']: r['reference'] for r in data['results']}
    # Fallback to ossicles
    ossicles_results = OSSICLES_DIR / f'benchmark_results_{dataset}'
    for f in ossicles_results.glob('*.json'):
        if '.old' in f.name or 'fusion' in f.stem or 'medical' in f.stem:
            continue
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and 'results' in data:
                return {r['file_id']: r['reference'] for r in data['results']}
        except:
            continue
    return {}


def run_model_dataset(model_name: str, dataset: str):
    """Run one model on one dataset, saving per-file results incrementally."""
    output_dir = MALLEUS_DIR / 'results' / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{model_name}.json'

    audio_dir = AUDIO_BASE / dataset
    if not audio_dir.exists():
        log.error(f'Audio directory not found: {audio_dir}')
        return

    references = load_references(dataset)
    audio_files = sorted(audio_dir.glob('*.wav'))
    log.info(f'{model_name} on {dataset}: {len(audio_files)} files')

    # Check for existing results to resume
    existing_results = {}
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text())
            existing_results = {r['file_id']: r for r in data.get('results', [])}
            log.info(f'  Resuming: {len(existing_results)} files already done')
        except:
            pass

    recognizer = create_recognizer(model_name)
    results = list(existing_results.values())

    for i, audio_file in enumerate(audio_files):
        file_id = audio_file.stem
        if file_id in existing_results:
            continue

        start = time.time()
        try:
            logprob_result = transcribe_with_logprobs(recognizer, str(audio_file))
        except Exception as e:
            log.error(f'  {file_id}: FAILED - {e}')
            continue
        elapsed = time.time() - start

        result = {
            'file_id': file_id,
            'hypothesis': logprob_result['text'],
            'reference': references.get(file_id, ''),
            'tokens': logprob_result['tokens'],
            'ys_log_probs': logprob_result['ys_log_probs'],
            'vocab_stats': logprob_result['vocab_stats'],
            'inference_time_s': round(elapsed, 3),
        }
        results.append(result)

        # Write incrementally
        output_file.write_text(json.dumps({
            'model': model_name,
            'dataset': dataset,
            'results': results,
        }, indent=2))

        log.info(f'  [{len(results)}/{len(audio_files)}] {file_id}: '
                 f'{len(logprob_result["tokens"])} tokens, '
                 f'{len(logprob_result["ys_log_probs"])} log_probs, '
                 f'{elapsed:.1f}s')

    log.info(f'{model_name} on {dataset}: done ({len(results)} files)')


def main():
    parser = argparse.ArgumentParser(description='ASR benchmark with log probs')
    parser.add_argument('--model', type=str, default='all',
                        choices=list(MODELS.keys()) + ['all'])
    parser.add_argument('--dataset', type=str, default='all',
                        choices=DATASETS + ['all'])
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == 'all' else [args.model]
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for model in models:
        for dataset in datasets:
            run_model_dataset(model, dataset)


if __name__ == '__main__':
    main()
