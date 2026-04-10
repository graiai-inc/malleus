#!/usr/bin/env python3
"""Run ASR with full token + vocab log probability capture.

Uses the local sherpa-onnx fork (must be built) to capture per-token
log probabilities and full vocabulary distributions for each transcription.
Handles chunking for models with max sequence length (e.g., parakeet 120s).

Saves per-file results incrementally.

Usage:
    PYTHONPATH=/home/grey/dev/graiai/sherpa-onnx/build/lib.linux-x86_64-cpython-312:$PYTHONPATH \\
        python scripts/run_benchmark_with_logprobs.py \\
        --model parakeet-tdt-0.6b-v2 --dataset primock57
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
OSSICLES_DIR = Path('/home/grey/dev/graiai/ossicles')
AUDIO_BASE = OSSICLES_DIR / 'assets' / 'audio'
MODELS_BASE = OSSICLES_DIR / 'assets' / 'models'

MODELS = {
    'parakeet-tdt-0.6b-v2': {
        'factory': 'from_transducer',
        'kwargs': {
            'encoder': 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx',
            'decoder': 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx',
            'joiner': 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx',
            'tokens': 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt',
            'model_type': 'nemo_transducer',
            'feature_dim': 128,
        },
        'chunk_seconds': 120,
    },
    'whisper-distil-v3.5': {
        'factory': 'from_whisper',
        'kwargs': {
            'encoder': 'sherpa-onnx-whisper-distil-large-v3.5/distil-large-v3.5-encoder.int8.onnx',
            'decoder': 'sherpa-onnx-whisper-distil-large-v3.5/distil-large-v3.5-decoder.int8.onnx',
            'tokens': 'sherpa-onnx-whisper-distil-large-v3.5/distil-large-v3.5-tokens.txt',
        },
        'chunk_seconds': 30,
    },
    'sensevoice-no-itn': {
        'factory': 'from_sense_voice',
        'kwargs': {
            'model': 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx',
            'tokens': 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt',
            'use_itn': False,
        },
        'chunk_seconds': None,
    },
}


def create_recognizer(model_name: str):
    import sherpa_onnx
    cfg = MODELS[model_name]
    factory = getattr(sherpa_onnx.OfflineRecognizer, cfg['factory'])
    kwargs = {k: (str(MODELS_BASE / v) if k in ('encoder', 'decoder', 'joiner', 'tokens', 'model') else v)
              for k, v in cfg['kwargs'].items()}
    kwargs['num_threads'] = 4
    return factory(**kwargs)


def summarize_vocab_dist(raw_logits: list[float]) -> dict:
    """Reduce a full vocab distribution to entropy + top-k probabilities.

    The raw arrays from sherpa-onnx may be raw logits or log-probs depending
    on the decoder. We log-softmax to be safe, then compute Shannon and
    Tsallis entropy + the top-5 token probabilities.
    """
    arr = np.asarray(raw_logits, dtype=np.float64)
    arr = arr - arr.max()
    probs = np.exp(arr)
    probs = probs / probs.sum()
    # Shannon entropy
    nz = probs[probs > 0]
    shannon = float(-(nz * np.log(nz)).sum())
    # Tsallis entropy (q=0.333, per dictator research doc)
    q = 0.333
    tsallis = float((1.0 - (probs ** q).sum()) / (q - 1.0))
    # Top 5 probabilities (indices not stored — vocab varies per model)
    top5 = np.sort(probs)[-5:][::-1].tolist()
    return {
        'shannon': round(shannon, 4),
        'tsallis': round(tsallis, 4),
        'top5_probs': [round(p, 6) for p in top5],
    }


def transcribe_with_logprobs(recognizer, audio_path: str, chunk_seconds):
    """Transcribe with log prob capture, chunking if needed.

    Stores per-token ys_log_probs (full) and per-token vocab summaries
    (entropy + top-5) instead of raw distributions to keep file sizes small.
    """
    data, sr = sf.read(audio_path, dtype='float32')
    if len(data.shape) > 1:
        data = data[:, 0]

    all_text = []
    all_tokens = []
    all_ys_log_probs = []
    all_vocab_summaries = []

    if chunk_seconds is None or len(data) <= chunk_seconds * sr:
        chunks = [data]
    else:
        max_samples = chunk_seconds * sr
        chunks = []
        offset = 0
        while offset < len(data):
            end = min(offset + max_samples, len(data))
            chunks.append(data[offset:end])
            offset = end

    for chunk in chunks:
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, chunk.tolist())
        recognizer.decode_stream(stream)
        result = stream.result
        text = result.text.strip()
        if text:
            all_text.append(text)
        all_tokens.extend(list(result.tokens))
        all_ys_log_probs.extend(list(result.ys_log_probs))
        for v in result.vocab_log_probs:
            all_vocab_summaries.append(summarize_vocab_dist(v))

    return {
        'text': ' '.join(all_text),
        'tokens': all_tokens,
        'ys_log_probs': all_ys_log_probs,
        'vocab_summaries': all_vocab_summaries,
    }


def load_references(dataset: str) -> dict:
    """Load references from ossicles benchmark results."""
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


def run(model_name: str, dataset: str, max_files: int = None):
    output_dir = MALLEUS_DIR / 'results' / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{model_name}.jsonl'

    audio_dir = AUDIO_BASE / dataset
    audio_files = sorted(audio_dir.glob('*.wav'))
    if max_files:
        audio_files = audio_files[:max_files]

    references = load_references(dataset)

    log.info(f'{model_name} on {dataset}: {len(audio_files)} files')
    log.info(f'Output: {output_file}')

    # Resume support: read existing file_ids
    done_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['file_id'])
                except:
                    pass
        log.info(f'Resuming: {len(done_ids)} files already done')

    recognizer = create_recognizer(model_name)
    chunk_seconds = MODELS[model_name].get('chunk_seconds')

    with open(output_file, 'a') as out:
        for i, audio_file in enumerate(audio_files):
            file_id = audio_file.stem
            if file_id in done_ids:
                continue

            start = time.time()
            try:
                r = transcribe_with_logprobs(recognizer, str(audio_file), chunk_seconds)
            except Exception as e:
                log.error(f'  [{i+1}/{len(audio_files)}] {file_id}: FAILED - {e}')
                continue
            elapsed = time.time() - start

            record = {
                'file_id': file_id,
                'hypothesis': r['text'],
                'reference': references.get(file_id, ''),
                'tokens': r['tokens'],
                'ys_log_probs': r['ys_log_probs'],
                'vocab_summaries': r['vocab_summaries'],
                'inference_time_s': round(elapsed, 3),
            }
            out.write(json.dumps(record) + '\n')
            out.flush()

            n_tokens = len(r['tokens'])
            log.info(f'  [{i+1}/{len(audio_files)}] {file_id}: '
                     f'{n_tokens} tokens, {elapsed:.1f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--max-files', type=int, default=None)
    args = parser.parse_args()
    run(args.model, args.dataset, args.max_files)


if __name__ == '__main__':
    main()
