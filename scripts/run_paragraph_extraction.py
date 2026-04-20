#!/usr/bin/env python3
"""Gemma text-only paragraph-level extraction for Parakeet low-confidence words.

Based on literature (HyPoradise, ProGRes, Yang et al. 2024, RLLM-CF 2025):
naive single-word correction INCREASES WER. What works is constrained
selection / structured multi-slot extraction with:
  - Domain priming (medical consultation)
  - Wider context (paragraph not 5-word window)
  - Structured output (JSON) — removes paraphrasing tendency
  - Multi-slot per call — amortizes prefill, lets model use discourse coherence

We can't do true N-best rescoring because malleus saved only top-5 *probabilities*,
not the top-5 token strings. So we use Parakeet's single hypothesis per slot but
constrain the format tightly.

Writes results incrementally per slot: /sensorium/results/text_infill/paragraph_results.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import litert_lm
from litert_lm import Backend

THALAMUS = Path('/home/grey/dev/graiai/thalamus')
MODEL_PATH = THALAMUS / 'checkpoints' / 'gemma4_e4b_litertlm' / 'model.litertlm'
JSONL_IN = Path('/home/grey/dev/graiai/malleus/results/primock57/parakeet-tdt-0.6b-v2.jsonl')
SPANS_TSV = Path('/home/grey/dev/graiai/sensorium/results/text_infill/flagged_spans.tsv')
OUT_TSV = Path('/home/grey/dev/graiai/sensorium/results/text_infill/paragraph_results.tsv')

BACKEND = Backend.CPU
MAX_NUM_TOKENS = 4096

# Chunking
SLOTS_PER_CHUNK = 15      # target # flagged words per Gemma call
CTX_BEFORE = 20           # words of context before first slot
CTX_AFTER = 20            # words of context after last slot


PROMPT_WITH_HINT = '''You are auditing an automatic speech-recognition transcript of a medical doctor-patient consultation. Some words in the transcript were flagged as uncertain; they appear in the text as <<SLOT_N:word>> where N is a slot number and "word" is what the ASR heard.

Transcript chunk:
"{chunk}"

For each slot, determine the most likely word spoken, considering:
- The surrounding medical conversation (doctor-patient question/answer rhythm, symptoms, medications, body parts, procedures).
- Grammar and natural speech patterns.
- Whether the ASR-heard word fits or is a likely mis-hearing.

If the ASR-heard word is correct in context, keep it. If a different word is clearly more likely, substitute it. Prefer common, everyday words over rare alternatives unless context demands otherwise.

Output format: a single JSON object mapping each slot number (as a string) to the single most likely word (lowercase, no punctuation). Output ONLY the JSON, no other text.

Example for a transcript with slots 1, 2, 3:
{{"1": "hello", "2": "should", "3": "toilet"}}'''


PROMPT_BLIND = '''You are reconstructing an automatic speech-recognition transcript of a medical doctor-patient consultation. Some words were unclear in the recording and appear in the text as <<N>> placeholders, where N is a slot number.

Transcript with missing words:
"{chunk}"

For each numbered slot, determine the single most likely word that was spoken at that position, based on:
- The surrounding medical conversation context (doctor-patient question/answer rhythm, symptoms, medications, body parts, procedures, common clinical phrases).
- Grammar and natural conversational speech patterns.

Output format: a single JSON object mapping each slot number (as a string) to the single most likely word (lowercase, no punctuation). Output ONLY the JSON, no other text.

Example for a transcript with slots 1, 2, 3:
{{"1": "hello", "2": "should", "3": "toilet"}}'''


def normalize_word(w: str) -> str:
    return re.sub(r'[^\w\']', '', w).lower().strip()


def subwords_to_words(tokens, logprobs):
    """Same aggregation as analyze_parakeet_confidences.py."""
    words = []
    cur_text = ''
    cur_start = 0
    for i, tok in enumerate(tokens):
        if tok.startswith(' ') and cur_text:
            words.append((cur_text, cur_start, i))
            cur_text = tok[1:]
            cur_start = i
        elif tok.startswith(' '):
            cur_text = tok[1:]
            cur_start = i
        else:
            cur_text += tok
    if cur_text:
        words.append((cur_text, cur_start, len(tokens)))
    return [(t, s, e) for (t, s, e) in words if e > s]


def load_flagged_by_file():
    """Map file_id -> list of (word_idx, hyp_word, hyp_norm, aligned_ref, parakeet_correct)."""
    by_file = defaultdict(list)
    with SPANS_TSV.open() as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            d = dict(zip(header, parts))
            if not d['aligned_ref_word']:
                continue  # skip pure insertions
            by_file[d['file_id']].append({
                'word_idx': int(d['word_idx']),
                'hyp_word': d['hyp_word'],
                'hyp_norm': d['hyp_norm'],
                'aligned_ref_word': d['aligned_ref_word'],
                'parakeet_correct': int(d['parakeet_correct']),
                'left_ctx': d['left_ctx'],
                'right_ctx': d['right_ctx'],
            })
    return by_file


def build_chunks(hyp_words, flagged_list, blind=False):
    """Group flagged words into chunks of <= SLOTS_PER_CHUNK, with context.

    If blind=True, slots are tagged as <<N>> (no Parakeet hint).
    Otherwise tagged as <<N:hyp_word>>.
    """
    chunks = []
    sorted_flagged = sorted(flagged_list, key=lambda x: x['word_idx'])

    i = 0
    while i < len(sorted_flagged):
        batch = sorted_flagged[i:i + SLOTS_PER_CHUNK]
        first_idx = batch[0]['word_idx']
        last_idx = batch[-1]['word_idx']
        start = max(0, first_idx - CTX_BEFORE)
        end = min(len(hyp_words), last_idx + CTX_AFTER + 1)

        out = []
        batch_by_idx = {b['word_idx']: (k + 1, b) for k, b in enumerate(batch)}
        for j in range(start, end):
            entry = batch_by_idx.get(j)
            if entry:
                slot_id, b = entry
                if blind:
                    out.append(f'<<{slot_id}>>')
                else:
                    out.append(f'<<{slot_id}:{b["hyp_word"]}>>')
                b['slot_id'] = slot_id
            else:
                out.append(hyp_words[j])

        chunk_text = ' '.join(out)
        chunks.append((batch, chunk_text))
        i += SLOTS_PER_CHUNK
    return chunks


def run_gemma(engine, prompt):
    with engine.create_conversation() as conv:
        stream = conv.send_message_async(prompt)
        buf = []
        for chunk in stream:
            for item in chunk.get('content', []):
                if item.get('type') == 'text':
                    buf.append(item.get('text', ''))
    return ''.join(buf).strip()


def parse_json_output(raw: str):
    """Extract the first JSON object from Gemma's response. Robust to preamble / code fences."""
    # Strip code fences if present
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if not m:
        return None, 'no json found'
    try:
        return json.loads(m.group(0)), None
    except Exception as e:
        return None, f'json parse failed: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-files', type=int, default=0, help='Max files to process (0=all)')
    ap.add_argument('--limit-chunks', type=int, default=0, help='Max chunks across all files (0=all)')
    ap.add_argument('--blind', action='store_true', help='Hide Parakeet hint (<<N>> instead of <<N:hyp>>)')
    ap.add_argument('--out', type=str, default=str(OUT_TSV), help='Output TSV path')
    args = ap.parse_args()

    out_path = Path(args.out)

    # Load per-file hypothesis word lists from JSONL
    print('Loading hypothesis words per file from malleus JSONL…', flush=True)
    hyps_per_file = {}
    for line in JSONL_IN.read_text().splitlines():
        d = json.loads(line)
        words = subwords_to_words(d['tokens'], d['ys_log_probs'])
        hyps_per_file[d['file_id']] = [w[0] for w in words]

    # Load flagged-span table
    print('Loading flagged spans…', flush=True)
    by_file = load_flagged_by_file()
    file_ids = sorted(by_file.keys())
    if args.limit_files:
        file_ids = file_ids[:args.limit_files]

    print(f'Files to process: {len(file_ids)}', flush=True)

    # Prepare output
    fh = out_path.open('w')
    fh.write('\t'.join([
        'file_id', 'chunk_id', 'slot_id', 'word_idx',
        'hyp_word', 'hyp_norm', 'aligned_ref_word', 'parakeet_correct',
        'gemma_out', 'gemma_matches_ref', 'gemma_matches_hyp',
        'verdict',  # fix / newerror / keep_correct / still_wrong / wrong_diff / json_error
        'chunk_ms', 'raw_response_preview',
    ]) + '\n')
    fh.flush()

    # Load model
    print(f'Loading {MODEL_PATH.name}…', flush=True)
    t0 = datetime.now(timezone.utc)
    engine = litert_lm.Engine(str(MODEL_PATH), backend=BACKEND, max_num_tokens=MAX_NUM_TOKENS)
    print(f'  loaded in {(datetime.now(timezone.utc) - t0).total_seconds():.1f}s', flush=True)
    print(flush=True)

    total_chunks = 0
    n_fix = n_newerror = n_keep = n_still_wrong = n_wrong_diff = n_json_err = 0
    start_wall = datetime.now(timezone.utc)

    for file_id in file_ids:
        hyp_words = hyps_per_file[file_id]
        flagged = by_file[file_id]
        chunks = build_chunks(hyp_words, flagged, blind=args.blind)
        print(f'[{file_id}] {len(flagged)} flagged, {len(chunks)} chunks', flush=True)

        for ci, (batch, chunk_text) in enumerate(chunks):
            if args.limit_chunks and total_chunks >= args.limit_chunks:
                break
            template = PROMPT_BLIND if args.blind else PROMPT_WITH_HINT
            prompt = template.format(chunk=chunk_text)
            t0 = datetime.now(timezone.utc)
            try:
                raw = run_gemma(engine, prompt)
            except Exception as e:
                raw = f'<ERROR:{e}>'
            dt_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

            parsed, err = parse_json_output(raw)
            preview = raw.replace('\n', ' ').replace('\t', ' ')[:200]

            for b in batch:
                slot_id = b['slot_id']
                hyp_norm = b['hyp_norm']
                ref = b['aligned_ref_word']
                pc = b['parakeet_correct']

                if parsed is None:
                    out = ''
                    verdict = 'json_error'
                    n_json_err += 1
                else:
                    out = normalize_word(str(parsed.get(str(slot_id), '')))
                    matches_ref = int(out == ref)
                    matches_hyp = int(out == hyp_norm)
                    if pc == 1:
                        if matches_ref:
                            verdict = 'keep_correct'; n_keep += 1
                        else:
                            verdict = 'newerror'; n_newerror += 1
                    else:
                        if matches_ref:
                            verdict = 'fix'; n_fix += 1
                        elif matches_hyp:
                            verdict = 'still_wrong'; n_still_wrong += 1
                        else:
                            verdict = 'wrong_diff'; n_wrong_diff += 1

                fh.write('\t'.join([
                    file_id, str(ci), str(slot_id), str(b['word_idx']),
                    b['hyp_word'], hyp_norm, ref, str(pc),
                    out, str(int(out == ref)), str(int(out == hyp_norm)),
                    verdict, str(dt_ms), preview,
                ]) + '\n')
                fh.flush()

            total_chunks += 1
            total = n_keep + n_newerror + n_fix + n_still_wrong + n_wrong_diff
            n_err = total - n_keep - n_newerror
            n_correct = n_keep + n_newerror
            fix_rate = n_fix / n_err if n_err else 0
            ne_rate = n_newerror / n_correct if n_correct else 0
            elapsed = (datetime.now(timezone.utc) - start_wall).total_seconds()
            print(
                f'  chunk {ci}: {len(batch)} slots  {dt_ms}ms  '
                f'fix={n_fix} ne={n_newerror} still_wrong={n_still_wrong} '
                f'wrong_diff={n_wrong_diff} json_err={n_json_err}  '
                f'fix_rate={fix_rate:.1%} ne_rate={ne_rate:.1%}  '
                f'[{total_chunks/elapsed:.2f} chunks/s]',
                flush=True,
            )
        if args.limit_chunks and total_chunks >= args.limit_chunks:
            break

    fh.close()
    print(flush=True)
    total = n_keep + n_newerror + n_fix + n_still_wrong + n_wrong_diff
    n_err = total - n_keep - n_newerror
    n_correct = n_keep + n_newerror
    print(f'Total slots: {total + n_json_err}  (json_err={n_json_err})', flush=True)
    print(f'  Parakeet correct: {n_correct}  Gemma kept: {n_keep} ({n_keep/n_correct:.1%})  Gemma introduced error: {n_newerror} ({n_newerror/n_correct:.1%})', flush=True)
    if n_err:
        print(f'  Parakeet wrong:   {n_err}  Gemma fixed: {n_fix} ({n_fix/n_err:.1%})  still_wrong: {n_still_wrong} ({n_still_wrong/n_err:.1%})  wrong_diff: {n_wrong_diff} ({n_wrong_diff/n_err:.1%})', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
