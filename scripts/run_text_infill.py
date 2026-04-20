#!/usr/bin/env python3
"""Run Gemma-4-E4B text-only infill on Parakeet's low-confidence spans.

Reads flagged_spans.tsv from analyze_parakeet_confidences.py.
For each flagged span, builds a masked-context prompt and asks Gemma for the
single most likely word. Scores vs the aligned reference word.

Must be run with the litert-lm uv tool:
    /home/grey/.local/share/uv/tools/litert-lm/bin/python run_text_infill.py [--limit N]

Writes output incrementally per-span (flush every row) to:
    sensorium/results/text_infill/infill_results.tsv

Skips rows with no aligned_ref_word (pure Parakeet insertions — Gemma can't
meaningfully infill "").
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import litert_lm
from litert_lm import Backend

THALAMUS = Path('/home/grey/dev/graiai/thalamus')
MODEL_PATH = THALAMUS / 'checkpoints' / 'gemma4_e4b_litertlm' / 'model.litertlm'
SPANS_TSV = Path('/home/grey/dev/graiai/sensorium/results/text_infill/flagged_spans.tsv')
OUT_TSV = Path('/home/grey/dev/graiai/sensorium/results/text_infill/infill_results.tsv')

BACKEND = Backend.CPU
MAX_NUM_TOKENS = 4096  # xnnpack_cache in the model dir is compiled against a
# specific value; mismatching can cause DYNAMIC_UPDATE_SLICE crashes at prefill.


PROMPT_TEMPLATE = '''You are reviewing a speech-recognition transcript. The ASR system is not fully confident about one word and you must judge whether it got that word right.

Transcript around the uncertain word:
"{left} [{hyp}] {right}"

The bracketed word [{hyp}] is what the ASR heard. Given the surrounding words and natural speech patterns, what word was most likely actually spoken at that position?

Rules:
- If the bracketed word is correct, output it unchanged.
- If a different word is more likely, output that word instead.
- Output only the single word, all lowercase, no punctuation, no explanation, no quotation marks.'''


def normalize_word(w: str) -> str:
    return re.sub(r'[^\w\']', '', w).lower().strip()


def infill_one(engine, left_ctx: str, right_ctx: str, hyp: str) -> str:
    """Ask Gemma to judge/correct the bracketed word. Return first word of
    its output, normalized."""
    prompt = PROMPT_TEMPLATE.format(left=left_ctx, right=right_ctx, hyp=hyp)
    with engine.create_conversation() as conv:
        stream = conv.send_message_async(prompt)
        buf = []
        for chunk in stream:
            for item in chunk.get('content', []):
                if item.get('type') == 'text':
                    buf.append(item.get('text', ''))
                    # Early stop: if we've got whitespace after a word, we're done
                    joined = ''.join(buf).strip()
                    if joined and ' ' in joined:
                        break
            if buf and ' ' in ''.join(buf).strip():
                break
    raw = ''.join(buf).strip()
    # Take the first whitespace-delimited token
    first = raw.split()[0] if raw.split() else ''
    return normalize_word(first)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='Max spans to process (0 = all)')
    ap.add_argument('--start', type=int, default=0, help='Skip this many spans from the start')
    ap.add_argument('--resume', action='store_true', help='Append to existing out file, skip already-done file_id+word_idx pairs')
    args = ap.parse_args()

    done_pairs: set[tuple[str, int]] = set()
    file_mode = 'w'
    if args.resume and OUT_TSV.exists():
        print(f'Resuming from {OUT_TSV}', flush=True)
        for ln in OUT_TSV.read_text().splitlines()[1:]:
            parts = ln.split('\t')
            if len(parts) >= 2:
                done_pairs.add((parts[0], int(parts[1])))
        print(f'  already processed: {len(done_pairs)} spans', flush=True)
        file_mode = 'a'

    # Load spans TSV
    span_rows = []
    with SPANS_TSV.open() as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            d = dict(zip(header, parts))
            if not d.get('aligned_ref_word'):
                continue  # skip pure insertions
            span_rows.append(d)

    print(f'Total infill-able spans: {len(span_rows)}', flush=True)
    if args.start:
        span_rows = span_rows[args.start:]
        print(f'  after --start {args.start}: {len(span_rows)}', flush=True)
    if args.limit:
        span_rows = span_rows[:args.limit]
        print(f'  after --limit {args.limit}: {len(span_rows)}', flush=True)

    # Load Gemma
    print(f'Loading {MODEL_PATH} on {BACKEND.name}…', flush=True)
    t0 = datetime.now(timezone.utc)
    engine = litert_lm.Engine(
        str(MODEL_PATH),
        backend=BACKEND,
        max_num_tokens=MAX_NUM_TOKENS,
    )
    load_s = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f'  loaded in {load_s:.1f}s', flush=True)
    print(flush=True)

    fh = OUT_TSV.open(file_mode)
    if file_mode == 'w':
        fh.write('\t'.join([
            'file_id', 'word_idx', 'hyp_word', 'parakeet_correct',
            'aligned_ref_word', 'gemma_infill',
            'gemma_matches_ref', 'gemma_matches_hyp',
            'verdict',  # fix / newerror / keep_correct / still_wrong / wrong_diff
            'infill_ms',
        ]) + '\n')
        fh.flush()

    # Stats
    n_fix = n_newerror = n_keep = n_still_wrong = n_wrong_diff = 0

    start_wall = datetime.now(timezone.utc)
    for i, d in enumerate(span_rows):
        key = (d['file_id'], int(d['word_idx']))
        if key in done_pairs:
            continue
        hyp_norm = d['hyp_norm']
        ref = d['aligned_ref_word']
        parakeet_correct = int(d['parakeet_correct'])
        left = d['left_ctx']
        right = d['right_ctx']

        t0 = datetime.now(timezone.utc)
        try:
            out = infill_one(engine, left, right, d['hyp_word'])
        except Exception as e:
            out = f'<ERROR:{e}>'
        dt_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

        gemma_matches_ref = 1 if out == ref else 0
        gemma_matches_hyp = 1 if out == hyp_norm else 0

        if parakeet_correct == 1:
            if gemma_matches_ref == 1:
                verdict = 'keep_correct'
                n_keep += 1
            else:
                verdict = 'newerror'
                n_newerror += 1
        else:
            if gemma_matches_ref == 1:
                verdict = 'fix'
                n_fix += 1
            elif gemma_matches_hyp == 1:
                verdict = 'still_wrong'  # Gemma agreed with Parakeet's wrong word
                n_still_wrong += 1
            else:
                verdict = 'wrong_diff'
                n_wrong_diff += 1

        fh.write('\t'.join([
            d['file_id'], d['word_idx'], d['hyp_word'], str(parakeet_correct),
            ref, out,
            str(gemma_matches_ref), str(gemma_matches_hyp),
            verdict, str(dt_ms),
        ]) + '\n')
        fh.flush()

        if (i + 1) % 25 == 0 or i < 5:
            elapsed = (datetime.now(timezone.utc) - start_wall).total_seconds()
            rate = (i + 1) / elapsed
            total = n_keep + n_newerror + n_fix + n_still_wrong + n_wrong_diff
            n_err_flagged = total - n_keep - n_newerror  # Parakeet-wrong spans
            n_correct_flagged = n_keep + n_newerror
            fix_rate = n_fix / n_err_flagged if n_err_flagged else 0
            ne_rate = n_newerror / n_correct_flagged if n_correct_flagged else 0
            print(
                f'  [{i+1}/{len(span_rows)}] {d["file_id"]} w{d["word_idx"]}: '
                f'hyp="{hyp_norm}" ref="{ref}" gemma="{out}" -> {verdict}  '
                f'({dt_ms}ms)  | fix={n_fix} ne={n_newerror}  '
                f'fix_rate={fix_rate:.1%} ne_rate={ne_rate:.1%}  '
                f'[{rate:.1f}/s, eta {(len(span_rows)-i-1)/rate/60:.0f}min]',
                flush=True,
            )

    fh.close()

    total = n_keep + n_newerror + n_fix + n_still_wrong + n_wrong_diff
    n_err = total - n_keep - n_newerror
    n_correct = n_keep + n_newerror
    print(flush=True)
    print(f'Total processed: {total}', flush=True)
    print(f'  Parakeet correct: {n_correct}  Gemma kept correct: {n_keep} ({n_keep/n_correct:.1%})   Gemma introduced error: {n_newerror} ({n_newerror/n_correct:.1%})', flush=True)
    print(f'  Parakeet wrong:   {n_err}  Gemma fixed: {n_fix} ({n_fix/n_err:.1%})  still_wrong (agreed): {n_still_wrong} ({n_still_wrong/n_err:.1%})  wrong_diff: {n_wrong_diff} ({n_wrong_diff/n_err:.1%})', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
