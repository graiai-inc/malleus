#!/usr/bin/env python3
"""Aggregate Parakeet subword log-probs to word level, align to reference,
flag low-confidence spans for infill experiment.

Writes an incremental TSV per flagged span. NO Gemma calls — that's a
separate script that consumes this output.

Output columns:
  file_id, word_idx, hyp_word, left_ctx, right_ctx,
  min_logprob, mean_logprob, n_subwords,
  aligned_ref_word, parakeet_correct

"parakeet_correct" is 1 if hyp_word (normalized) == ref word at aligned pos,
else 0. "aligned_ref_word" is None if this hyp word has no ref counterpart
(insertion by Parakeet). We still include insertions — they're the class of
errors we'd want text-only infill to remove (output "").
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import jiwer

JSONL_IN = Path('/home/grey/dev/graiai/malleus/results/primock57/parakeet-tdt-0.6b-v2.jsonl')
OUT_DIR = Path('/home/grey/dev/graiai/sensorium/results/text_infill')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FLAGGED_TSV = OUT_DIR / 'flagged_spans.tsv'
SUMMARY_TSV = OUT_DIR / 'file_summary.tsv'

CTX_WORDS = 5
BOTTOM_PCT = 0.20  # flag bottom 20% by min_logprob


def normalize_word(w: str) -> str:
    """Strip punctuation and lowercase for alignment + equality comparison."""
    return re.sub(r'[^\w\']', '', w).lower()


def subwords_to_words(tokens: list[str], logprobs: list[float]):
    """Return list of (word_text, start_idx, end_idx_exclusive, min_lp, mean_lp).

    Word boundaries: a token starting with a space begins a new word.
    Punctuation-only tokens are attached to the preceding word.
    """
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

    # Compute per-word log-prob stats
    out = []
    for text, s, e in words:
        lps = logprobs[s:e]
        if not lps:
            continue
        min_lp = min(lps)
        mean_lp = sum(lps) / len(lps)
        out.append((text, s, e, min_lp, mean_lp))
    return out


def align_hyp_to_ref(hyp_words: list[str], ref_words: list[str]):
    """Use jiwer to align hypothesis word list to reference word list.

    Returns a list parallel to hyp_words. Each entry is either the aligned
    reference word (string) or None if this hyp word is an insertion.
    """
    hyp_norm = [normalize_word(w) for w in hyp_words]
    ref_norm = [normalize_word(w) for w in ref_words]

    # Remove empty words (caused by pure-punctuation hyp tokens)
    hyp_indices = [i for i, w in enumerate(hyp_norm) if w]
    hyp_clean = [hyp_norm[i] for i in hyp_indices]
    ref_clean = [w for w in ref_norm if w]

    if not hyp_clean or not ref_clean:
        return [None] * len(hyp_words)

    out_hyp = jiwer.process_words([' '.join(ref_clean)], [' '.join(hyp_clean)])
    alignments = out_hyp.alignments[0]  # list of AlignmentChunk

    # Build mapping: hyp_clean index -> ref_clean index (or None for insertions)
    mapping: dict[int, int | None] = {}
    for chunk in alignments:
        op = chunk.type
        rs, re_ = chunk.ref_start_idx, chunk.ref_end_idx
        hs, he = chunk.hyp_start_idx, chunk.hyp_end_idx
        if op in ('equal', 'substitute'):
            for off in range(he - hs):
                mapping[hs + off] = rs + off if (rs + off) < re_ else None
        elif op == 'insert':
            for off in range(he - hs):
                mapping[hs + off] = None
        # 'delete' affects ref only; no hyp index mapping

    # Now lift back to original hyp_words index
    result: list[str | None] = [None] * len(hyp_words)
    for ci, oi in enumerate(hyp_indices):
        ri = mapping.get(ci)
        if ri is not None and ri < len(ref_clean):
            result[oi] = ref_clean[ri]
    return result


def main() -> None:
    fh = FLAGGED_TSV.open('w')
    fh.write('\t'.join([
        'file_id', 'word_idx', 'hyp_word', 'hyp_norm',
        'left_ctx', 'right_ctx',
        'min_logprob', 'mean_logprob', 'n_subwords',
        'aligned_ref_word', 'parakeet_correct',
    ]) + '\n')
    fh.flush()

    sum_fh = SUMMARY_TSV.open('w')
    sum_fh.write('file_id\tn_hyp_words\tn_flagged\tp_correct_overall\tp_correct_flagged\n')
    sum_fh.flush()

    print(f'Reading {JSONL_IN}', flush=True)
    print(f'Writing spans → {FLAGGED_TSV}', flush=True)
    print(f'Writing summary → {SUMMARY_TSV}', flush=True)
    print(flush=True)

    total_flagged = 0
    total_words = 0
    for line_no, line in enumerate(JSONL_IN.read_text().splitlines(), 1):
        d = json.loads(line)
        file_id = d['file_id']
        tokens = d['tokens']
        logprobs = d['ys_log_probs']
        ref = d['reference']

        words = subwords_to_words(tokens, logprobs)
        if not words:
            print(f'[{line_no:02d}] {file_id}: NO WORDS — skipped', flush=True)
            continue

        hyp_words = [w[0] for w in words]
        ref_words = ref.split()
        aligned = align_hyp_to_ref(hyp_words, ref_words)

        # Determine threshold for bottom BOTTOM_PCT of min_logprob
        mlps = sorted(w[3] for w in words)
        k = max(1, int(len(mlps) * BOTTOM_PCT))
        threshold = mlps[k - 1]  # all words with min_lp <= threshold are flagged

        # Count overall correctness
        correct_overall = sum(
            1 for w, r in zip(hyp_words, aligned)
            if r is not None and normalize_word(w) == r
        )
        p_correct = correct_overall / len(hyp_words)

        # Emit flagged rows
        flagged_in_file = 0
        correct_flagged = 0
        for i, (w_text, s_idx, e_idx, min_lp, mean_lp) in enumerate(words):
            if min_lp > threshold:
                continue
            hyp_norm = normalize_word(w_text)
            aligned_ref = aligned[i]
            correct = 1 if (aligned_ref is not None and hyp_norm == aligned_ref) else 0
            correct_flagged += correct
            left_ctx = ' '.join(hyp_words[max(0, i - CTX_WORDS):i])
            right_ctx = ' '.join(hyp_words[i + 1:i + 1 + CTX_WORDS])
            row = '\t'.join([
                file_id, str(i), w_text, hyp_norm,
                left_ctx, right_ctx,
                f'{min_lp:.4f}', f'{mean_lp:.4f}', str(e_idx - s_idx),
                aligned_ref if aligned_ref else '',
                str(correct),
            ]) + '\n'
            fh.write(row)
            fh.flush()
            flagged_in_file += 1

        p_flagged_correct = correct_flagged / flagged_in_file if flagged_in_file else float('nan')
        sum_row = (
            f'{file_id}\t{len(hyp_words)}\t{flagged_in_file}\t'
            f'{p_correct:.4f}\t{p_flagged_correct:.4f}\n'
        )
        sum_fh.write(sum_row)
        sum_fh.flush()

        total_flagged += flagged_in_file
        total_words += len(hyp_words)
        print(
            f'[{line_no:02d}] {file_id}: words={len(hyp_words)}  '
            f'flagged={flagged_in_file}  p_correct_overall={p_correct:.2%}  '
            f'p_correct_flagged={p_flagged_correct:.2%}',
            flush=True,
        )

    fh.close()
    sum_fh.close()
    print(flush=True)
    print(f'Total: {total_words} hyp words, {total_flagged} flagged ({total_flagged/total_words:.1%})', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
