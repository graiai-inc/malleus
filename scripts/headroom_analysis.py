#!/usr/bin/env python3
"""Headroom analysis: where does the oracle→shannon gap live?

After fixing the apostrophe-stripping bug, ROVER shannon-gated = 13.30%,
parakeet solo = 13.76%, oracle = 9.07%. This script decomposes the 4.23pp
gap between shannon and oracle.

Questions answered:
    1. Of parakeet's errors, what fraction are S/D/I (after WNORM)?
    2. Of those errors, how many does whisper "know the answer" to?
       (i.e. at that position in the reference, whisper is correct)
    3. Of parakeet's CORRECT words, how often is whisper WRONG?
       (these are slots the fusion must protect from whisper noise)
    4. What is the TRUE oracle (per-word pick) vs the ROVER oracle
       (which is constrained to aligned slot picks)?
"""

import json
import sys
from collections import Counter
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR.parent / 'results' / 'primock57'
W = EnglishTextNormalizer()


def load(model: str) -> dict:
    out = {}
    with open(RESULTS / f'{model}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


def align_to_ref(ref_words: list[str], hyp_words: list[str]) -> list[str | None]:
    """For each reference word, return the hypothesis word that jiwer aligned
    to it (via 'equal' or 'substitute'), or None if it was deleted.
    Ignores insertions (hyp words with no ref position)."""
    out: list[str | None] = [None] * len(ref_words)
    # jiwer chunks: equal / substitute / insert / delete
    result = jiwer.process_words(
        ' '.join(ref_words) if ref_words else '',
        ' '.join(hyp_words) if hyp_words else '',
    )
    for ch in result.alignments[0]:
        t = ch.type
        if t == 'equal':
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                out[ch.ref_start_idx + k] = hyp_words[ch.hyp_start_idx + k]
        elif t == 'substitute':
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                j = ch.hyp_start_idx + k
                if j < ch.hyp_end_idx:
                    out[ch.ref_start_idx + k] = hyp_words[j]
        # insert/delete: no ref→hyp mapping
    return out


def main():
    p = load('parakeet-tdt-0.6b-v2')
    w = load('whisper-distil-v3.5')
    s = load('sensevoice-no-itn')

    common = sorted(p.keys() & w.keys() & s.keys())
    print(f'{len(common)} common files\n')

    # Category counts (all against ref, after WNORM)
    total_ref = 0
    p_err = Counter()  # type → count
    # For parakeet-wrong slots, is whisper correct?
    p_wrong_w_right = Counter()
    p_wrong_w_wrong_s_right = Counter()
    p_wrong_all_wrong = Counter()
    # For parakeet-correct slots, how often does whisper/sensevoice differ (risk)
    p_right_w_wrong = 0
    p_right_s_wrong = 0
    p_right_any_wrong = 0
    p_right = 0

    # Per-ref-word 3-way oracle: reference word "recoverable" if any model has
    # it at the aligned position.
    oracle_2way = 0  # recoverable by parakeet OR whisper
    oracle_3way = 0  # recoverable by any of 3

    for fid in common:
        ref = W(p[fid]['reference'])
        hp = W(p[fid]['hypothesis'])
        hw = W(w[fid]['hypothesis'])
        hs = W(s[fid]['hypothesis'])
        ref_words = ref.split()
        total_ref += len(ref_words)

        pw = align_to_ref(ref_words, hp.split())
        ww = align_to_ref(ref_words, hw.split())
        sw = align_to_ref(ref_words, hs.split())

        # Parakeet error chunks (for S/D breakdown)
        result = jiwer.process_words(ref, hp)
        for ch in result.alignments[0]:
            if ch.type == 'equal':
                continue
            if ch.type == 'insert':
                p_err['I'] += ch.hyp_end_idx - ch.hyp_start_idx
            elif ch.type == 'delete':
                p_err['D'] += ch.ref_end_idx - ch.ref_start_idx
            elif ch.type == 'substitute':
                p_err['S'] += ch.ref_end_idx - ch.ref_start_idx

        # Per-ref-word recoverability
        for i, rw in enumerate(ref_words):
            p_ok = pw[i] == rw
            w_ok = ww[i] == rw
            s_ok = sw[i] == rw

            if p_ok:
                p_right += 1
                if not w_ok:
                    p_right_w_wrong += 1
                if not s_ok:
                    p_right_s_wrong += 1
                if not w_ok or not s_ok:
                    p_right_any_wrong += 1
            else:
                # Parakeet got this ref word wrong (sub or del)
                # Was it a sub (pw[i] is some other word) or del (pw[i] is None)?
                was_del = pw[i] is None
                key = 'del' if was_del else 'sub'
                if w_ok:
                    p_wrong_w_right[f'W_only_{key}'] += 1
                elif s_ok:
                    p_wrong_w_wrong_s_right[f'S_only_{key}'] += 1
                else:
                    p_wrong_all_wrong[f'none_{key}'] += 1

            if p_ok or w_ok:
                oracle_2way += 1
            if p_ok or w_ok or s_ok:
                oracle_3way += 1

    print('=== Parakeet solo error breakdown (of total ref words) ===')
    print(f'total ref words: {total_ref}')
    total_p_err = p_err['S'] + p_err['D'] + p_err['I']
    print(f'parakeet total errors: {total_p_err} ({total_p_err/total_ref*100:.2f}%)')
    print(f'  substitutions: {p_err["S"]} ({p_err["S"]/total_ref*100:.2f}%)')
    print(f'  deletions    : {p_err["D"]} ({p_err["D"]/total_ref*100:.2f}%)')
    print(f'  insertions   : {p_err["I"]} ({p_err["I"]/total_ref*100:.2f}%)')
    print()

    print('=== Per-ref-word recoverability (substitutions + deletions only) ===')
    p_wrong_count = total_ref - p_right
    print(f'parakeet right     : {p_right} ({p_right/total_ref*100:.2f}%)')
    print(f'parakeet wrong     : {p_wrong_count} ({p_wrong_count/total_ref*100:.2f}%)')
    w_sub = p_wrong_w_right['W_only_sub']
    w_del = p_wrong_w_right['W_only_del']
    print(f'  whisper rescues (parakeet sub): {w_sub} ({w_sub/total_ref*100:.2f}%)')
    print(f'  whisper rescues (parakeet del): {w_del} ({w_del/total_ref*100:.2f}%)')
    s_sub = p_wrong_w_wrong_s_right['S_only_sub']
    s_del = p_wrong_w_wrong_s_right['S_only_del']
    print(f'  sensevoice-only rescues (sub) : {s_sub} ({s_sub/total_ref*100:.2f}%)')
    print(f'  sensevoice-only rescues (del) : {s_del} ({s_del/total_ref*100:.2f}%)')
    n_sub = p_wrong_all_wrong['none_sub']
    n_del = p_wrong_all_wrong['none_del']
    print(f'  all three wrong (sub)         : {n_sub} ({n_sub/total_ref*100:.2f}%)')
    print(f'  all three wrong (del)         : {n_del} ({n_del/total_ref*100:.2f}%)')
    print()

    print('=== Risk on parakeet-correct slots (fusion must not break these) ===')
    print(f'parakeet right slots where whisper disagrees   : '
          f'{p_right_w_wrong} ({p_right_w_wrong/total_ref*100:.2f}%)')
    print(f'parakeet right slots where sensevoice disagrees: '
          f'{p_right_s_wrong} ({p_right_s_wrong/total_ref*100:.2f}%)')
    print(f'parakeet right slots where ANY other disagrees : '
          f'{p_right_any_wrong} ({p_right_any_wrong/total_ref*100:.2f}%)')
    print()

    print('=== Oracles (per-ref-word picker, alignment-based) ===')
    print(f'parakeet+whisper per-word oracle: '
          f'{(total_ref - oracle_2way)/total_ref*100:.2f}% WER '
          f'(recoverable: {oracle_2way}/{total_ref})')
    print(f'parakeet+whisper+sensevoice    : '
          f'{(total_ref - oracle_3way)/total_ref*100:.2f}% WER '
          f'(recoverable: {oracle_3way}/{total_ref})')
    print()
    print('NOTE: alignment-based oracles ignore insertions. The 9.07% "ROVER '
          'oracle" from rover_variants.py includes insertion penalty.')


if __name__ == '__main__':
    main()
