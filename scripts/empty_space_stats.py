#!/usr/bin/env python3
"""For every 'one has word, other silent' event, measure: should we take it?

Two event types, based on jiwer per-ref-word alignment of each model to ref:

    A. parakeet has a word, whisper is silent at that ref position
       - Right answer = take parakeet's word IF parakeet's word == ref word
       - Otherwise the right answer is still "silence" (parakeet sub is wrong
         too, but choosing silence would at least avoid doubling the error)

    B. whisper has a word, parakeet is silent

Plus the two "pure-insertion" buckets (words in hyp that align to no ref slot):

    C. parakeet inserts a word (no ref alignment), whisper silent
       - Right answer: always "drop" (inserted word = guaranteed error)
    D. whisper inserts a word, parakeet silent
       - Right answer: always "drop"
"""

import json
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'primock57'
W = EnglishTextNormalizer()


def load(model: str) -> dict:
    out = {}
    with open(RESULTS / f'{model}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


def align_pair(ref: str, hyp: str):
    """Return list of chunks with types equal/substitute/insert/delete."""
    if not ref or not hyp:
        return []
    return jiwer.process_words(ref, hyp).alignments[0]


def per_ref_alignment(ref_words, hyp_words):
    """For each ref index, return hyp word aligned to it (or None if deleted).
    Also return the set of hyp indices that were pure insertions (not aligned
    to any ref position)."""
    out = [None] * len(ref_words)
    inserted_hyp_idxs = []
    chunks = jiwer.process_words(
        ' '.join(ref_words) if ref_words else '',
        ' '.join(hyp_words) if hyp_words else '',
    ).alignments[0]
    for ch in chunks:
        if ch.type == 'equal':
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                out[ch.ref_start_idx + k] = hyp_words[ch.hyp_start_idx + k]
        elif ch.type == 'substitute':
            r_len = ch.ref_end_idx - ch.ref_start_idx
            h_len = ch.hyp_end_idx - ch.hyp_start_idx
            for k in range(min(r_len, h_len)):
                out[ch.ref_start_idx + k] = hyp_words[ch.hyp_start_idx + k]
        elif ch.type == 'insert':
            for k in range(ch.hyp_end_idx - ch.hyp_start_idx):
                inserted_hyp_idxs.append(ch.hyp_start_idx + k)
        # delete: ref words with no hyp — already None
    return out, inserted_hyp_idxs


def main():
    p = load('parakeet-tdt-0.6b-v2')
    w = load('whisper-distil-v3.5')
    common = sorted(p.keys() & w.keys())

    # At each ref position:
    caseA_take_word_right = 0   # parakeet has word, whisper None, p word == ref
    caseA_take_silence_right = 0  # parakeet has word, whisper None, p word != ref
    caseB_take_word_right = 0   # whisper has word, parakeet None, w word == ref
    caseB_take_silence_right = 0  # whisper has word, parakeet None, w word != ref
    both_word = 0
    both_silent = 0

    # Pure insertions (word in hyp, no ref slot at all):
    p_insertions = 0
    w_insertions = 0

    for fid in common:
        ref = W(p[fid]['reference'])
        hp = W(p[fid]['hypothesis']).split()
        hw = W(w[fid]['hypothesis']).split()
        ref_words = ref.split()

        pw, p_ins = per_ref_alignment(ref_words, hp)
        ww, w_ins = per_ref_alignment(ref_words, hw)
        p_insertions += len(p_ins)
        w_insertions += len(w_ins)

        for i, rw in enumerate(ref_words):
            pw_i = pw[i]
            ww_i = ww[i]
            if pw_i is not None and ww_i is None:
                if pw_i == rw:
                    caseA_take_word_right += 1
                else:
                    caseA_take_silence_right += 1
            elif pw_i is None and ww_i is not None:
                if ww_i == rw:
                    caseB_take_word_right += 1
                else:
                    caseB_take_silence_right += 1
            elif pw_i is not None and ww_i is not None:
                both_word += 1
            else:
                both_silent += 1

    tot_A = caseA_take_word_right + caseA_take_silence_right
    tot_B = caseB_take_word_right + caseB_take_silence_right

    print('=== "one has word, other silent" decisions ===\n')
    print(f'Case A — parakeet has word, whisper silent   (N = {tot_A})')
    if tot_A:
        print(f'  take word    (parakeet correct)  : {caseA_take_word_right:>5}  '
              f'({caseA_take_word_right/tot_A*100:.1f}%)')
        print(f'  take silence (parakeet wrong)    : {caseA_take_silence_right:>5}  '
              f'({caseA_take_silence_right/tot_A*100:.1f}%)')
    print()
    print(f'Case B — whisper has word, parakeet silent   (N = {tot_B})')
    if tot_B:
        print(f'  take word    (whisper correct)   : {caseB_take_word_right:>5}  '
              f'({caseB_take_word_right/tot_B*100:.1f}%)')
        print(f'  take silence (whisper wrong)     : {caseB_take_silence_right:>5}  '
              f'({caseB_take_silence_right/tot_B*100:.1f}%)')
    print()
    print(f'Both have word : {both_word}')
    print(f'Both silent    : {both_silent}')
    print()
    print(f'Pure parakeet insertions (no ref slot): {p_insertions} — always wrong')
    print(f'Pure whisper  insertions (no ref slot): {w_insertions} — always wrong')
    print()
    print('Interpretation: if the fraction "take word" is >> 50%, a dumb rule')
    print('"always take the word" wins at that empty-space slot. If << 50%,')
    print('"always stay silent" wins. The closer to 50%, the more a local')
    print('signal (not a global ε) is needed to decide.')


if __name__ == '__main__':
    main()
