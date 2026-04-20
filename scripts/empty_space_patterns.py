#!/usr/bin/env python3
"""Feature-by-feature conditional analysis of 'take-the-word' correctness.

For every Case B event (whisper has word, parakeet silent at that ref slot),
extract a pile of features and report correctness rate vs 'take the word'
per feature bin. Same for Case A and for pure insertions.

The goal: find features that split events into high-correct-rate buckets
(where a local rule can confidently take the word) vs low-correct-rate
buckets (where it should reject). Features with large spread across bins
carry signal for a local gate.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rover_variants import words_with_confidence  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'primock57'
W = EnglishTextNormalizer()

FILLERS = {'um', 'uh', 'mm', 'mhm', 'hmm', 'oh', 'ah', 'er', 'huh',
           'yeah', 'yep', 'yes', 'no', 'ok', 'okay', 'hi', 'hello'}
FUNCTION = {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on',
            'at', 'by', 'for', 'with', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'am', 'do', 'does', 'did', 'have', 'has',
            'had', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
            'their', 'this', 'that', 'these', 'those', 'so', 'if', 'as',
            'than', 'then', 'there', 'here', 'when', 'where', 'what',
            'which', 'who', 'how', 'why', 'not', 'no'}


def load(m):
    out = {}
    with open(RESULTS / f'{m}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


def per_ref_align(ref_words, hyp_words):
    """Return list of length len(ref): each entry is
    (hyp_word_or_None, hyp_index_or_None). Plus list of pure-insertion
    hyp indices (not aligned to any ref)."""
    out = [(None, None)] * len(ref_words)
    inserted = []
    if not ref_words and not hyp_words:
        return out, inserted
    chunks = jiwer.process_words(
        ' '.join(ref_words), ' '.join(hyp_words)
    ).alignments[0]
    for ch in chunks:
        if ch.type == 'equal':
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                j = ch.hyp_start_idx + k
                out[ch.ref_start_idx + k] = (hyp_words[j], j)
        elif ch.type == 'substitute':
            r_len = ch.ref_end_idx - ch.ref_start_idx
            h_len = ch.hyp_end_idx - ch.hyp_start_idx
            for k in range(min(r_len, h_len)):
                j = ch.hyp_start_idx + k
                out[ch.ref_start_idx + k] = (hyp_words[j], j)
        elif ch.type == 'insert':
            for k in range(ch.hyp_end_idx - ch.hyp_start_idx):
                inserted.append(ch.hyp_start_idx + k)
    return out, inserted


def categorize_word(w: str) -> str:
    wl = w.lower()
    if wl in FILLERS:
        return 'filler'
    if wl in FUNCTION:
        return 'function'
    if len(wl) <= 3:
        return 'short'
    if len(wl) >= 8:
        return 'long_content'
    return 'mid_content'


def length_bin(n: int) -> str:
    if n <= 2:
        return '1-2'
    if n <= 4:
        return '3-4'
    if n <= 6:
        return '5-6'
    if n <= 9:
        return '7-9'
    return '10+'


def conf_bin(x: float) -> str:
    if x < 0.5:
        return '<.50'
    if x < 0.70:
        return '.50-.70'
    if x < 0.85:
        return '.70-.85'
    if x < 0.95:
        return '.85-.95'
    return '.95+'


def report(title: str, events: list[dict], feature_names: list[str]):
    print(f'\n{"=" * 72}')
    print(title)
    print('=' * 72)
    total = len(events)
    if total == 0:
        print('  (no events)')
        return
    base_right = sum(e['take_word_right'] for e in events) / total
    print(f'N events: {total}    baseline take-word-right: {base_right*100:.1f}%\n')
    for feat in feature_names:
        buckets = defaultdict(lambda: [0, 0])
        for e in events:
            v = e.get(feat, '(missing)')
            if v is None:
                v = '(None)'
            buckets[v][0] += 1
            if e['take_word_right']:
                buckets[v][1] += 1
        keys = sorted(buckets.keys(), key=lambda k: -buckets[k][0])
        rates = []
        for k in keys:
            n, r = buckets[k]
            if n < 20:
                continue
            rates.append((k, n, r / n * 100))
        if not rates:
            continue
        lo = min(r[2] for r in rates)
        hi = max(r[2] for r in rates)
        spread = hi - lo
        marker = '*' if spread >= 10 else ' '
        print(f'  [{marker}] {feat}  (spread {spread:5.1f}pp)')
        for k, n, rate in rates:
            delta = rate - base_right * 100
            print(f'       {str(k):<16} n={n:>5}  right={rate:5.1f}%  '
                  f'({delta:+5.1f}pp vs base)')
    print()


def main():
    p = load('parakeet-tdt-0.6b-v2')
    w = load('whisper-distil-v3.5')

    caseA_events: list[dict] = []
    caseB_events: list[dict] = []
    whisper_ins_events: list[dict] = []  # pure insertions

    for fid in sorted(p.keys() & w.keys()):
        pr = p[fid]
        wr = w[fid]
        ref = W(pr['reference'])
        hp = W(pr['hypothesis']).split()
        hw = W(wr['hypothesis']).split()
        ref_words = ref.split()

        pw_align, p_ins = per_ref_align(ref_words, hp)
        ww_align, w_ins = per_ref_align(ref_words, hw)

        # Per-word confidence tables (for hp / hw order)
        p_wds = words_with_confidence(
            pr['tokens'], pr['ys_log_probs'], pr['vocab_summaries']
        )
        w_wds = words_with_confidence(
            wr['tokens'], wr['ys_log_probs'], wr['vocab_summaries']
        )
        # Normalize via WNORM split so indices align to hp/hw split
        # (words_with_confidence strips punctuation already, but WNORM may
        # split or merge — match by position-in-order as a best-effort).
        p_conf = [d.get('top1_mean', 0.0) for d in p_wds]
        w_conf = [d.get('top1_mean', 0.0) for d in w_wds]
        p_shan = [d.get('shannon_mean', 0.0) for d in p_wds]
        w_shan = [d.get('shannon_mean', 0.0) for d in w_wds]
        # Truncate/pad to match hp/hw lengths
        def at(arr, i, default=0.0):
            return arr[i] if 0 <= i < len(arr) else default

        # --- Case A: parakeet word, whisper silent at ref slot i ---
        for i, rw in enumerate(ref_words):
            p_word, p_hi = pw_align[i]
            w_word, w_hi = ww_align[i]
            if p_word is not None and w_word is None:
                right = (p_word == rw)
                # parakeet's own confidence on this word
                p_c = at(p_conf, p_hi if p_hi is not None else -1, 0.0)
                # whisper's neighbor context confidences
                # find nearest whisper word index in the reference vicinity
                left_j = None
                for k in range(i - 1, -1, -1):
                    if ww_align[k][1] is not None:
                        left_j = ww_align[k][1]
                        break
                right_j = None
                for k in range(i + 1, len(ref_words)):
                    if ww_align[k][1] is not None:
                        right_j = ww_align[k][1]
                        break
                w_left_c = at(w_conf, left_j, 0.0) if left_j is not None else 0.0
                w_right_c = at(w_conf, right_j, 0.0) if right_j is not None else 0.0

                caseA_events.append({
                    'take_word_right': right,
                    'word_type': categorize_word(p_word),
                    'word_len': length_bin(len(p_word)),
                    'p_conf': conf_bin(p_c),
                    'w_left_conf': conf_bin(w_left_c),
                    'w_right_conf': conf_bin(w_right_c),
                    'w_ctx_max': conf_bin(max(w_left_c, w_right_c)),
                    'pos_frac': (
                        'start' if i < len(ref_words) * 0.1
                        else 'end' if i > len(ref_words) * 0.9
                        else 'mid'
                    ),
                })

            elif p_word is None and w_word is not None:
                right = (w_word == rw)
                w_c = at(w_conf, w_hi if w_hi is not None else -1, 0.0)
                w_s = at(w_shan, w_hi if w_hi is not None else -1, 0.0)
                left_j = None
                for k in range(i - 1, -1, -1):
                    if pw_align[k][1] is not None:
                        left_j = pw_align[k][1]
                        break
                right_j = None
                for k in range(i + 1, len(ref_words)):
                    if pw_align[k][1] is not None:
                        right_j = pw_align[k][1]
                        break
                p_left_c = at(p_conf, left_j, 0.0) if left_j is not None else 0.0
                p_right_c = at(p_conf, right_j, 0.0) if right_j is not None else 0.0

                caseB_events.append({
                    'take_word_right': right,
                    'word_type': categorize_word(w_word),
                    'word_len': length_bin(len(w_word)),
                    'w_conf': conf_bin(w_c),
                    'w_shan': conf_bin(1.0 - min(w_s / 3.0, 1.0)),
                    'p_left_conf': conf_bin(p_left_c),
                    'p_right_conf': conf_bin(p_right_c),
                    'p_ctx_min': conf_bin(min(p_left_c, p_right_c)),
                    'p_ctx_max': conf_bin(max(p_left_c, p_right_c)),
                    'pos_frac': (
                        'start' if i < len(ref_words) * 0.1
                        else 'end' if i > len(ref_words) * 0.9
                        else 'mid'
                    ),
                })

        # Whisper pure insertions: word in hw[j] that maps to no ref slot.
        # These are always wrong (take_word_right = False). Record feature
        # distributions so we can compare to Case B's right events.
        for j in w_ins:
            if j >= len(hw):
                continue
            w_c = at(w_conf, j, 0.0)
            whisper_ins_events.append({
                'take_word_right': False,
                'word_type': categorize_word(hw[j]),
                'word_len': length_bin(len(hw[j])),
                'w_conf': conf_bin(w_c),
            })

    report(
        'CASE A — parakeet has word, whisper silent  (take word = parakeet)',
        caseA_events,
        ['word_type', 'word_len', 'p_conf', 'w_ctx_max', 'pos_frac'],
    )
    report(
        'CASE B — whisper has word, parakeet silent  (take word = whisper)',
        caseB_events,
        ['word_type', 'word_len', 'w_conf', 'w_shan',
         'p_left_conf', 'p_right_conf', 'p_ctx_min', 'p_ctx_max', 'pos_frac'],
    )
    print('\n=== whisper pure insertions (always wrong) vs Case B ===')
    print('If whisper insertions and Case B rights have very different feature')
    print('distributions, a feature-based gate can separate them.\n')
    # Compare conf distribution
    from collections import Counter
    caseB_right = [e for e in caseB_events if e['take_word_right']]
    bin_counts_right = Counter(e['w_conf'] for e in caseB_right)
    bin_counts_ins = Counter(e['w_conf'] for e in whisper_ins_events)
    all_bins = sorted(set(bin_counts_right) | set(bin_counts_ins))
    total_r = sum(bin_counts_right.values()) or 1
    total_i = sum(bin_counts_ins.values()) or 1
    print(f'{"w_conf bin":<12} {"caseB-right %":>14} {"whisper-ins %":>14}')
    for b in all_bins:
        r = bin_counts_right[b] / total_r * 100
        i = bin_counts_ins[b] / total_i * 100
        print(f'{b:<12} {r:>13.1f}% {i:>13.1f}%')

    print()
    bin_counts_right = Counter(e['word_type'] for e in caseB_right)
    bin_counts_ins = Counter(e['word_type'] for e in whisper_ins_events)
    all_bins = sorted(set(bin_counts_right) | set(bin_counts_ins))
    print(f'{"word_type":<14} {"caseB-right %":>14} {"whisper-ins %":>14}')
    for b in all_bins:
        r = bin_counts_right[b] / total_r * 100
        i = bin_counts_ins[b] / total_i * 100
        print(f'{b:<14} {r:>13.1f}% {i:>13.1f}%')


if __name__ == '__main__':
    main()
