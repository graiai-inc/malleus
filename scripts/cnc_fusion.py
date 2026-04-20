#!/usr/bin/env python3
"""Confusion Network Combination (Mangu et al. 2000 style, pivot-based).

Correct successor to Fiscus's original ROVER for the 2-or-more-system case.
Handles insertions and deletions via ε (null) arcs in the confusion network,
which is what a plain word-picker approach cannot do.

Algorithm:
    1. Pick the strongest hypothesis as the pivot (parakeet by default).
    2. Initialize a list of slots, one per pivot word:
           slots[i] = {pivot_word_i: pivot_confidence_i}
    3. For each additional system:
         a. Align the system's words against the current slot sequence
            using DP with epsilon support (match / substitute / insert / delete).
         b. For each aligned pair (slot_idx, system_word):
            - match: add system_conf to slots[slot_idx][system_word]
            - substitution: add system's word as a new key at slots[slot_idx]
            - deletion (system says nothing at this slot): add ε vote
              with weight system_mean_top1
            - insertion (system has a word with no matching slot): insert a
              new slot between existing slots with {system_word: system_conf,
              ε: (sum of other systems' mean confidences)}
    4. Decode: for each slot, pick the candidate with the highest total score;
       emit the word if it isn't ε.

Per-file WER is written incrementally as a JSONL.
"""

import argparse
import functools
import json
import logging
import math
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from rover_variants import words_with_confidence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MALLEUS_DIR = SCRIPT_DIR.parent
WNORM = EnglishTextNormalizer()

EPS = '<eps>'  # null arc sentinel


# ----------------------------- alignment --------------------------------


def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


@functools.lru_cache(maxsize=None)
def soft_match(w1: str, w2: str) -> float:
    """Return similarity in [0, 1] between two words based on char edit dist.
    Exact match = 1.0. Used for fuzzy slot matching in alignment."""
    if w1 == w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    dist = levenshtein(w1, w2)
    max_len = max(len(w1), len(w2))
    sim = 1.0 - dist / max_len
    return max(0.0, sim)


def align_to_slots(slot_words: list[str], hyp_words: list[str]) -> list[tuple]:
    """Align a hypothesis word sequence against a sequence of slot 'heads'.

    slot_words[i] is the current best word at slot i (used as the alignment
    representative). hyp_words is the new system's word sequence.

    Returns list of (slot_idx, hyp_idx) with None where either is absent:
        (i, j)       — slot i matches hyp j  (possibly with substitution)
        (i, None)    — hyp deleted at slot i
        (None, j)    — hyp inserted a word with no matching slot
    """
    m = len(slot_words)
    n = len(hyp_words)

    if m == 0:
        return [(None, j) for j in range(n)]
    if n == 0:
        return [(i, None) for i in range(m)]

    # Standard DP. Use negative edit cost so max-path = min-edit-distance.
    # Use soft_match to bias matches toward lower-cost transitions when words
    # are close (helps avoid spurious ε insertions for near-matches).
    NEG_INF = float('-inf')
    dp = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    bt = [[''] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] - 1.0
        bt[i][0] = 'D'
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] - 1.0
        bt[0][j] = 'I'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = soft_match(slot_words[i - 1], hyp_words[j - 1])
            # Match cost: +sim (higher is better). Indel cost: -1.
            match = dp[i - 1][j - 1] + (2.0 * sim - 1.0)  # +1 if exact match, -1 if totally different
            delete = dp[i - 1][j] - 1.0
            insert = dp[i][j - 1] - 1.0
            best = max(match, delete, insert)
            dp[i][j] = best
            if best == match:
                bt[i][j] = 'M'
            elif best == delete:
                bt[i][j] = 'D'
            else:
                bt[i][j] = 'I'

    path = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            path.append((None, j - 1))
            j -= 1
        elif j == 0:
            path.append((i - 1, None))
            i -= 1
        elif bt[i][j] == 'M':
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif bt[i][j] == 'D':
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    return list(reversed(path))


# ------------------------------- CNC core -------------------------------


class Slot:
    __slots__ = ('candidates', 'n_systems_voted')

    def __init__(self):
        self.candidates: dict[str, float] = {}  # word (or EPS) -> accumulated score
        self.n_systems_voted = 0

    def vote(self, word: str, weight: float):
        self.candidates[word] = self.candidates.get(word, 0.0) + weight

    def head(self) -> str:
        """Return the highest-scoring word in the slot (used as alignment rep)."""
        if not self.candidates:
            return EPS
        return max(self.candidates, key=self.candidates.get)

    def decode(self) -> str | None:
        """Return the winning word, or None if ε wins."""
        if not self.candidates:
            return None
        w = self.head()
        return None if w == EPS else w


def build_cnc(
    hyps: list[list[dict]],
    eps_del_scale: float = 1.0,
    eps_ins_scale: float = 1.0,
) -> list[Slot]:
    """Build a confusion network by progressive pivot alignment.

    hyps[0] is the pivot (used as the initial slot sequence).

    Two ε knobs because the two uses of silence pull in opposite directions:

    eps_del_scale
        Weight of non-pivot silence voted AT an existing slot (the non-pivot
        deleted a word the pivot has). High → aggressive insertion removal
        (whisper silence kicks out parakeet spurious words). Risky if the
        pivot was actually right.

    eps_ins_scale
        Weight of pivot silence voted AT a new slot created by a non-pivot
        (the non-pivot inserted a word the pivot missed). Low → aggressive
        deletion recovery (whisper word fills parakeet gap). Risky if the
        non-pivot was hallucinating.
    """
    if not hyps or not hyps[0]:
        return []

    # Initialize slots from pivot.
    slots: list[Slot] = []
    for wd in hyps[0]:
        s = Slot()
        s.vote(wd['word'], wd['top1_mean'])
        s.n_systems_voted = 1
        slots.append(s)

    # Unscaled mean top-1 per system; scales applied at vote time.
    sys_mean = [
        sum(d['top1_mean'] for d in wds) / len(wds) if wds else 0.5
        for wds in hyps
    ]

    # Progressively fold in each additional hypothesis.
    for k in range(1, len(hyps)):
        new_hyp = hyps[k]
        new_words = [d['word'] for d in new_hyp]
        new_confs = [d['top1_mean'] for d in new_hyp]
        # Non-pivot silence voted at an existing slot (insertion-removal ε).
        eps_del = sys_mean[k] * eps_del_scale

        slot_heads = [s.head() for s in slots]
        path = align_to_slots(slot_heads, new_words)

        # Walk the path in order and update slots. We may need to insert new
        # slots for hyp-insertions (None, j). We build a new ordered list.
        new_slots: list[Slot] = []
        i_consumed = 0
        for step in path:
            slot_i, hyp_j = step
            if slot_i is not None and hyp_j is not None:
                # Match or substitution: add hyp's word to this slot.
                while i_consumed < slot_i:
                    new_slots.append(slots[i_consumed])
                    i_consumed += 1
                s = slots[slot_i]
                s.vote(new_words[hyp_j], new_confs[hyp_j])
                s.n_systems_voted += 1
                new_slots.append(s)
                i_consumed = slot_i + 1
            elif slot_i is not None and hyp_j is None:
                # Deletion by new system at this slot: vote for ε.
                while i_consumed < slot_i:
                    new_slots.append(slots[i_consumed])
                    i_consumed += 1
                s = slots[slot_i]
                s.vote(EPS, eps_del)
                s.n_systems_voted += 1
                new_slots.append(s)
                i_consumed = slot_i + 1
            else:
                # Insertion by new system: create a new slot here. All
                # previously-voted systems (incl. pivot) effectively voted
                # silence here — use the deletion-recovery ε scale.
                new_slot = Slot()
                new_slot.vote(new_words[hyp_j], new_confs[hyp_j])
                for m in range(k):
                    new_slot.vote(EPS, sys_mean[m] * eps_ins_scale)
                new_slot.n_systems_voted = k + 1
                new_slots.append(new_slot)
        # Append any trailing slots that weren't consumed.
        while i_consumed < len(slots):
            new_slots.append(slots[i_consumed])
            i_consumed += 1

        slots = new_slots

    return slots


def decode_cn(slots: list[Slot]) -> str:
    out = []
    for s in slots:
        w = s.decode()
        if w is not None:
            out.append(w)
    return ' '.join(out)


# ------------------------------- runner ---------------------------------


def load_results(model_name: str, dataset: str) -> dict:
    path = MALLEUS_DIR / 'results' / dataset / f'{model_name}.jsonl'
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True,
                        help='Comma-separated list, first is pivot')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--eps-del-scale', type=float, default=1.0,
                        help='Scale for non-pivot silence at existing slot '
                        '(insertion-removal ε; higher = more aggressive)')
    parser.add_argument('--eps-ins-scale', type=float, default=1.0,
                        help='Scale for pivot silence at new non-pivot slot '
                        '(deletion-recovery ε; lower = more aggressive)')
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]
    log.info(f'CNC fusion with pivot {model_names[0]} + {model_names[1:]}')

    results = [load_results(m, args.dataset) for m in model_names]
    common = set(results[0].keys())
    for r in results[1:]:
        common &= set(r.keys())
    common = sorted(common)
    log.info(f'{len(common)} common files')

    out_dir = MALLEUS_DIR / 'results' / args.dataset
    tag = '_+_'.join(model_names)
    scale_tag = (
        '' if (args.eps_del_scale == 1.0 and args.eps_ins_scale == 1.0)
        else f'_d{args.eps_del_scale:.2f}_i{args.eps_ins_scale:.2f}'
    )
    per_file_path = out_dir / f'cnc_{tag}{scale_tag}.jsonl'
    summary_path = out_dir / f'cnc_{tag}{scale_tag}.json'
    per_file_path.unlink(missing_ok=True)
    fh = open(per_file_path, 'w')

    totals = {
        'cnc_errors': 0,
        'pivot_errors': 0,
        'words': 0,
    }

    for idx, fid in enumerate(common, 1):
        recs = [r[fid] for r in results]
        wds = [
            words_with_confidence(rec['tokens'], rec['ys_log_probs'], rec['vocab_summaries'])
            for rec in recs
        ]
        if any(not w for w in wds):
            log.warning(f'{fid}: empty hypothesis from one of the models, skipping')
            continue

        slots = build_cnc(
            wds,
            eps_del_scale=args.eps_del_scale,
            eps_ins_scale=args.eps_ins_scale,
        )
        cnc_hyp = decode_cn(slots)
        pivot_hyp = recs[0]['hypothesis']
        ref = recs[0]['reference']

        ref_n = WNORM(ref)
        n_words = len(ref_n.split())
        if n_words == 0:
            continue

        cnc_wer = jiwer.wer(ref_n, WNORM(cnc_hyp))
        pivot_wer = jiwer.wer(ref_n, WNORM(pivot_hyp))

        totals['cnc_errors'] += round(cnc_wer * n_words)
        totals['pivot_errors'] += round(pivot_wer * n_words)
        totals['words'] += n_words

        rec_out = {
            'file_id': fid,
            'n_ref_words': n_words,
            'cnc_wer': round(cnc_wer, 4),
            'pivot_wer': round(pivot_wer, 4),
            'cnc_hyp': cnc_hyp,
            'pivot_hyp': pivot_hyp,
            'reference': ref,
            'n_slots': len(slots),
        }
        fh.write(json.dumps(rec_out) + '\n')
        fh.flush()

        log.info(
            f'[{idx}/{len(common)}] {fid}: '
            f'pivot={pivot_wer:.4f}  cnc={cnc_wer:.4f}  '
            f'slots={len(slots)}  delta={cnc_wer - pivot_wer:+.4f}'
        )

    fh.close()

    agg_cnc = totals['cnc_errors'] / totals['words'] * 100 if totals['words'] else 0
    agg_pivot = totals['pivot_errors'] / totals['words'] * 100 if totals['words'] else 0

    summary = {
        'models': model_names,
        'dataset': args.dataset,
        'n_files': len(common),
        'aggregate_pivot_wer_pct': round(agg_pivot, 2),
        'aggregate_cnc_wer_pct': round(agg_cnc, 2),
        'delta_pp': round(agg_cnc - agg_pivot, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f'\n{"=" * 60}')
    print(f'CNC fusion: {model_names} on {args.dataset}')
    print(f'{"=" * 60}')
    print(f'  Pivot ({model_names[0]}) WER: {agg_pivot:.2f}%')
    print(f'  CNC fused WER              : {agg_cnc:.2f}%')
    print(f'  Delta                      : {agg_cnc - agg_pivot:+.2f}pp')
    print(f'\nSaved to {per_file_path}')


if __name__ == '__main__':
    main()
