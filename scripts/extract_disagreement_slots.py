#!/usr/bin/env python3
"""Extract disagreement slots between two ASR models, with features + labels.

For each file in a dataset, align the two hypotheses, find every slot where
both models produced a word but the words differ. For each such slot:

  * compute reference-free features (confidence, entropy, margin, burst,
    neighbor agreement, phonetic/edit distance, file-global stats, ...)
  * compute the LABEL by aligning each hypothesis independently to the
    reference and checking whether this word was a 'match' or not:
        m1_only   — m1 correct, m2 wrong
        m2_only   — m2 correct, m1 wrong
        both      — both correct (should be rare; normalizer edge cases)
        neither   — both wrong (oracle-unreachable)

Writes one CSV row per disagreement slot, incrementally (per-file flush).
This gives us the labeled dataset for training a 'which model to trust'
classifier and for feature-importance analysis.
"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import jellyfish
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from rover_variants import align_sequences, words_with_confidence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MALLEUS_DIR = SCRIPT_DIR.parent
WNORM = EnglishTextNormalizer()

# Model vocab sizes, used to normalize entropy onto a comparable scale.
# Each entry is the size used for log(V) — the upper bound of Shannon in nats.
VOCAB_SIZES = {
    'parakeet-tdt-0.6b-v2': 1025,
    'whisper-distil-v3.5': 51865,
    'sensevoice-no-itn': 25055,
    'qwen3-asr': 152064,
    'whisper-turbo': 51865,
}


# ----------------------------- helpers ------------------------------------


def load_results(model_name: str, dataset: str) -> dict:
    path = MALLEUS_DIR / 'results' / dataset / f'{model_name}.jsonl'
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


def correctness_mask(ref_text: str, hyp_words: list[str]) -> list[bool]:
    """Per-hyp-word: was this word a match in the alignment with reference?

    Uses jiwer's word-level alignment. Both inputs are expected to be
    whitespace-tokenized strings; ref is normalized with WNORM. We pass
    hypothesis as space-joined lowercased basic-stripped words (the
    same normalization used in words_with_confidence)."""
    hyp_text = ' '.join(hyp_words)
    if not hyp_text.strip():
        return []
    out = jiwer.process_words(WNORM(ref_text), hyp_text)
    mask = [False] * len(hyp_words)
    for chunk in out.alignments[0]:
        if chunk.type == 'equal':
            for k in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                idx = chunk.hyp_start_idx + k
                if 0 <= idx < len(mask):
                    mask[idx] = True
    return mask


def neighbor_agree(alignment: list[tuple], pos: int, offset: int) -> int:
    """Is the slot `pos + offset` in the alignment an agreement (same word,
    both present)? Returns 1 for agree, 0 for disagree/missing/out-of-bounds.
    """
    target = pos + offset
    if target < 0 or target >= len(alignment):
        return 0
    w1, w2, _, _ = alignment[target]
    if w1 is not None and w2 is not None and w1 == w2:
        return 1
    return 0


def burst_size(disagreement_flags: list[bool], pos: int) -> int:
    """Length of the contiguous disagreement run containing position `pos`."""
    if not disagreement_flags[pos]:
        return 0
    left = pos
    while left > 0 and disagreement_flags[left - 1]:
        left -= 1
    right = pos
    while right < len(disagreement_flags) - 1 and disagreement_flags[right + 1]:
        right += 1
    return right - left + 1


def safe_z(value: float, mean: float, std: float) -> float:
    if std < 1e-9:
        return 0.0
    return (value - mean) / std


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


# ----------------------------- main work ---------------------------------


FIELDNAMES = [
    'file_id', 'slot_idx',
    'word_m1', 'word_m2',
    # confidence
    'm1_top1_mean', 'm2_top1_mean',
    'm1_top1_min', 'm2_top1_min',
    'm1_margin_min', 'm2_margin_min',
    'm1_logprob_mean', 'm2_logprob_mean',
    'm1_logprob_min', 'm2_logprob_min',
    # entropy (raw + normalized by log(vocab_size))
    'm1_shannon_max', 'm2_shannon_max',
    'm1_shannon_mean', 'm2_shannon_mean',
    'm1_tsallis_max', 'm2_tsallis_max',
    'm1_shannon_norm_max', 'm2_shannon_norm_max',
    # subword structure
    'm1_n_subwords', 'm2_n_subwords',
    # self-normalized (z-score relative to this file's mean for that model)
    'm1_top1_z', 'm2_top1_z',
    'm1_shannon_z', 'm2_shannon_z',
    'm1_logprob_z', 'm2_logprob_z',
    # candidate-word structural features
    'char_edit_dist', 'char_edit_ratio',
    'phonetic_edit_dist', 'phonetic_match',
    'char_len_m1', 'char_len_m2',
    # context / alignment
    'burst_size',
    'left_agree_1', 'right_agree_1',
    'left_agree_2', 'right_agree_2',
    'left_agree_3', 'right_agree_3',
    # file-global signals (same value on every row from a given file)
    'file_disagreement_rate',
    'file_m1_mean_logprob', 'file_m2_mean_logprob',
    'file_m1_mean_shannon', 'file_m2_mean_shannon',
    # labels
    'm1_correct', 'm2_correct', 'label',
]


def extract_for_file(fid, d1, d2, model1, model2, writer, fh):
    ref = d1['reference']
    wd1 = words_with_confidence(d1['tokens'], d1['ys_log_probs'], d1['vocab_summaries'])
    wd2 = words_with_confidence(d2['tokens'], d2['ys_log_probs'], d2['vocab_summaries'])
    words1 = [d['word'] for d in wd1]
    words2 = [d['word'] for d in wd2]
    if not words1 or not words2:
        return 0, 0

    alignment = align_sequences(words1, words2)
    n_slots = len(alignment)

    # Correctness masks per model against reference.
    mask1 = correctness_mask(ref, words1)
    mask2 = correctness_mask(ref, words2)

    # File-global stats per model for z-scoring.
    m1_top1s = [d['top1_mean'] for d in wd1]
    m2_top1s = [d['top1_mean'] for d in wd2]
    m1_shannons = [d['shannon_mean'] for d in wd1]
    m2_shannons = [d['shannon_mean'] for d in wd2]
    m1_logprobs = [d['logprob_mean'] for d in wd1]
    m2_logprobs = [d['logprob_mean'] for d in wd2]
    m1_top1_mu, m1_top1_sd = mean_std(m1_top1s)
    m2_top1_mu, m2_top1_sd = mean_std(m2_top1s)
    m1_sh_mu, m1_sh_sd = mean_std(m1_shannons)
    m2_sh_mu, m2_sh_sd = mean_std(m2_shannons)
    m1_lp_mu, m1_lp_sd = mean_std(m1_logprobs)
    m2_lp_mu, m2_lp_sd = mean_std(m2_logprobs)

    disagreement_flags = [
        (w1 is not None and w2 is not None and w1 != w2)
        for w1, w2, _, _ in alignment
    ]
    n_disagree = sum(disagreement_flags)
    file_disagree_rate = n_disagree / n_slots if n_slots else 0.0

    log_v1 = math.log(VOCAB_SIZES.get(model1, 1025))
    log_v2 = math.log(VOCAB_SIZES.get(model2, 51865))

    n_written = 0
    for pos, (w1, w2, i1, i2) in enumerate(alignment):
        if not disagreement_flags[pos]:
            continue  # only write disagreement rows

        d_1 = wd1[i1]
        d_2 = wd2[i2]

        # Phonetic / edit features
        char_ed = jellyfish.levenshtein_distance(w1, w2)
        max_char = max(len(w1), len(w2)) or 1
        char_er = char_ed / max_char
        meta1 = jellyfish.metaphone(w1) or ''
        meta2 = jellyfish.metaphone(w2) or ''
        phon_ed = jellyfish.levenshtein_distance(meta1, meta2)
        phon_match = 1 if (meta1 == meta2 and meta1 != '') else 0

        row = {
            'file_id': fid,
            'slot_idx': pos,
            'word_m1': w1,
            'word_m2': w2,
            'm1_top1_mean': round(d_1['top1_mean'], 6),
            'm2_top1_mean': round(d_2['top1_mean'], 6),
            'm1_top1_min': round(d_1['top1_min'], 6),
            'm2_top1_min': round(d_2['top1_min'], 6),
            'm1_margin_min': round(d_1['margin_min'], 6),
            'm2_margin_min': round(d_2['margin_min'], 6),
            'm1_logprob_mean': round(d_1['logprob_mean'], 6),
            'm2_logprob_mean': round(d_2['logprob_mean'], 6),
            'm1_logprob_min': round(d_1['logprob_min'], 6),
            'm2_logprob_min': round(d_2['logprob_min'], 6),
            'm1_shannon_max': round(d_1['shannon_max'], 6),
            'm2_shannon_max': round(d_2['shannon_max'], 6),
            'm1_shannon_mean': round(d_1['shannon_mean'], 6),
            'm2_shannon_mean': round(d_2['shannon_mean'], 6),
            'm1_tsallis_max': round(d_1['tsallis_max'], 6),
            'm2_tsallis_max': round(d_2['tsallis_max'], 6),
            'm1_shannon_norm_max': round(d_1['shannon_max'] / log_v1, 6),
            'm2_shannon_norm_max': round(d_2['shannon_max'] / log_v2, 6),
            'm1_n_subwords': d_1['n_subwords'],
            'm2_n_subwords': d_2['n_subwords'],
            'm1_top1_z': round(safe_z(d_1['top1_mean'], m1_top1_mu, m1_top1_sd), 4),
            'm2_top1_z': round(safe_z(d_2['top1_mean'], m2_top1_mu, m2_top1_sd), 4),
            'm1_shannon_z': round(safe_z(d_1['shannon_mean'], m1_sh_mu, m1_sh_sd), 4),
            'm2_shannon_z': round(safe_z(d_2['shannon_mean'], m2_sh_mu, m2_sh_sd), 4),
            'm1_logprob_z': round(safe_z(d_1['logprob_mean'], m1_lp_mu, m1_lp_sd), 4),
            'm2_logprob_z': round(safe_z(d_2['logprob_mean'], m2_lp_mu, m2_lp_sd), 4),
            'char_edit_dist': char_ed,
            'char_edit_ratio': round(char_er, 4),
            'phonetic_edit_dist': phon_ed,
            'phonetic_match': phon_match,
            'char_len_m1': len(w1),
            'char_len_m2': len(w2),
            'burst_size': burst_size(disagreement_flags, pos),
            'left_agree_1': neighbor_agree(alignment, pos, -1),
            'right_agree_1': neighbor_agree(alignment, pos, +1),
            'left_agree_2': neighbor_agree(alignment, pos, -2),
            'right_agree_2': neighbor_agree(alignment, pos, +2),
            'left_agree_3': neighbor_agree(alignment, pos, -3),
            'right_agree_3': neighbor_agree(alignment, pos, +3),
            'file_disagreement_rate': round(file_disagree_rate, 4),
            'file_m1_mean_logprob': round(m1_lp_mu, 4),
            'file_m2_mean_logprob': round(m2_lp_mu, 4),
            'file_m1_mean_shannon': round(m1_sh_mu, 4),
            'file_m2_mean_shannon': round(m2_sh_mu, 4),
            'm1_correct': int(mask1[i1]) if 0 <= i1 < len(mask1) else 0,
            'm2_correct': int(mask2[i2]) if 0 <= i2 < len(mask2) else 0,
        }
        c1, c2 = row['m1_correct'], row['m2_correct']
        row['label'] = (
            'both' if c1 and c2
            else 'm1_only' if c1
            else 'm2_only' if c2
            else 'neither'
        )
        writer.writerow(row)
        n_written += 1

    fh.flush()
    return n_written, n_disagree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', required=True)
    parser.add_argument('--model2', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    log.info(f'Loading {args.model1} and {args.model2} on {args.dataset}')
    r1 = load_results(args.model1, args.dataset)
    r2 = load_results(args.model2, args.dataset)
    common = sorted(set(r1.keys()) & set(r2.keys()))
    log.info(f'{len(common)} common files')

    out_dir = MALLEUS_DIR / 'results' / args.dataset
    csv_path = out_dir / f'disagreements_{args.model1}_+_{args.model2}.csv'

    fh = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    writer.writeheader()
    fh.flush()

    totals = {'rows': 0, 'disagreements_total': 0}
    label_counts = {'m1_only': 0, 'm2_only': 0, 'both': 0, 'neither': 0}

    for idx, fid in enumerate(common, 1):
        d1 = r1[fid]
        d2 = r2[fid]
        n_rows, n_dis = extract_for_file(fid, d1, d2, args.model1, args.model2, writer, fh)
        totals['rows'] += n_rows
        totals['disagreements_total'] += n_dis
        log.info(f'[{idx}/{len(common)}] {fid}: {n_rows} disagreement rows')

    fh.close()

    # Second pass: count labels from the CSV we just wrote (memory-light).
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_counts[row['label']] += 1

    log.info('=' * 60)
    log.info(f'Wrote {totals["rows"]} disagreement rows to {csv_path}')
    log.info(f'Label distribution:')
    total = sum(label_counts.values())
    for name in ['m1_only', 'm2_only', 'both', 'neither']:
        c = label_counts[name]
        pct = c / total * 100 if total else 0.0
        log.info(f'  {name:<10s} {c:>5d}  ({pct:5.1f}%)')
    oracle_headroom = label_counts['m1_only'] + label_counts['m2_only'] + label_counts['both']
    log.info(f'Recoverable disagreements (oracle headroom): '
             f'{oracle_headroom} / {total}  ({oracle_headroom / total * 100:.1f}%)'
             if total else '')


if __name__ == '__main__':
    main()
