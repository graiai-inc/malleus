#!/usr/bin/env python3
"""ROVER fusion variants for confidence-weighted hypothesis combination.

Three variants:
1. naive — equal weighting, classic ROVER (Fiscus 1997)
2. confidence — per-token log-prob weighting
3. entropy — Shannon entropy gating (low entropy = high confidence)

Loads malleus benchmark results (JSONL with tokens, ys_log_probs, vocab_summaries)
for two models, fuses them with each variant, computes WER against reference.

Usage:
    python scripts/rover_variants.py \\
        --model1 parakeet-tdt-0.6b-v2 --model2 whisper-distil-v3.5 \\
        --dataset primock57
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MALLEUS_DIR = SCRIPT_DIR.parent
WNORM = EnglishTextNormalizer()


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return [w for w in text.split() if w]


def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def word_similarity(w1: str, w2: str) -> float:
    if w1 == w2:
        return 2.0
    dist = levenshtein(w1, w2)
    max_len = max(len(w1), len(w2))
    if max_len == 0:
        return -1.0
    sim = 1.0 - (dist / max_len)
    return sim if sim > 0.7 else -1.0


def align_sequences(seq1: list[str], seq2: list[str]) -> list[tuple]:
    """DP alignment returning list of (word1_or_None, word2_or_None, idx1, idx2)."""
    m, n = len(seq1), len(seq2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[''] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] - 1
        bt[i][0] = 'D'
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] - 1
        bt[0][j] = 'I'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = word_similarity(seq1[i-1], seq2[j-1])
            scores = [
                dp[i-1][j-1] + match,
                dp[i-1][j] - 1,
                dp[i][j-1] - 1,
            ]
            best = max(scores)
            dp[i][j] = best
            bt[i][j] = ['M', 'D', 'I'][scores.index(best)]

    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            alignment.append((None, seq2[j-1], -1, j-1))
            j -= 1
        elif j == 0:
            alignment.append((seq1[i-1], None, i-1, -1))
            i -= 1
        elif bt[i][j] == 'M':
            alignment.append((seq1[i-1], seq2[j-1], i-1, j-1))
            i -= 1
            j -= 1
        elif bt[i][j] == 'D':
            alignment.append((seq1[i-1], None, i-1, -1))
            i -= 1
        else:
            alignment.append((None, seq2[j-1], -1, j-1))
            j -= 1
    return list(reversed(alignment))


def fuse_naive(text1: str, text2: str, conf1=None, conf2=None) -> str:
    """Equal-weight ROVER. Ignores any provided confidence."""
    return _fuse(text1, text2, [0.5] * 9999, [0.5] * 9999)


def fuse_confidence_weighted(text1: str, text2: str,
                              conf1: list[float], conf2: list[float]) -> str:
    """ROVER with per-token confidence as weights.
    conf values are log-probs from the model; convert to probabilities."""
    p1 = [math.exp(c) for c in conf1]
    p2 = [math.exp(c) for c in conf2]
    return _fuse(text1, text2, p1, p2)


def fuse_entropy_gated(text1: str, text2: str,
                       entropy1: list[float], entropy2: list[float]) -> str:
    """ROVER weighted by inverse entropy. Lower entropy = higher weight."""
    # Convert entropy to weight: lower entropy = more confident = higher weight
    # Use 1/(1+entropy) to bound and avoid div-by-zero
    w1 = [1.0 / (1.0 + max(0, e)) for e in entropy1]
    w2 = [1.0 / (1.0 + max(0, e)) for e in entropy2]
    return _fuse(text1, text2, w1, w2)


def _fuse(text1: str, text2: str, w1: list[float], w2: list[float]) -> str:
    """Core ROVER fusion using per-position weights."""
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    if not words1: return text2
    if not words2: return text1

    alignment = align_sequences(words1, words2)
    out = []
    for word1, word2, idx1, idx2 in alignment:
        candidates = {}
        if word1 is not None:
            weight = w1[idx1] if 0 <= idx1 < len(w1) else 0.5
            candidates[word1] = candidates.get(word1, 0) + weight
        if word2 is not None:
            weight = w2[idx2] if 0 <= idx2 < len(w2) else 0.5
            candidates[word2] = candidates.get(word2, 0) + weight
        if candidates:
            best = max(candidates, key=candidates.get)
            out.append(best)
    return ' '.join(out)


def load_results(model_name: str, dataset: str) -> dict:
    """Load JSONL results indexed by file_id."""
    path = MALLEUS_DIR / 'results' / dataset / f'{model_name}.jsonl'
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


def expand_token_confidence_to_words(tokens: list[str], token_confs: list[float],
                                      hypothesis: str) -> list[float]:
    """Map token-level confidence to word-level confidence.

    Tokens may be subword units. We tokenize the hypothesis into words and
    assign each word the mean log-prob of its constituent subword tokens.
    Falls back to neutral 0.5 if mapping fails.
    """
    words = tokenize(hypothesis)
    if not words:
        return []
    if not tokens or not token_confs:
        return [0.5] * len(words)

    # Simplest mapping: distribute tokens evenly across words
    # (good enough for ASR where token-to-word ratio is roughly constant)
    if len(token_confs) >= len(words):
        ratio = len(token_confs) / len(words)
        word_confs = []
        for wi in range(len(words)):
            start = int(wi * ratio)
            end = int((wi + 1) * ratio)
            chunk = token_confs[start:end] if end > start else [token_confs[start]]
            word_confs.append(sum(chunk) / len(chunk))
        return word_confs
    else:
        # Fewer tokens than words (unusual) — repeat
        return [token_confs[min(i, len(token_confs)-1)] for i in range(len(words))]


def compute_wer(ref: str, hyp: str) -> float:
    return jiwer.wer(WNORM(ref), WNORM(hyp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', required=True)
    parser.add_argument('--model2', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    log.info(f'Loading results for {args.model1} and {args.model2} on {args.dataset}')
    r1 = load_results(args.model1, args.dataset)
    r2 = load_results(args.model2, args.dataset)
    common = set(r1.keys()) & set(r2.keys())
    log.info(f'{len(common)} common files')

    results = {
        'naive': {'errors': 0, 'words': 0, 'per_file': []},
        'confidence': {'errors': 0, 'words': 0, 'per_file': []},
        'entropy': {'errors': 0, 'words': 0, 'per_file': []},
        'm1_only': {'errors': 0, 'words': 0, 'per_file': []},
        'm2_only': {'errors': 0, 'words': 0, 'per_file': []},
    }

    for fid in sorted(common):
        d1 = r1[fid]
        d2 = r2[fid]
        ref = d1['reference']
        hyp1 = d1['hypothesis']
        hyp2 = d2['hypothesis']

        # Map token confidences to word confidences
        c1 = expand_token_confidence_to_words(d1['tokens'], d1['ys_log_probs'], hyp1)
        c2 = expand_token_confidence_to_words(d2['tokens'], d2['ys_log_probs'], hyp2)
        e1 = expand_token_confidence_to_words(
            d1['tokens'],
            [v['shannon'] for v in d1['vocab_summaries']],
            hyp1)
        e2 = expand_token_confidence_to_words(
            d2['tokens'],
            [v['shannon'] for v in d2['vocab_summaries']],
            hyp2)

        fused_naive = fuse_naive(hyp1, hyp2)
        fused_conf = fuse_confidence_weighted(hyp1, hyp2, c1, c2)
        fused_ent = fuse_entropy_gated(hyp1, hyp2, e1, e2)

        ref_w = WNORM(ref)
        n_words = len(ref_w.split())
        if n_words == 0:
            continue

        for name, h in [('naive', fused_naive),
                        ('confidence', fused_conf),
                        ('entropy', fused_ent),
                        ('m1_only', hyp1),
                        ('m2_only', hyp2)]:
            wer = compute_wer(ref, h)
            errors = round(wer * n_words)
            results[name]['errors'] += errors
            results[name]['words'] += n_words
            results[name]['per_file'].append({'file_id': fid, 'wer': round(wer, 4)})

    print(f'\n{"="*60}')
    print(f'ROVER variants: {args.model1} + {args.model2} on {args.dataset}')
    print(f'{"="*60}')
    print(f'{"Method":<20s} {"WER":>10s} {"Files better than naive":>25s}')
    print('-' * 60)

    naive_per_file = {pf['file_id']: pf['wer'] for pf in results['naive']['per_file']}

    for name in ['m1_only', 'm2_only', 'naive', 'confidence', 'entropy']:
        agg_wer = results[name]['errors'] / results[name]['words'] * 100
        if name in ('confidence', 'entropy'):
            better = sum(1 for pf in results[name]['per_file']
                         if pf['wer'] < naive_per_file[pf['file_id']])
            note = f'{better}/{len(results[name]["per_file"])}'
        else:
            note = ''
        print(f'  {name:<18s} {agg_wer:>8.2f}%  {note:>25s}')

    out_path = MALLEUS_DIR / 'results' / args.dataset / f'rover_{args.model1}_+_{args.model2}.json'
    out_path.write_text(json.dumps({
        'model1': args.model1,
        'model2': args.model2,
        'dataset': args.dataset,
        'aggregate_wer': {
            name: round(results[name]['errors'] / results[name]['words'] * 100, 2)
            for name in results
        },
        'per_file': {name: results[name]['per_file'] for name in results},
    }, indent=2))
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
