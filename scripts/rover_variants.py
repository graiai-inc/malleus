#!/usr/bin/env python3
"""ROVER fusion variants with subword-aware word-level confidence.

Variants:
    m1_only, m2_only           — solo baselines
    oracle                      — upper bound (any voting scheme)
    naive                       — equal-weight ROVER (Fiscus 1997)
    confidence                  — mean token log-prob per word
    shannon                     — inverse max Shannon entropy per word
    tsallis                     — inverse max Tsallis entropy per word
    margin                      — min (top1 − top2) over subwords
    asymmetric                  — default to m1, override only when m1
                                  uncertain AND m2 confident

Plus per-file signal analysis for a "when to fuse" gate.

Per-file results are written incrementally to JSONL as they complete.
"""

import argparse
import csv
import functools
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

# Asymmetric thresholds on top-1 probability (aggregated per word).
# Override m1 with m2 only if m1_top1 < M1_LOW and m2_top1 > M2_HIGH.
M1_LOW = 0.60
M2_HIGH = 0.90


# ----------------------------- subword → word -----------------------------


def words_with_confidence(tokens: list[str],
                          ys_log_probs: list[float],
                          vocab_summaries: list[dict]) -> list[dict]:
    """Group subword tokens into normalized words with aggregated signals.

    Word boundaries are detected from a leading space on the token string
    (both parakeet and whisper follow this convention). Trailing punctuation
    tokens attach to the current word.

    For each word returns:
        word          — lowercased, punctuation-stripped (matches tokenize())
        n_subwords    — number of tokens composing the word
        logprob_mean  — mean subword log-prob
        logprob_min   — weakest subword log-prob
        top1_mean     — mean top-1 probability across subwords
        top1_min      — weakest subword top-1 probability
        shannon_max   — max Shannon entropy across subwords
        shannon_mean  — mean Shannon entropy
        tsallis_max   — max Tsallis entropy
        tsallis_mean  — mean Tsallis entropy
        margin_min    — min (top1 − top2) across subwords
    """
    if not tokens:
        return []

    # Group token indices by word boundary.
    groups: list[tuple[str, list[int]]] = []
    cur_word = ''
    cur_idxs: list[int] = []
    for i, tok in enumerate(tokens):
        # A token starting with ' ' begins a new word (after flushing current).
        if tok.startswith(' ') and cur_word:
            groups.append((cur_word, cur_idxs))
            cur_word = ''
            cur_idxs = []
        cur_word += tok
        cur_idxs.append(i)
    if cur_word:
        groups.append((cur_word, cur_idxs))

    out = []
    for raw, idxs in groups:
        # Normalize identically to tokenize(): lowercase, strip non-word chars.
        norm = re.sub(r'[^\w\s]', '', raw.lower()).strip()
        if not norm:
            continue

        lps = [ys_log_probs[i] for i in idxs]
        shs = [vocab_summaries[i]['shannon'] for i in idxs]
        tss = [vocab_summaries[i]['tsallis'] for i in idxs]
        top5s = [vocab_summaries[i]['top5_probs'] for i in idxs]
        top1s = [t[0] if t else 0.0 for t in top5s]
        margins = [
            (t[0] - t[1]) if len(t) >= 2 else (t[0] if t else 0.0)
            for t in top5s
        ]

        out.append({
            'word': norm,
            'n_subwords': len(idxs),
            'logprob_mean': sum(lps) / len(lps),
            'logprob_min': min(lps),
            'top1_mean': sum(top1s) / len(top1s),
            'top1_min': min(top1s),
            'shannon_max': max(shs),
            'shannon_mean': sum(shs) / len(shs),
            'tsallis_max': max(tss),
            'tsallis_mean': sum(tss) / len(tss),
            'margin_min': min(margins),
        })
    return out


# ------------------------------ alignment --------------------------------


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
    """DP alignment returning (word1_or_None, word2_or_None, idx1, idx2)."""
    m, n = len(seq1), len(seq2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[''] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] - 1
        bt[i][0] = 'D'
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] - 1
        bt[0][j] = 'I'
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = word_similarity(seq1[i - 1], seq2[j - 1])
            scores = [
                dp[i - 1][j - 1] + match,
                dp[i - 1][j] - 1,
                dp[i][j - 1] - 1,
            ]
            best = max(scores)
            dp[i][j] = best
            bt[i][j] = ['M', 'D', 'I'][scores.index(best)]

    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            alignment.append((None, seq2[j - 1], -1, j - 1))
            j -= 1
        elif j == 0:
            alignment.append((seq1[i - 1], None, i - 1, -1))
            i -= 1
        elif bt[i][j] == 'M':
            alignment.append((seq1[i - 1], seq2[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif bt[i][j] == 'D':
            alignment.append((seq1[i - 1], None, i - 1, -1))
            i -= 1
        else:
            alignment.append((None, seq2[j - 1], -1, j - 1))
            j -= 1
    return list(reversed(alignment))


# ------------------------------- fusion ----------------------------------


def fuse_weighted_aligned(alignment: list[tuple],
                          w1: list[float], w2: list[float]) -> str:
    """Apply weighted voting over a precomputed alignment."""
    out = []
    for word1, word2, idx1, idx2 in alignment:
        candidates: dict[str, float] = {}
        if word1 is not None:
            wt = w1[idx1] if 0 <= idx1 < len(w1) else 0.5
            candidates[word1] = candidates.get(word1, 0.0) + wt
        if word2 is not None:
            wt = w2[idx2] if 0 <= idx2 < len(w2) else 0.5
            candidates[word2] = candidates.get(word2, 0.0) + wt
        if candidates:
            out.append(max(candidates, key=candidates.get))
    return ' '.join(out)


def fuse_asymmetric_aligned(alignment: list[tuple],
                            data1: list[dict], data2: list[dict],
                            m1_low: float = M1_LOW,
                            m2_high: float = M2_HIGH) -> str:
    """Default to m1 over a precomputed alignment. Override with m2 only when
    m1_top1 < m1_low AND m2_top1 > m2_high. Never insert/delete from m2.
    """
    out = []
    for word1, word2, idx1, idx2 in alignment:
        if word1 is None:
            continue
        if word2 is None or word1 == word2:
            out.append(word1)
            continue
        c1 = data1[idx1]['top1_mean'] if 0 <= idx1 < len(data1) else 1.0
        c2 = data2[idx2]['top1_mean'] if 0 <= idx2 < len(data2) else 0.0
        if c1 < m1_low and c2 > m2_high:
            out.append(word2)
        else:
            out.append(word1)
    return ' '.join(out)


# -------------------------- metrics + oracle -----------------------------


def compute_wer(ref: str, hyp: str) -> float:
    return jiwer.wer(WNORM(ref), WNORM(hyp))


def oracle_errors(ref: str, hyp1: str, hyp2: str) -> tuple[int, int]:
    """Upper bound on any voting scheme: a ref word is recoverable iff at
    least one hypothesis aligns it as a match. Returns (unrecoverable, n_ref).
    """
    ref_n = WNORM(ref)
    n = len(ref_n.split())
    if n == 0:
        return 0, 0
    recovered = [False] * n
    for hyp in (hyp1, hyp2):
        out = jiwer.process_words(ref_n, WNORM(hyp))
        for chunk in out.alignments[0]:
            if chunk.type == 'equal':
                for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    recovered[chunk.ref_start_idx + k] = True
    return sum(1 for r in recovered if not r), n


# ----------------------------- per-file signals --------------------------


def compute_signals(words1: list[str], words2: list[str],
                    data1: list[dict], data2: list[dict],
                    alignment: list[tuple]) -> dict:
    """Reference-free signals for the 'when to fuse' gate."""
    if not words1 or not words2:
        return {
            'disagreement': 1.0,
            'logprob_mean_m1': 0.0,
            'logprob_mean_m2': 0.0,
            'shannon_mean_m1': 0.0,
            'shannon_mean_m2': 0.0,
            'conf_gap': 0.0,
            'n_words_m1': len(words1),
            'n_words_m2': len(words2),
        }

    n_slots = len(alignment)
    n_differ_or_gap = sum(
        1 for w1, w2, _, _ in alignment if w1 != w2 or w1 is None or w2 is None
    )
    disagreement = n_differ_or_gap / n_slots if n_slots else 0.0

    lp1 = sum(d['logprob_mean'] for d in data1) / len(data1)
    lp2 = sum(d['logprob_mean'] for d in data2) / len(data2)
    sh1 = sum(d['shannon_mean'] for d in data1) / len(data1)
    sh2 = sum(d['shannon_mean'] for d in data2) / len(data2)

    return {
        'disagreement': round(disagreement, 4),
        'logprob_mean_m1': round(lp1, 4),
        'logprob_mean_m2': round(lp2, 4),
        'shannon_mean_m1': round(sh1, 4),
        'shannon_mean_m2': round(sh2, 4),
        'conf_gap': round(abs(lp1 - lp2), 4),
        'n_words_m1': len(words1),
        'n_words_m2': len(words2),
    }


# --------------------------------- main ----------------------------------


def load_results(model_name: str, dataset: str) -> dict:
    path = MALLEUS_DIR / 'results' / dataset / f'{model_name}.jsonl'
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


VARIANTS = [
    'm1_only', 'm2_only',
    'naive', 'confidence', 'shannon', 'tsallis', 'margin', 'asymmetric',
    'oracle',
]


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
    per_file_path = out_dir / f'rover_v2_{args.model1}_+_{args.model2}.jsonl'
    summary_path = out_dir / f'rover_v2_{args.model1}_+_{args.model2}.json'

    # Incremental: open JSONL for append-as-we-go.
    per_file_path.unlink(missing_ok=True)
    per_file_fh = open(per_file_path, 'w')

    totals: dict[str, dict[str, int]] = {
        v: {'errors': 0, 'words': 0} for v in VARIANTS
    }
    per_file_records = []

    for idx, fid in enumerate(common, 1):
        d1 = r1[fid]
        d2 = r2[fid]
        ref = d1['reference']
        hyp1 = d1['hypothesis']
        hyp2 = d2['hypothesis']

        wd1 = words_with_confidence(
            d1['tokens'], d1['ys_log_probs'], d1['vocab_summaries'],
        )
        wd2 = words_with_confidence(
            d2['tokens'], d2['ys_log_probs'], d2['vocab_summaries'],
        )
        words1 = [d['word'] for d in wd1]
        words2 = [d['word'] for d in wd2]

        # Weight lists per variant.
        w_naive_1 = [1.0] * len(wd1)
        w_naive_2 = [1.0] * len(wd2)
        w_conf_1 = [math.exp(d['logprob_mean']) for d in wd1]
        w_conf_2 = [math.exp(d['logprob_mean']) for d in wd2]
        w_sh_1 = [1.0 / (1.0 + max(0.0, d['shannon_max'])) for d in wd1]
        w_sh_2 = [1.0 / (1.0 + max(0.0, d['shannon_max'])) for d in wd2]
        w_ts_1 = [1.0 / (1.0 + max(0.0, d['tsallis_max'])) for d in wd1]
        w_ts_2 = [1.0 / (1.0 + max(0.0, d['tsallis_max'])) for d in wd2]
        w_mg_1 = [max(0.0, d['margin_min']) for d in wd1]
        w_mg_2 = [max(0.0, d['margin_min']) for d in wd2]

        # Compute alignment ONCE and reuse across all variants + signals.
        if words1 and words2:
            alignment = align_sequences(words1, words2)
        else:
            alignment = []

        if not words1:
            hypotheses = {
                'm1_only': hyp1,
                'm2_only': hyp2,
                'naive': ' '.join(words2),
                'confidence': ' '.join(words2),
                'shannon': ' '.join(words2),
                'tsallis': ' '.join(words2),
                'margin': ' '.join(words2),
                'asymmetric': ' '.join(words2),
            }
        elif not words2:
            hypotheses = {
                'm1_only': hyp1,
                'm2_only': hyp2,
                'naive': ' '.join(words1),
                'confidence': ' '.join(words1),
                'shannon': ' '.join(words1),
                'tsallis': ' '.join(words1),
                'margin': ' '.join(words1),
                'asymmetric': ' '.join(words1),
            }
        else:
            hypotheses = {
                'm1_only': hyp1,
                'm2_only': hyp2,
                'naive': fuse_weighted_aligned(alignment, w_naive_1, w_naive_2),
                'confidence': fuse_weighted_aligned(alignment, w_conf_1, w_conf_2),
                'shannon': fuse_weighted_aligned(alignment, w_sh_1, w_sh_2),
                'tsallis': fuse_weighted_aligned(alignment, w_ts_1, w_ts_2),
                'margin': fuse_weighted_aligned(alignment, w_mg_1, w_mg_2),
                'asymmetric': fuse_asymmetric_aligned(alignment, wd1, wd2),
            }

        ref_n = WNORM(ref)
        n_words = len(ref_n.split())
        if n_words == 0:
            continue

        file_record = {
            'file_id': fid,
            'n_ref_words': n_words,
            'wer': {},
            'signals': compute_signals(words1, words2, wd1, wd2, alignment),
        }

        for name, h in hypotheses.items():
            wer = compute_wer(ref, h)
            errors = round(wer * n_words)
            totals[name]['errors'] += errors
            totals[name]['words'] += n_words
            file_record['wer'][name] = round(wer, 4)

        # Oracle
        unrec, n_ref = oracle_errors(ref, hyp1, hyp2)
        totals['oracle']['errors'] += unrec
        totals['oracle']['words'] += n_ref
        file_record['wer']['oracle'] = round(unrec / n_ref, 4) if n_ref else 0.0

        per_file_fh.write(json.dumps(file_record) + '\n')
        per_file_fh.flush()
        per_file_records.append(file_record)

        log.info(
            f'[{idx}/{len(common)}] {fid}: '
            f'm1={file_record["wer"]["m1_only"]:.3f} '
            f'm2={file_record["wer"]["m2_only"]:.3f} '
            f'naive={file_record["wer"]["naive"]:.3f} '
            f'asym={file_record["wer"]["asymmetric"]:.3f} '
            f'oracle={file_record["wer"]["oracle"]:.3f}'
        )

    per_file_fh.close()

    # --------------------- aggregate + gate analysis ---------------------

    agg_wer = {
        name: round(totals[name]['errors'] / totals[name]['words'] * 100, 2)
        if totals[name]['words'] else 0.0
        for name in VARIANTS
    }

    # For each file, which variant was best, and did any fusion beat best solo?
    fusion_variants = ['naive', 'confidence', 'shannon', 'tsallis', 'margin', 'asymmetric']
    gate_rows = []
    for rec in per_file_records:
        best_solo = min(rec['wer']['m1_only'], rec['wer']['m2_only'])
        best_fusion = min(rec['wer'][v] for v in fusion_variants)
        best_fusion_name = min(fusion_variants, key=lambda v: rec['wer'][v])
        helps = best_fusion < best_solo - 1e-9
        gate_rows.append({
            'file_id': rec['file_id'],
            'best_solo_wer': best_solo,
            'best_fusion_wer': best_fusion,
            'best_fusion_name': best_fusion_name,
            'fusion_helps': helps,
            **rec['signals'],
        })

    # Save gate CSV for offline inspection.
    gate_csv = out_dir / f'rover_v2_gate_{args.model1}_+_{args.model2}.csv'
    with open(gate_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gate_rows)

    # Simple threshold search on each signal.
    def threshold_report(signal: str) -> dict:
        rows = sorted(gate_rows, key=lambda r: r[signal])
        best = None
        for i in range(1, len(rows)):
            thr = (rows[i - 1][signal] + rows[i][signal]) / 2
            below = [r for r in rows if r[signal] < thr]
            above = [r for r in rows if r[signal] >= thr]
            if not below or not above:
                continue
            # Policy: fuse if signal >= thr, else use best solo.
            err_if_gated = 0
            words_if_gated = 0
            for r in rows:
                rec = next(x for x in per_file_records if x['file_id'] == r['file_id'])
                n = rec['n_ref_words']
                if r[signal] >= thr:
                    wer = r['best_fusion_wer']
                else:
                    wer = r['best_solo_wer']
                err_if_gated += round(wer * n)
                words_if_gated += n
            gated_wer = err_if_gated / words_if_gated if words_if_gated else 0
            if best is None or gated_wer < best['gated_wer']:
                best = {'threshold': round(thr, 4), 'gated_wer': round(gated_wer * 100, 2)}
        return best or {'threshold': None, 'gated_wer': agg_wer['m1_only']}

    gate_analysis = {
        sig: threshold_report(sig)
        for sig in ['disagreement', 'conf_gap', 'logprob_mean_m1', 'logprob_mean_m2',
                    'shannon_mean_m1', 'shannon_mean_m2']
    }

    summary = {
        'model1': args.model1,
        'model2': args.model2,
        'dataset': args.dataset,
        'n_files': len(per_file_records),
        'aggregate_wer': agg_wer,
        'n_files_fusion_helps': sum(1 for r in gate_rows if r['fusion_helps']),
        'best_fusion_name_counts': {
            name: sum(1 for r in gate_rows if r['best_fusion_name'] == name
                      and r['fusion_helps'])
            for name in fusion_variants
        },
        'gate_analysis': gate_analysis,
        'notes': (
            'Per-file results in rover_v2_<m1>_+_<m2>.jsonl. '
            'Gate CSV in rover_v2_gate_<m1>_+_<m2>.csv. '
            'gate_analysis shows the best single-signal threshold: if signal '
            '>= threshold, use best fusion, else use best solo. gated_wer is '
            'the resulting corpus WER under that oracle-of-methods policy.'
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    # --------------------------- print summary --------------------------

    print(f'\n{"=" * 64}')
    print(f'ROVER v2: {args.model1} + {args.model2} on {args.dataset}')
    print(f'{"=" * 64}')
    print(f'{"Method":<14s} {"WER":>10s}')
    print('-' * 26)
    for name in VARIANTS:
        print(f'  {name:<12s} {agg_wer[name]:>7.2f}%')

    print(f'\nFiles where any fusion beat best solo: '
          f'{summary["n_files_fusion_helps"]}/{len(gate_rows)}')
    print('Best-fusion method counts (among files where fusion helped):')
    for name, count in summary['best_fusion_name_counts'].items():
        print(f'  {name:<12s} {count:>3d}')

    print(f'\nGate analysis (use fusion above threshold, else solo):')
    print(f'  {"signal":<20s} {"threshold":>12s} {"gated WER":>12s}')
    for sig, res in gate_analysis.items():
        t = res['threshold']
        w = res['gated_wer']
        t_str = f'{t:.4f}' if t is not None else 'n/a'
        print(f'  {sig:<20s} {t_str:>12s} {w:>10.2f}%')

    print(f'\nSaved:')
    print(f'  {per_file_path}')
    print(f'  {gate_csv}')
    print(f'  {summary_path}')


if __name__ == '__main__':
    main()
