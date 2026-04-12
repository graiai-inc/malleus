#!/usr/bin/env python3
"""Train a 'which model to trust' classifier on disagreement-slot features.

Loads the CSV produced by extract_disagreement_slots.py, drops 'both' and
'neither' rows (ambiguous / unreachable), fits logistic regression + random
forest with cross-validation, and reports:
    * per-feature correlation with the binary label
    * CV accuracy for each model
    * confusion matrix on out-of-fold predictions
    * translated WER delta on primock57 (how many words we'd flip)

Writes feature-importance tables and per-row predictions to results/.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MALLEUS_DIR = SCRIPT_DIR.parent

# Numeric features used for training. Non-numeric columns (file_id, slot_idx,
# word_m1, word_m2, label, m1_correct, m2_correct) are excluded.
FEATURE_COLS = [
    'm1_top1_mean', 'm2_top1_mean',
    'm1_top1_min', 'm2_top1_min',
    'm1_margin_min', 'm2_margin_min',
    'm1_logprob_mean', 'm2_logprob_mean',
    'm1_logprob_min', 'm2_logprob_min',
    'm1_shannon_max', 'm2_shannon_max',
    'm1_shannon_mean', 'm2_shannon_mean',
    'm1_tsallis_max', 'm2_tsallis_max',
    'm1_shannon_norm_max', 'm2_shannon_norm_max',
    'm1_n_subwords', 'm2_n_subwords',
    'm1_top1_z', 'm2_top1_z',
    'm1_shannon_z', 'm2_shannon_z',
    'm1_logprob_z', 'm2_logprob_z',
    'char_edit_dist', 'char_edit_ratio',
    'phonetic_edit_dist', 'phonetic_match',
    'char_len_m1', 'char_len_m2',
    'burst_size',
    'left_agree_1', 'right_agree_1',
    'left_agree_2', 'right_agree_2',
    'left_agree_3', 'right_agree_3',
    'file_disagreement_rate',
    'file_m1_mean_logprob', 'file_m2_mean_logprob',
    'file_m1_mean_shannon', 'file_m2_mean_shannon',
]


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Load CSV, drop 'both' and 'neither' rows. Return X, y, groups, raw rows.

    y = 1 means 'trust m2' (m2_only); y = 0 means 'trust m1' (m1_only).
    groups = file_id for GroupKFold.
    """
    X_rows = []
    y_rows = []
    groups = []
    raw = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            if label not in ('m1_only', 'm2_only'):
                continue
            X_rows.append([float(row[c]) for c in FEATURE_COLS])
            y_rows.append(1 if label == 'm2_only' else 0)
            groups.append(row['file_id'])
            raw.append(row)
    return np.array(X_rows), np.array(y_rows), groups, raw


def single_feature_report(X: np.ndarray, y: np.ndarray) -> list[tuple]:
    """For each feature, compute correlation with label and single-feature
    logistic-regression accuracy. Useful sanity check."""
    results = []
    for i, name in enumerate(FEATURE_COLS):
        col = X[:, i]
        # Pearson correlation with label
        if col.std() < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y)[0, 1])
        # Single-feature LR accuracy (no CV, quick)
        try:
            clf = LogisticRegression(max_iter=200, class_weight='balanced')
            clf.fit(col.reshape(-1, 1), y)
            acc = float(clf.score(col.reshape(-1, 1), y))
        except Exception:
            acc = 0.0
        results.append((name, corr, acc))
    results.sort(key=lambda r: abs(r[1]), reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', required=True)
    parser.add_argument('--model2', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    out_dir = MALLEUS_DIR / 'results' / args.dataset
    csv_path = out_dir / f'disagreements_{args.model1}_+_{args.model2}.csv'
    log.info(f'Loading {csv_path}')

    X, y, groups, raw = load_csv(csv_path)
    log.info(f'Loaded {len(X)} rows, {int(y.sum())} m2_only, {int((1 - y).sum())} m1_only')
    log.info(f'"Always trust m1" baseline accuracy: {(1 - y).mean():.4f}')

    # --------------------------- single-feature --------------------------

    log.info('Computing single-feature correlations + accuracy...')
    single = single_feature_report(X, y)
    log.info('Top-15 features by |correlation| with label (label=1 => m2_only):')
    log.info(f'  {"feature":<28s} {"corr":>8s} {"1feat_acc":>12s}')
    for name, corr, acc in single[:15]:
        log.info(f'  {name:<28s} {corr:>+8.4f} {acc:>10.4f}')

    (out_dir / f'selector_feature_corr_{args.model1}_+_{args.model2}.csv').write_text(
        'feature,correlation,single_feature_accuracy\n'
        + '\n'.join(f'{n},{c:.6f},{a:.6f}' for n, c, a in single)
    )

    # --------------------------- CV classifiers ---------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GroupKFold by file_id — no file leaks between train and val.
    n_splits = min(5, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)

    classifiers = {
        'logreg_balanced': LogisticRegression(
            max_iter=1000, class_weight='balanced', C=1.0,
        ),
        'logreg_unbalanced': LogisticRegression(
            max_iter=1000, C=1.0,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight='balanced',
            random_state=0, n_jobs=-1,
        ),
    }

    reports = {}
    for name, clf in classifiers.items():
        log.info(f'Cross-validating {name} with GroupKFold({n_splits})...')
        preds = cross_val_predict(
            clf, X_scaled, y, groups=groups, cv=cv, method='predict'
        )
        probs = cross_val_predict(
            clf, X_scaled, y, groups=groups, cv=cv, method='predict_proba'
        )[:, 1]

        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = float('nan')
        cm = confusion_matrix(y, preds).tolist()

        # WER delta: number of slots where we'd flip vs always-m1 and
        # what fraction of flips are correct.
        flipped_to_m2 = (preds == 1).sum()
        correct_flips = ((preds == 1) & (y == 1)).sum()
        wrong_flips = ((preds == 1) & (y == 0)).sum()
        net_fixed = correct_flips - wrong_flips

        log.info(
            f'  acc={acc:.4f}  auc={auc:.4f}  cm={cm}  '
            f'flipped={flipped_to_m2}  correct={correct_flips}  '
            f'wrong={wrong_flips}  net_fixed={net_fixed}'
        )
        reports[name] = {
            'accuracy': round(acc, 4),
            'auc': round(auc, 4) if auc == auc else None,
            'confusion_matrix_rows_true_cols_pred': cm,
            'flipped_to_m2': int(flipped_to_m2),
            'correct_flips': int(correct_flips),
            'wrong_flips': int(wrong_flips),
            'net_fixed_disagreements': int(net_fixed),
        }

    # -------------------- translate to WER delta ------------------------

    # Sum total reference words from the rover_v2 per-file file.
    per_file_jsonl = out_dir / f'rover_v2_{args.model1}_+_{args.model2}.jsonl'
    total_ref_words = 0
    total_m1_errors = 0
    if per_file_jsonl.exists():
        with open(per_file_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                n = rec['n_ref_words']
                total_ref_words += n
                total_m1_errors += round(rec['wer']['m1_only'] * n)
    log.info(
        f'Total ref words: {total_ref_words}, m1 errors: {total_m1_errors}, '
        f'm1 WER: {total_m1_errors / total_ref_words * 100:.2f}%'
        if total_ref_words else 'rover_v2 jsonl not found; skipping WER delta'
    )

    # For each classifier, project WER improvement:
    # new_errors ≈ m1_errors − net_fixed_disagreements
    # (assumes correct flips convert wrong→right, wrong flips convert
    #  right→wrong, and non-disagreement errors are untouched. This is a
    #  lower-bound estimate because reference alignment may not 1:1 map
    #  every disagreement slot to a single error, but close enough.)
    wer_projection = {}
    for name, r in reports.items():
        projected_errors = max(0, total_m1_errors - r['net_fixed_disagreements'])
        projected_wer = projected_errors / total_ref_words if total_ref_words else 0.0
        wer_projection[name] = {
            'projected_wer_pct': round(projected_wer * 100, 2),
            'delta_vs_m1_pp': round(
                (projected_wer - total_m1_errors / total_ref_words) * 100, 2
            ) if total_ref_words else 0.0,
        }
        log.info(
            f'  {name}: projected WER ≈ {projected_wer * 100:.2f}%  '
            f'(m1 baseline 13.76%)'
        )

    summary = {
        'model1': args.model1,
        'model2': args.model2,
        'dataset': args.dataset,
        'n_rows_total': len(X),
        'n_m1_only': int((1 - y).sum()),
        'n_m2_only': int(y.sum()),
        'baseline_always_m1_accuracy': round(float((1 - y).mean()), 4),
        'classifier_reports': reports,
        'wer_projection': wer_projection,
        'top_features_by_corr': [
            {'feature': n, 'correlation': round(c, 4), 'single_feat_acc': round(a, 4)}
            for n, c, a in single[:20]
        ],
    }
    summary_path = out_dir / f'selector_summary_{args.model1}_+_{args.model2}.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f'Saved summary to {summary_path}')


if __name__ == '__main__':
    main()
