#!/usr/bin/env python3
"""Diagnose why learned CNC doesn't beat hand-tuned CNC.

Dump learned-prob distribution and see how well it separates labels."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from learned_fusion import (extract_file_events, load, W)

p = load('parakeet-tdt-0.6b-v2')
w = load('whisper-distil-v3.5')
common = sorted(p.keys() & w.keys())

file_events = {}
for fid in common:
    events, hp, hw = extract_file_events(fid, p[fid], w[fid])
    file_events[fid] = events

X = np.stack([e[0] for evs in file_events.values() for e in evs])
y = np.array([e[1] for evs in file_events.values() for e in evs])
raw_conf = X[:, 7]  # own_conf is index 7 (5 wtype + word_len + n_subwords + own_conf)

print(f'total events: {len(y)}, positive: {y.sum()} ({y.mean()*100:.1f}%)')
print()
print('RAW top1_mean (own_conf) as a classifier:')
for thr in [0.5, 0.7, 0.85, 0.9, 0.95, 0.97]:
    pred = (raw_conf > thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    acc = (tp + tn) / len(y)
    print(f'  thr={thr:.2f}  acc={acc*100:.1f}%  '
          f'TP={tp} FP={fp} FN={fn} TN={tn}')
print()

# GBM 5-fold CV, dump probability distributions
fids = np.array(common)
gkf = GroupKFold(n_splits=5)
all_probs = np.zeros(len(y))
idx = 0
file_to_event_range = {}
for fid in common:
    n = len(file_events[fid])
    file_to_event_range[fid] = (idx, idx + n)
    idx += n

for train_gi, test_gi in gkf.split(fids, groups=np.arange(len(fids))):
    train_fids = fids[train_gi]
    test_fids = fids[test_gi]
    X_train = []
    y_train = []
    for fid in train_fids:
        for e in file_events[fid]:
            X_train.append(e[0])
            y_train.append(e[1])
    X_train = np.stack(X_train)
    y_train = np.array(y_train)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=0,
    ).fit(X_train, y_train)
    for fid in test_fids:
        lo, hi = file_to_event_range[fid]
        X_te = np.stack([e[0] for e in file_events[fid]])
        all_probs[lo:hi] = clf.predict_proba(X_te)[:, 1]

# Distribution of predicted probs
print('GBM predicted prob distribution:')
bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    mask = (all_probs >= lo) & (all_probs < hi + (1e-9 if i == len(bins)-2 else 0))
    n = mask.sum()
    pos = y[mask].sum() if n else 0
    print(f'  [{lo:.2f}, {hi:.2f})  n={n:>6}  positive_rate={pos/max(n,1)*100:5.1f}%')
print()

# Predictive power
print('GBM prob as classifier:')
for thr in [0.3, 0.5, 0.7, 0.85, 0.9]:
    pred = (all_probs > thr).astype(int)
    acc = (pred == y).mean()
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    print(f'  thr={thr:.2f}  acc={acc*100:.1f}%  TP={tp} FP={fp} FN={fn} TN={tn}')

# Compare how well GBM vs raw_conf separates the 'keep' decision on CASE B
# equivalents: words with label=0 (should drop) — where do they land?
drop_mask = (y == 0)
print()
print(f'Words with label=0 (should drop): {drop_mask.sum()}')
print(f'  raw_conf mean: {raw_conf[drop_mask].mean():.3f}  '
      f'std: {raw_conf[drop_mask].std():.3f}')
print(f'  GBM prob mean: {all_probs[drop_mask].mean():.3f}  '
      f'std: {all_probs[drop_mask].std():.3f}')

keep_mask = (y == 1)
print(f'Words with label=1 (should keep): {keep_mask.sum()}')
print(f'  raw_conf mean: {raw_conf[keep_mask].mean():.3f}  '
      f'std: {raw_conf[keep_mask].std():.3f}')
print(f'  GBM prob mean: {all_probs[keep_mask].mean():.3f}  '
      f'std: {all_probs[keep_mask].std():.3f}')

# separation: (keep mean - drop mean) / pooled std
raw_sep = (raw_conf[keep_mask].mean() - raw_conf[drop_mask].mean()) / raw_conf.std()
gbm_sep = (all_probs[keep_mask].mean() - all_probs[drop_mask].mean()) / all_probs.std()
print(f'\nSeparation (std units):')
print(f'  raw_conf: {raw_sep:.3f}')
print(f'  GBM prob: {gbm_sep:.3f}')
