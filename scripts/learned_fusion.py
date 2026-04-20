#!/usr/bin/env python3
"""Unified per-word keep classifier + CNC fusion with learned weights.

Idea: instead of voting each word by its raw top1_mean, train a classifier
to predict P(this word is correct at its aligned ref position) and use that
probability as the vote weight in CNC. Covers all slot types at once —
agreements, substitutions, empty-space, and pure insertions — with a single
model.

Evaluation: 5-fold GroupKFold by file_id. Each fold trains the classifier
on training files and runs CNC with learned weights on held-out files.
Aggregated WER is reported; no file appears in both train and test.

Features per word (own model, aligned positions in other model):
    word_type        filler/function/short/mid_content/long_content
    word_len         char count
    n_subwords       subword token count
    own_conf         top1_mean
    own_shannon      shannon_mean
    own_margin       margin_min
    own_prev_conf    previous word's top1_mean in same model (0 if none)
    own_next_conf    next word's top1_mean in same model (0 if none)
    position_frac    word idx / total words
    model            0 = parakeet, 1 = whisper
    other_has_here   1 if other model has aligned word at same ref pos
    other_here_conf  other model's word conf at same ref pos (0 if not)
    other_left_conf  other model's conf at nearest aligned left neighbor
    other_right_conf other model's conf at nearest aligned right neighbor

Label: 1 if the word equals the reference word at its aligned ref position.
Pure-insertion words (no ref alignment) get label 0 (always wrong to keep).
"""

import argparse
import functools
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from whisper_normalizer.english import EnglishTextNormalizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rover_variants import words_with_confidence  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('learned_fusion')

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'primock57'
W = EnglishTextNormalizer()
EPS = '<eps>'

FILLERS = {'um', 'uh', 'mm', 'mhm', 'hmm', 'oh', 'ah', 'er', 'huh',
           'yeah', 'yep', 'yes', 'no', 'ok', 'okay', 'hi', 'hello'}
FUNCTION = {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on',
            'at', 'by', 'for', 'with', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'am', 'do', 'does', 'did', 'have', 'has',
            'had', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
            'their', 'this', 'that', 'these', 'those', 'so', 'if', 'as',
            'than', 'then', 'there', 'here', 'when', 'where', 'what',
            'which', 'who', 'how', 'why', 'not'}
WTYPE_KEYS = ['filler', 'function', 'short', 'mid_content', 'long_content']


def classify(w: str) -> str:
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


# ---------- load + expand confidences to WNORM'd hypothesis space ----------


def load(m):
    out = {}
    with open(RESULTS / f'{m}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            out[r['file_id']] = r
    return out


def expand_wnorm(wds: list[dict]) -> list[dict]:
    """Expand each word_with_confidence entry across the WNORM splits of
    its text so the returned list is aligned positionally to
    WNORM(full_hypothesis).split()."""
    out = []
    for d in wds:
        subtoks = W(d['word']).split()
        if not subtoks:
            continue
        for st in subtoks:
            out.append({
                'word': st,
                'top1_mean': d.get('top1_mean', 0.0),
                'shannon_mean': d.get('shannon_mean', 0.0),
                'margin_min': d.get('margin_min', 0.0),
                'n_subwords': d.get('n_subwords', 1),
            })
    return out


# ---------- feature extraction + labels ----------


def align_words_to_ref(ref_words, hyp_words):
    """Return list of length len(ref_words), each entry is the hyp index
    aligned to that ref position (or None). Also return the set of hyp
    indices that are pure insertions."""
    out = [None] * len(ref_words)
    inserted = []
    if not ref_words and not hyp_words:
        return out, inserted
    chunks = jiwer.process_words(
        ' '.join(ref_words), ' '.join(hyp_words)
    ).alignments[0]
    for ch in chunks:
        if ch.type == 'equal':
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                out[ch.ref_start_idx + k] = ch.hyp_start_idx + k
        elif ch.type == 'substitute':
            r_len = ch.ref_end_idx - ch.ref_start_idx
            h_len = ch.hyp_end_idx - ch.hyp_start_idx
            for k in range(min(r_len, h_len)):
                out[ch.ref_start_idx + k] = ch.hyp_start_idx + k
        elif ch.type == 'insert':
            for k in range(ch.hyp_end_idx - ch.hyp_start_idx):
                inserted.append(ch.hyp_start_idx + k)
    return out, inserted


FEATURE_NAMES = (
    [f'wtype_{k}' for k in WTYPE_KEYS]
    + ['word_len', 'n_subwords',
       'own_conf', 'own_shannon', 'own_margin',
       'own_prev_conf', 'own_next_conf',
       'position_frac', 'is_whisper',
       'other_has_here', 'other_here_conf',
       'other_left_conf', 'other_right_conf', 'same_word']
)


def word_features_xmodel(
    wds: list[dict], idx: int, total: int,
    cross_idx: int | None,
    other_wds: list[dict],
    own_idx_to_cross: list[int | None],
    is_whisper: bool,
) -> np.ndarray:
    """Features using hyp-vs-hyp (not ref) cross-model alignment."""
    d = wds[idx]
    wt = classify(d['word'])
    vec = [1.0 if k == wt else 0.0 for k in WTYPE_KEYS]
    vec.append(float(len(d['word'])))
    vec.append(float(d.get('n_subwords', 1)))
    vec.append(float(d.get('top1_mean', 0.0)))
    vec.append(float(d.get('shannon_mean', 0.0)))
    vec.append(float(d.get('margin_min', 0.0)))
    prev_c = wds[idx - 1].get('top1_mean', 0.0) if idx > 0 else 0.0
    next_c = wds[idx + 1].get('top1_mean', 0.0) if idx + 1 < len(wds) else 0.0
    vec.append(float(prev_c))
    vec.append(float(next_c))
    vec.append(float(idx / max(total, 1)))
    vec.append(1.0 if is_whisper else 0.0)

    # Cross-model features via hyp-vs-hyp alignment.
    other_here = 0.0
    other_here_c = 0.0
    other_left_c = 0.0
    other_right_c = 0.0
    same_word = 0.0
    if cross_idx is not None and 0 <= cross_idx < len(other_wds):
        other_here = 1.0
        other_here_c = float(other_wds[cross_idx].get('top1_mean', 0.0))
        if other_wds[cross_idx]['word'] == d['word']:
            same_word = 1.0
    # nearest aligned left/right neighbor in own-index space
    for k in range(idx - 1, -1, -1):
        j = own_idx_to_cross[k]
        if j is not None and 0 <= j < len(other_wds):
            other_left_c = float(other_wds[j].get('top1_mean', 0.0))
            break
    for k in range(idx + 1, len(own_idx_to_cross)):
        j = own_idx_to_cross[k]
        if j is not None and 0 <= j < len(other_wds):
            other_right_c = float(other_wds[j].get('top1_mean', 0.0))
            break
    vec.extend([other_here, other_here_c, other_left_c, other_right_c,
                same_word])
    return np.array(vec, dtype=np.float32)


def extract_file_events(fid, p_rec, w_rec):
    """For one file, return list of events. Each event is
    (features, label, hyp_model_idx, word_idx) where hyp_model_idx is
    'p' or 'w'.

    Cross-model features are derived from a hypothesis-to-hypothesis
    alignment (parakeet <-> whisper) — NOT from reference alignment —
    so the features are computable at inference time without ref.
    Labels are still derived from ref alignment.
    """
    ref = W(p_rec['reference'])
    ref_words = ref.split()
    hp_raw = expand_wnorm(
        words_with_confidence(p_rec['tokens'], p_rec['ys_log_probs'],
                              p_rec['vocab_summaries'])
    )
    hw_raw = expand_wnorm(
        words_with_confidence(w_rec['tokens'], w_rec['ys_log_probs'],
                              w_rec['vocab_summaries'])
    )
    hp_words = [d['word'] for d in hp_raw]
    hw_words = [d['word'] for d in hw_raw]

    # Labels: alignment to reference (train-time only).
    p_ref_align, _ = align_words_to_ref(ref_words, hp_words)
    w_ref_align, _ = align_words_to_ref(ref_words, hw_words)
    p_word_to_ref = [None] * len(hp_words)
    for ri, j in enumerate(p_ref_align):
        if j is not None and 0 <= j < len(hp_words):
            p_word_to_ref[j] = ri
    w_word_to_ref = [None] * len(hw_words)
    for ri, j in enumerate(w_ref_align):
        if j is not None and 0 <= j < len(hw_words):
            w_word_to_ref[j] = ri

    # Cross-model features: parakeet <-> whisper alignment (inference-safe).
    # For each parakeet word j, find whisper word aligned to it (if any)
    # via jiwer's alignment of hp vs hw.
    p_to_w = [None] * len(hp_words)  # whisper index aligned to each parakeet idx
    w_to_p = [None] * len(hw_words)
    if hp_words and hw_words:
        chunks = jiwer.process_words(
            ' '.join(hp_words), ' '.join(hw_words)
        ).alignments[0]
        for ch in chunks:
            if ch.type in ('equal', 'substitute'):
                r_len = ch.ref_end_idx - ch.ref_start_idx
                h_len = ch.hyp_end_idx - ch.hyp_start_idx
                for k in range(min(r_len, h_len)):
                    pi = ch.ref_start_idx + k
                    wi = ch.hyp_start_idx + k
                    p_to_w[pi] = wi
                    w_to_p[wi] = pi

    events = []

    for j in range(len(hp_words)):
        ref_i = p_word_to_ref[j]
        label = 0
        if ref_i is not None and hp_words[j] == ref_words[ref_i]:
            label = 1
        feats = word_features_xmodel(
            hp_raw, j, len(hp_words),
            cross_idx=p_to_w[j],
            other_wds=hw_raw,
            own_idx_to_cross=p_to_w,
            is_whisper=False,
        )
        events.append((feats, label, 'p', j))

    for j in range(len(hw_words)):
        ref_i = w_word_to_ref[j]
        label = 0
        if ref_i is not None and hw_words[j] == ref_words[ref_i]:
            label = 1
        feats = word_features_xmodel(
            hw_raw, j, len(hw_words),
            cross_idx=w_to_p[j],
            other_wds=hp_raw,
            own_idx_to_cross=w_to_p,
            is_whisper=True,
        )
        events.append((feats, label, 'w', j))

    return events, hp_raw, hw_raw


# ---------- CNC with arbitrary per-word vote weights ----------


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
    if w1 == w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    dist = levenshtein(w1, w2)
    mx = max(len(w1), len(w2))
    return max(0.0, 1.0 - dist / mx)


def align_to_slots(slot_words, hyp_words):
    m = len(slot_words)
    n = len(hyp_words)
    if m == 0:
        return [(None, j) for j in range(n)]
    if n == 0:
        return [(i, None) for i in range(m)]
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
            match = dp[i - 1][j - 1] + (2.0 * sim - 1.0)
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


class Slot:
    __slots__ = ('candidates',)

    def __init__(self):
        self.candidates: dict[str, float] = {}

    def vote(self, w, wt):
        self.candidates[w] = self.candidates.get(w, 0.0) + wt

    def head(self):
        if not self.candidates:
            return EPS
        return max(self.candidates, key=self.candidates.get)

    def decode(self):
        if not self.candidates:
            return None
        w = self.head()
        return None if w == EPS else w


def build_cnc_learned(p_wds, w_wds, p_weights, w_weights,
                      eps_del, eps_ins):
    """CNC with parakeet pivot + whisper, using per-word keep-prob weights.
    eps_del/eps_ins are constants (already tuned)."""
    slots: list[Slot] = []
    for j, d in enumerate(p_wds):
        s = Slot()
        s.vote(d['word'], p_weights[j])
        slots.append(s)

    slot_heads = [s.head() for s in slots]
    hw_words = [d['word'] for d in w_wds]
    path = align_to_slots(slot_heads, hw_words)
    new_slots: list[Slot] = []
    i_consumed = 0
    for step in path:
        slot_i, hyp_j = step
        if slot_i is not None and hyp_j is not None:
            while i_consumed < slot_i:
                new_slots.append(slots[i_consumed])
                i_consumed += 1
            s = slots[slot_i]
            s.vote(hw_words[hyp_j], w_weights[hyp_j])
            new_slots.append(s)
            i_consumed = slot_i + 1
        elif slot_i is not None and hyp_j is None:
            while i_consumed < slot_i:
                new_slots.append(slots[i_consumed])
                i_consumed += 1
            s = slots[slot_i]
            s.vote(EPS, eps_del)
            new_slots.append(s)
            i_consumed = slot_i + 1
        else:
            ns = Slot()
            ns.vote(hw_words[hyp_j], w_weights[hyp_j])
            ns.vote(EPS, eps_ins)
            new_slots.append(ns)
    while i_consumed < len(slots):
        new_slots.append(slots[i_consumed])
        i_consumed += 1
    out = []
    for s in new_slots:
        w = s.decode()
        if w is not None:
            out.append(w)
    return ' '.join(out)


# ------------------------------ CV harness ---------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps-del', type=float, default=0.55)
    ap.add_argument('--eps-ins', type=float, default=0.65)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--model', choices=['logreg', 'gbm'], default='gbm')
    ap.add_argument(
        '--mode', choices=['weight', 'gate', 'hybrid'], default='hybrid',
        help='weight: use prob as vote weight; '
             'gate: drop low-prob words then raw-weight CNC; '
             'hybrid: drop if prob<drop_thr, else raw conf',
    )
    ap.add_argument('--drop-thr', type=float, default=0.3,
                    help='gate/hybrid: drop words with prob below this')
    ap.add_argument('--write-features', type=str, default='')
    args = ap.parse_args()

    p_all = load('parakeet-tdt-0.6b-v2')
    w_all = load('whisper-distil-v3.5')
    common = sorted(p_all.keys() & w_all.keys())
    log.info(f'{len(common)} common files')

    # Extract features per file (cached in-memory).
    file_events: dict = {}
    for fid in common:
        events, hp_raw, hw_raw = extract_file_events(fid, p_all[fid], w_all[fid])
        file_events[fid] = {'events': events, 'hp': hp_raw, 'hw': hw_raw}

    total_events = sum(len(v['events']) for v in file_events.values())
    labels_all = [lab for v in file_events.values() for (_, lab, _, _) in v['events']]
    log.info(f'total events: {total_events}, positive rate: '
             f'{sum(labels_all)/len(labels_all)*100:.1f}%')

    # 5-fold GroupKFold by file_id.
    fids = np.array(common)
    groups = np.arange(len(fids))
    gkf = GroupKFold(n_splits=args.n_folds)

    total_err = 0
    total_words = 0
    baseline_err = 0

    fold_idx = 0
    per_fold_wers = []
    for train_gi, test_gi in gkf.split(fids, groups=groups):
        fold_idx += 1
        train_fids = fids[train_gi]
        test_fids = fids[test_gi]

        X_train = []
        y_train = []
        for fid in train_fids:
            for feats, lab, _, _ in file_events[fid]['events']:
                X_train.append(feats)
                y_train.append(lab)
        X_train = np.stack(X_train)
        y_train = np.array(y_train)

        if args.model == 'logreg':
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Xs, y_train)
        else:
            scaler = None
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=0,
            )
            clf.fit(X_train, y_train)

        fold_err = 0
        fold_words = 0
        fold_baseline = 0
        for fid in test_fids:
            v = file_events[fid]
            hp, hw = v['hp'], v['hw']
            feats = np.stack([e[0] for e in v['events']])
            X_in = scaler.transform(feats) if scaler is not None else feats
            probs = clf.predict_proba(X_in)[:, 1]

            # Build per-word weight arrays according to mode.
            p_weights = np.array([d.get('top1_mean', 0.0) for d in hp])
            w_weights = np.array([d.get('top1_mean', 0.0) for d in hw])
            p_keep = np.ones(len(hp), dtype=bool)
            w_keep = np.ones(len(hw), dtype=bool)
            for (_, _, m, wi), pr in zip(v['events'], probs):
                if m == 'p':
                    if args.mode == 'weight':
                        p_weights[wi] = pr
                    elif args.mode == 'gate':
                        if pr < args.drop_thr:
                            p_keep[wi] = False
                    elif args.mode == 'hybrid':
                        if pr < args.drop_thr:
                            p_keep[wi] = False
                else:
                    if args.mode == 'weight':
                        w_weights[wi] = pr
                    elif args.mode == 'gate':
                        if pr < args.drop_thr:
                            w_keep[wi] = False
                    elif args.mode == 'hybrid':
                        if pr < args.drop_thr:
                            w_keep[wi] = False

            # Apply filter: remove dropped words from hyps (and their weights)
            hp_f = [d for d, k in zip(hp, p_keep) if k]
            hw_f = [d for d, k in zip(hw, w_keep) if k]
            p_w_f = np.array([wt for wt, k in zip(p_weights, p_keep) if k])
            w_w_f = np.array([wt for wt, k in zip(w_weights, w_keep) if k])

            hyp = build_cnc_learned(hp_f, hw_f, p_w_f, w_w_f,
                                    eps_del=args.eps_del,
                                    eps_ins=args.eps_ins)
            ref_n = W(p_all[fid]['reference'])
            n = len(ref_n.split())
            if n == 0:
                continue
            er = jiwer.wer(ref_n, W(hyp))
            baseline_wer = jiwer.wer(ref_n, W(p_all[fid]['hypothesis']))
            fold_err += round(er * n)
            fold_baseline += round(baseline_wer * n)
            fold_words += n

        fold_wer = fold_err / fold_words * 100 if fold_words else 0
        fold_base = fold_baseline / fold_words * 100 if fold_words else 0
        per_fold_wers.append(fold_wer)
        log.info(f'fold {fold_idx}: learned={fold_wer:.2f}%  '
                 f'parakeet-solo={fold_base:.2f}%  '
                 f'Δ={fold_wer - fold_base:+.2f}pp  n={len(test_fids)}')

        total_err += fold_err
        total_words += fold_words
        baseline_err += fold_baseline

    agg = total_err / total_words * 100
    base = baseline_err / total_words * 100
    print()
    print('=' * 60)
    print(f'5-fold CV learned CNC: {agg:.2f}%')
    print(f'parakeet solo      : {base:.2f}%')
    print(f'Δ                  : {agg - base:+.2f}pp')
    print('=' * 60)

    # Feature importances on a final full-data refit for interpretation.
    X_all = np.stack([e[0] for v in file_events.values() for e in v['events']])
    y_all = np.array([e[1] for v in file_events.values() for e in v['events']])
    if args.model == 'logreg':
        scaler = StandardScaler().fit(X_all)
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_all), y_all)
        importances = np.abs(clf.coef_[0])
        signed = clf.coef_[0]
        print('\nTop logreg coefficients (standardized):')
        for k in np.argsort(-importances)[:15]:
            print(f'  {FEATURE_NAMES[k]:<22} {signed[k]:+.3f}')
    else:
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=0,
        ).fit(X_all, y_all)
        importances = clf.feature_importances_
        print('\nTop GBM feature importances:')
        for k in np.argsort(-importances)[:15]:
            print(f'  {FEATURE_NAMES[k]:<22} {importances[k]:.3f}')

    if args.write_features:
        np.savez(args.write_features, X=X_all, y=y_all,
                 names=np.array(FEATURE_NAMES))
        print(f'\nfeatures saved to {args.write_features}')


if __name__ == '__main__':
    main()
