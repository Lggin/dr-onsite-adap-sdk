"""쌍 분리력 측정 — 품목 수를 모를 때 맞는 채점 방식.

목적: RFDETR 검출 후 "이 둘이 같은 박스인가 다른 박스인가"를 판단하는 능력.
현장에서는 품목 수가 정해져 있지 않으므로 k 를 알려주는 군집화(kmeans/ward)는
운영과 맞지 않는다. 실제 파이프라인은 **거리 임계(TAU)** 또는 **정답 최근접**이다.

채점: 정답 6종에서
  같은 종류 쌍(positive) 의 거리 분포  vs  다른 종류 쌍(negative) 의 거리 분포
  두 분포가 잘 분리될수록 좋다 → **AUC**(1.0=완벽, 0.5=무작위)
  덤으로 최적 임계에서의 정확도·EER 도 본다.

AUC 는 k 를 몰라도 되고, 임계 하나로 자를 수 있는지를 직접 말해준다.
"""
import os, pickle, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
D = pickle.load(open(S + '/grid_feats.pkl', 'rb'))
FEATS, LAB = D['feats'], D['labels']
TYPES = sorted(set(LAB))
y = np.array([TYPES.index(t) for t in LAB])
print(f"객체 {len(LAB)} / {len(TYPES)}종\n")


def whiten(Z, y_, lam=0.10):
    D_ = np.concatenate([Z[y_ == c] - Z[y_ == c].mean(0) for c in np.unique(y_)])
    Sw = D_.T @ D_ / max(len(D_) - len(np.unique(y_)), 1)
    Sw += lam * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    w, V = np.linalg.eigh(Sw)
    return V @ np.diag(w ** -0.5) @ V.T


def pair_dists(Z, dist):
    """모든 쌍의 거리 행렬."""
    if dist == 'cosine':
        n = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        return 1 - n @ n.T
    if dist == 'correl':
        C = Z - Z.mean(1, keepdims=True)
        n = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        return 1 - n @ n.T
    diff = Z[:, None] - Z[None]
    return np.linalg.norm(diff, axis=2)


def score(Dm, yv):
    iu = np.triu_indices(len(yv), 1)
    d = Dm[iu]
    same = (yv[iu[0]] == yv[iu[1]]).astype(int)
    auc = roc_auc_score(same, -d)            # 거리가 작을수록 같은 종류
    fpr, tpr, thr = roc_curve(same, -d)
    eer_i = np.argmin(np.abs((1 - tpr) - fpr))
    acc = ((tpr + (1 - fpr)) / 2).max()      # 균형 정확도 최대
    return auc, float(fpr[eer_i]), float(acc), same.mean()


print(f"{'피처':<10}{'거리':<9}{'AUC':>8}{'EER':>8}{'균형정확도':>11}")
print("-" * 47)
rows = []
for fname, X in FEATS.items():
    Z0 = StandardScaler().fit_transform(X)
    for dist in ('euclid', 'cosine', 'correl'):
        auc, eer, acc, pr = score(pair_dists(Z0, dist), y)
        rows.append((fname, dist, auc, eer, acc))
        print(f"{fname:<10}{dist:<9}{auc:>8.3f}{eer:>8.3f}{acc:>11.3f}")

# 마할라노비스는 교차검증(3종 학습 → 3종 평가)으로만 잰다
print(f"\n{'피처':<10}{'거리':<9}{'AUC(교차검증)':>14}{'±':>8}")
print("-" * 43)
splits = [tuple(c) for c in itertools.combinations(TYPES, 3)]
splits = splits[:len(splits) // 2]
for fname, X in FEATS.items():
    Zall = StandardScaler().fit_transform(X)
    aucs = []
    for tr in splits:
        te = tuple(t for t in TYPES if t not in tr)
        for a, b in ((tr, te), (te, tr)):
            mtr, mte = np.isin(LAB, a), np.isin(LAB, b)
            ytr = np.array([sorted(a).index(t) for t in LAB[mtr]])
            M = whiten(Zall[mtr], ytr)
            Zt = Zall @ M.T
            yte = np.array([sorted(b).index(t) for t in LAB[mte]])
            aucs.append(score(pair_dists(Zt[mte], 'euclid'), yte)[0])
    rows.append((fname, 'maha(cv)', float(np.mean(aucs)), None, None))
    print(f"{fname:<10}{'maha':<9}{np.mean(aucs):>14.3f}{np.std(aucs):>8.3f}")

# 같은 조건(3종 부분집합)에서 비학습 거리도 재야 공정 비교
print(f"\n── 같은 3종 부분집합 기준 공정 비교 ──")
print(f"{'피처':<10}{'거리':<9}{'AUC':>8}{'±':>8}")
for fname, X in FEATS.items():
    Zall = StandardScaler().fit_transform(X)
    for dist in ('euclid', 'cosine'):
        aucs = []
        for tr in splits:
            te = tuple(t for t in TYPES if t not in tr)
            for b in (tr, te):
                m = np.isin(LAB, b)
                yb = np.array([sorted(b).index(t) for t in LAB[m]])
                aucs.append(score(pair_dists(Zall[m], dist), yb)[0])
        print(f"{fname:<10}{dist:<9}{np.mean(aucs):>8.3f}{np.std(aucs):>8.3f}")
pickle.dump(rows, open(S + '/pairwise_auc.pkl', 'wb'))
