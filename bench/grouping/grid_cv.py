"""마할라노비스 교차검증 — 0.810 이 진짜인가.

그리드 서치의 마할라노비스는 **정답으로 S_W 를 학습한 뒤 같은 데이터로 채점**했으므로
낙관적 상한이다. 진짜 값은 "처음 보는 종류"에서 재야 한다.

방법: 6종을 3+3 으로 나눠 한쪽으로 거리를 배우고 **다른 쪽에서 군집화·채점**한다.
      분할이 하나면 운에 좌우되므로 가능한 10가지 분할을 모두 돌려 평균·표준편차를 본다.
      비교 대상(유클리드·코사인)도 **같은 3종 부분집합**에서 재야 공정하다.
"""
import os, pickle, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
D = pickle.load(open(S + '/grid_feats.pkl', 'rb'))
FEATS, LAB = D['feats'], D['labels']
TYPES = sorted(set(LAB))
print(f"객체 {len(LAB)} / {len(TYPES)}종\n")


def whiten(Z, y_, lam=0.10):
    D_ = np.concatenate([Z[y_ == c] - Z[y_ == c].mean(0) for c in np.unique(y_)])
    Sw = D_.T @ D_ / max(len(D_) - len(np.unique(y_)), 1)
    Sw += lam * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    w, V = np.linalg.eigh(Sw)
    return V @ np.diag(w ** -0.5) @ V.T


def cluster(Z, k, dist, link):
    if dist == 'cosine':
        n = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        Dm = 1 - n @ n.T
        if link == 'kmeans':
            return None
        return AgglomerativeClustering(n_clusters=k, metric='precomputed',
                                       linkage=link).fit_predict(Dm)
    if link == 'kmeans':
        return KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    return AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(Z)


COMBOS = [('euclid', 'kmeans'), ('euclid', 'ward'),
          ('cosine', 'complete'), ('cosine', 'average'),
          ('maha', 'ward'), ('maha', 'kmeans')]

splits = [tuple(c) for c in itertools.combinations(TYPES, 3)]
splits = splits[:len(splits) // 2]          # 대칭 제거(A|B == B|A)
print(f"분할 {len(splits)}가지 × 양방향 = {len(splits)*2}회 평가\n")

out = {}
for fname, X in FEATS.items():
    Zall = StandardScaler().fit_transform(X)
    for dist, link in COMBOS:
        scores = []
        for tr in splits:
            te = tuple(t for t in TYPES if t not in tr)
            for a, b in ((tr, te), (te, tr)):
                mtr = np.isin(LAB, a)
                mte = np.isin(LAB, b)
                Z = Zall.copy()
                if dist == 'maha':
                    # 학습 조에서만 화이트닝 행렬을 배운다
                    ytr = np.array([sorted(a).index(t) for t in LAB[mtr]])
                    M = whiten(Zall[mtr], ytr)
                    Z = Zall @ M.T
                yte = np.array([sorted(b).index(t) for t in LAB[mte]])
                pred = cluster(Z[mte], len(b), dist, link)
                if pred is None:
                    continue
                scores.append(adjusted_rand_score(yte, pred))
        if scores:
            out[(fname, dist, link)] = (float(np.mean(scores)), float(np.std(scores)),
                                        len(scores))

print(f"{'피처':<10}{'거리':<8}{'연결':<10}{'ARI 평균':>10}{'표준편차':>9}{'n':>5}")
print("-" * 54)
for (f, d, l), (m, s, n) in sorted(out.items(), key=lambda kv: -kv[1][0]):
    print(f"{f:<10}{d:<8}{l:<10}{m:>10.3f}{s:>9.3f}{n:>5}")

print("\n── 같은 조건 비교: 마할라노비스가 정말 나은가 ──")
for f in FEATS:
    base = max([(k, v) for k, v in out.items() if k[0] == f and k[1] != 'maha'],
               key=lambda kv: kv[1][0])
    mah = max([(k, v) for k, v in out.items() if k[0] == f and k[1] == 'maha'],
              key=lambda kv: kv[1][0], default=None)
    if mah:
        d = mah[1][0] - base[1][0]
        print(f"  {f:<10} 최고(비학습) {base[1][0]:.3f} ({base[0][1]},{base[0][2]})"
              f"   마할라 {mah[1][0]:.3f} ({mah[0][2]})   차이 {d:+.3f}")
pickle.dump(out, open(S + '/grid_cv.pkl', 'wb'))
