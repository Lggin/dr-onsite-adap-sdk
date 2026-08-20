"""박스 유사도 그리드 서치 ② — 피처 × 거리 × 연결방식 전조합 채점.

정답(6종)이 있으므로 "무엇이 좋은가"를 감이 아니라 점수로 고른다.

채점 지표
  ARI  : 조정 랜드 지수. 군집 결과가 정답 분할과 얼마나 일치하나(1=완벽, 0=무작위)
  NMI  : 정규화 상호정보량. 같은 취지의 다른 척도
  purity: 각 군집을 다수결 라벨로 봤을 때의 정확도
  k    : 만들어진 군집 수 (정답은 6)

거리
  euclid   : 표준화 후 유클리드 (현행)
  cosine   : 코사인 거리 — 벡터 크기를 버리고 방향만
  correl   : 상관 거리 — 각 벡터를 평균 제거 후 코사인(패턴 모양만 비교)
  maha     : 마할라노비스 = 클래스 내 산포로 화이트닝한 유클리드(=학습된 거리)
             ⚠️ 정답을 써서 학습하므로 **낙관적 상한**이다. 교차검증으로 별도 표기.
"""
import os, pickle, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
D = pickle.load(open(S + '/grid_feats.pkl', 'rb'))
FEATS, LAB = D['feats'], D['labels']
TYPES = sorted(set(LAB))
NG = len(TYPES)
y = np.array([TYPES.index(t) for t in LAB])
print(f"객체 {len(LAB)} / 정답 {NG}종 {dict(zip(*np.unique(LAB, return_counts=True)))}\n")


def purity(y_true, y_pred):
    tot = 0
    for c in set(y_pred):
        m = y_pred == c
        if m.sum():
            tot += np.bincount(y_true[m]).max()
    return tot / len(y_true)


def whiten(Z, y_, lam=0.10):
    """클래스 내 산포 S_W 로 화이트닝(마할라노비스). y_ 를 쓰므로 낙관적."""
    D_ = np.concatenate([Z[y_ == c] - Z[y_ == c].mean(0) for c in np.unique(y_)])
    Sw = D_.T @ D_ / max(len(D_) - len(np.unique(y_)), 1)
    Sw += lam * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    w, V = np.linalg.eigh(Sw)
    return V @ np.diag(w ** -0.5) @ V.T


def dist_matrix(Z, kind, y_=None):
    if kind == 'euclid':
        return None, Z                       # 원공간 그대로
    if kind == 'cosine':
        n = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        return 1 - n @ n.T, None
    if kind == 'correl':
        C = Z - Z.mean(1, keepdims=True)
        n = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        return 1 - n @ n.T, None
    if kind == 'maha':
        return None, Z @ whiten(Z, y_).T
    raise ValueError(kind)


rows = []
for fname, X in FEATS.items():
    Z = StandardScaler().fit_transform(X)
    for dkind in ('euclid', 'cosine', 'correl', 'maha'):
        Dm, Zt = dist_matrix(Z, dkind, y)
        for link in ('complete', 'average', 'ward', 'kmeans'):
            if link == 'ward' and Dm is not None:
                continue                     # ward 는 유클리드 공간에서만 정의
            if link == 'kmeans':
                if Dm is not None:
                    continue
                pred = KMeans(NG, n_init=10, random_state=0).fit_predict(Zt)
            elif Dm is not None:
                pred = AgglomerativeClustering(n_clusters=NG, metric='precomputed',
                                               linkage=link).fit_predict(Dm)
            else:
                pred = AgglomerativeClustering(n_clusters=NG,
                                               linkage=link).fit_predict(Zt)
            rows.append(dict(feat=fname, dist=dkind, link=link,
                             ari=adjusted_rand_score(y, pred),
                             nmi=normalized_mutual_info_score(y, pred),
                             pur=purity(y, pred), k=len(set(pred))))

rows.sort(key=lambda r: -r['ari'])
print(f"{'피처':<10}{'거리':<9}{'연결':<10}{'ARI':>7}{'NMI':>7}{'purity':>8}")
print("-" * 52)
for r in rows:
    star = ' ←' if r is rows[0] else ''
    mark = '*' if r['dist'] == 'maha' else ' '
    print(f"{r['feat']:<10}{r['dist']:<9}{r['link']:<10}"
          f"{r['ari']:>7.3f}{r['nmi']:>7.3f}{r['pur']:>8.3f}{mark}{star}")
print("\n* = 마할라노비스는 정답으로 학습하므로 낙관적 상한(교차검증 별도 필요)")

print("\n── 피처 세트별 최고(마할라노비스 제외) ──")
for f in FEATS:
    best = max([r for r in rows if r['feat'] == f and r['dist'] != 'maha'],
               key=lambda r: r['ari'])
    print(f"  {f:<10} ARI {best['ari']:.3f}  ({best['dist']}, {best['link']})")

print("\n── 거리별 최고(F3_rich 기준) ──")
for d in ('euclid', 'cosine', 'correl', 'maha'):
    c = [r for r in rows if r['feat'] == 'F3_rich' and r['dist'] == d]
    if c:
        b = max(c, key=lambda r: r['ari'])
        print(f"  {d:<9} ARI {b['ari']:.3f}  ({b['link']})")
pickle.dump(rows, open(S + '/grid_results.pkl', 'wb'))
