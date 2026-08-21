"""색·질감 공간 분석 — ultimate(라벨없음) 군집 + box_types(정답 6종) 중첩."""
import os, pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FP)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FP).get_name()
plt.rcParams['axes.unicode_minus'] = False
S = os.environ['S']
OUT = os.path.expanduser('~/eda_이미지')
os.makedirs(OUT, exist_ok=True)

rows = pickle.load(open(S + '/eda_objs.pkl', 'rb'))
X = np.array([r['feats'] for r in rows])
src = np.array([r['src'] for r in rows])
lab = np.array([r['label'] for r in rows])
AX = ['L(밝기)', 'a(적녹)', 'b(황청)', 'tex1(하이패스)', 'tex2(에지밀도)']

# ── 1. 축별 기초 통계: 학습셋 vs 현장 ────────────────────────────────────
u, b = src == 'ultimate', src == 'box_types'
print("=" * 92)
print("① 축별 분포 — 학습셋(ultimate) vs 현장 실촬(box_types)")
print(f"{'축':<16}{'학습셋 중앙(IQR)':>26}{'현장 중앙(IQR)':>26}{'표준화 거리':>12}")
for i, nm in enumerate(AX):
    us, bs = X[u, i], X[b, i]
    uq = np.percentile(us, [25, 50, 75]); bq = np.percentile(bs, [25, 50, 75])
    d = abs(bq[1] - uq[1]) / (us.std() + 1e-9)
    print(f"{nm:<16}{f'{uq[1]:6.1f} ({uq[0]:.0f}~{uq[2]:.0f})':>26}"
          f"{f'{bq[1]:6.1f} ({bq[0]:.0f}~{bq[2]:.0f})':>26}{d:>11.2f}σ")

# ── 2. 학습셋 Carton 군집 = 숨은 품목 구조 ───────────────────────────────
sc = StandardScaler().fit(X[u])
Zu = sc.transform(X[u])
print("\n" + "=" * 92)
print("② 학습셋 Carton 의 숨은 군집 구조 (라벨이 없으므로 비지도)")
best = None
for k in range(2, 11):
    km = KMeans(k, n_init=6, random_state=0).fit(Zu)
    s = silhouette_score(Zu[::5], km.labels_[::5])
    print(f"   k={k:2d}  실루엣 {s:.3f}   군집크기 {sorted(Counter(km.labels_).values(), reverse=True)[:6]}")
    if best is None or s > best[1]:
        best = (k, s, km)
K, SIL, KM = best
print(f"   → 최적 k={K} (실루엣 {SIL:.3f})")

# ── 3. 현장 6종이 그 군집 어디에 떨어지나 ────────────────────────────────
Zb = sc.transform(X[b])
assign = KM.predict(Zb)
print("\n" + "=" * 92)
print("③ 현장 6종이 학습셋 군집 어디에 배정되나 (학습셋이 현장을 덮는가)")
print(f"{'종류':<16}{'n':>5}  군집 분포")
for t in sorted(set(lab[b])):
    m = lab[b] == t
    c = Counter(assign[m])
    tot = m.sum()
    dist = ', '.join(f'C{k}:{v}({v/tot*100:.0f}%)' for k, v in c.most_common(3))
    # 배정된 군집 중심까지의 거리 = 학습셋에 얼마나 가까운가
    dmin = np.linalg.norm(Zb[m][:, None] - KM.cluster_centers_[None], axis=2).min(1)
    print(f"{t:<16}{tot:>5}  {dist}   중심거리 중앙 {np.median(dmin):.2f}")
uc = np.linalg.norm(Zu[:, None] - KM.cluster_centers_[None], axis=2).min(1)
print(f"\n   (참고) 학습셋 자체의 중심거리 중앙 {np.median(uc):.2f}, 90%ile {np.percentile(uc,90):.2f}")

# ── 4. PCA 2D 시각화 ─────────────────────────────────────────────────────
p = PCA(2, random_state=0).fit(Zu)
Pu, Pb = p.transform(Zu), p.transform(Zb)
fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
a0 = axs[0]
orig = np.array([r['origin'] for r in rows])[u]
for o, c in (('외부/기타', '#95a5a6'), ('합성', '#3498db'), ('현장실촬', '#e74c3c')):
    m = orig == o
    a0.scatter(Pu[m, 0], Pu[m, 1], s=5, alpha=.35, c=c, label=f'{o} ({m.sum():,})')
a0.set_title(f'학습셋(ultimate) Carton {u.sum():,}개 — 출처별', fontsize=12)
a0.legend(fontsize=9, markerscale=3); a0.grid(alpha=.3)
a0.set_xlabel(f'PC1 ({p.explained_variance_ratio_[0]*100:.0f}%)')
a0.set_ylabel(f'PC2 ({p.explained_variance_ratio_[1]*100:.0f}%)')

a1 = axs[1]
a1.scatter(Pu[:, 0], Pu[:, 1], s=5, alpha=.15, c='#bdc3c7', label='학습셋')
cols = plt.cm.tab10(np.linspace(0, .9, 6))
for c, t in zip(cols, sorted(set(lab[b]))):
    m = lab[b] == t
    a1.scatter(Pb[m, 0], Pb[m, 1], s=26, alpha=.85, color=c, edgecolor='k',
               linewidth=.3, label=f'{t} ({m.sum()})')
a1.set_title('현장 실촬 6종을 같은 공간에 겹침', fontsize=12)
a1.legend(fontsize=8.5, markerscale=1.4); a1.grid(alpha=.3)
a1.set_xlabel('PC1'); a1.set_ylabel('PC2')
fig.suptitle('색·질감 공간 — 학습셋이 현장 박스를 덮는가', fontsize=14)
fig.tight_layout(); fig.savefig(OUT + '/eda1_space.png', dpi=110, facecolor='w'); plt.close(fig)

# ── 5. 축별 히스토그램 비교 ──────────────────────────────────────────────
fig, axs = plt.subplots(1, 5, figsize=(19, 3.6))
for i, (ax, nm) in enumerate(zip(axs, AX)):
    ax.hist(X[u, i], bins=45, alpha=.55, color='#3498db', density=True, label='학습셋')
    ax.hist(X[b, i], bins=25, alpha=.65, color='#e74c3c', density=True, label='현장')
    ax.set_title(nm, fontsize=11); ax.grid(alpha=.3)
    if i == 0:
        ax.legend(fontsize=9)
fig.suptitle('축별 분포 비교 — 겹치지 않는 구간이 도메인 간극', fontsize=13)
fig.tight_layout(); fig.savefig(OUT + '/eda2_hist.png', dpi=110, facecolor='w'); plt.close(fig)

pickle.dump(dict(X=X, src=src, lab=lab, sc=sc, KM=KM, pca=p), open(S + '/eda_model.pkl', 'wb'))
print("\n그림:", OUT)
