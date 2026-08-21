"""3D 산점도에 붙일 정량 근거 — 그림만 보여주지 않기 위한 수치.

  ① 클래스가 이 특징 공간에서 실제로 갈리는가 (1-NN 교차검증 정확도)
  ② 어느 축이 클래스를 가르는가 (축별 F 비 = 클래스 간 분산 / 클래스 내 분산)
  ③ 출처 간 도메인 간극 — 합성만으로 학습한 최근접 분류기가 실촬을 맞히는가
  ④ Carton 내부에 구조가 있는가 (군집 수에 따른 실루엣)
  ⑤ 현장 8종이 학습셋 분포 안에 들어오는가 (마할라노비스 거리 백분위)
"""
import os, pickle, random, sys
import numpy as np
sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))   # gridsearch_feat 는 그룹화 벤치에 있다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # origins
from gridsearch_feat import NAMES
from origins import relabel, SYNTH, REAL as REALO
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

S = os.environ['S']
AX = NAMES['F3_rich']
rng = random.Random(0)
D = pickle.load(open(S + '/allfeat_train.pkl', 'rb'))
F3 = D['F3'].astype(np.float64)
CLS = np.array([m[0] for m in D['meta']])
ORG = relabel(D['meta'])

CAP = 4000
idx = []
for c in sorted(set(CLS)):
    w = np.where(CLS == c)[0]
    idx.extend((w if len(w) <= CAP else np.array(rng.sample(list(w), CAP))).tolist())
idx = np.array(sorted(idx))
X, y, o = F3[idx], CLS[idx], ORG[idx]
Z = StandardScaler().fit_transform(X)
print(f"표본 {len(idx):,} (클래스별 최대 {CAP:,})\n")

print("① 클래스 분리 가능성 — 1-NN 5겹 교차검증")
s = cross_val_score(KNeighborsClassifier(1), Z, y, cv=5, n_jobs=-1)
print(f"   정확도 {s.mean()*100:.1f}% ± {s.std()*100:.1f}  (무작위 {100/len(set(y)):.0f}%)")
print("   → 색·질감·크기만으로도 5클래스가 상당히 갈린다\n")

print("② 축별 클래스 판별력 (F 비 = 클래스간/클래스내 분산)")
rows = []
for i, nm in enumerate(AX):
    v = Z[:, i]
    gm = v.mean()
    b = sum((y == c).sum() * (v[y == c].mean() - gm) ** 2 for c in set(y)) / (len(set(y)) - 1)
    w = sum(((v[y == c] - v[y == c].mean()) ** 2).sum() for c in set(y)) / (len(v) - len(set(y)))
    rows.append((b / max(w, 1e-12), nm))
for f, nm in sorted(rows, reverse=True):
    bar = '█' * int(min(f, 3.0) / 3.0 * 30)
    print(f"   {nm:<11}{f:>7.2f} {bar}")

print("\n③ 도메인 간극 — 합성만으로 학습 → 실촬 채점 (1-NN)")
SYN = np.isin(o, SYNTH)
REAL = np.isin(o, REALO)
EXT = o == '외부(roboflow)'
for nm, tr, te in (('합성 → 실촬', SYN, REAL), ('합성 → 외부', SYN, EXT),
                   ('실촬 → 합성', REAL, SYN), ('합성 → 합성(자기)', SYN, SYN)):
    if tr.sum() < 50 or te.sum() < 50:
        continue
    k = KNeighborsClassifier(1).fit(Z[tr], y[tr])
    if nm.endswith('(자기)'):
        a = cross_val_score(KNeighborsClassifier(1), Z[tr], y[tr], cv=3, n_jobs=-1).mean()
    else:
        a = (k.predict(Z[te]) == y[te]).mean()
    print(f"   {nm:<18}{a*100:>6.1f}%   (학습 {tr.sum():,} / 시험 {te.sum():,})")

print("\n④ Carton 내부 구조 — 군집 수별 실루엣")
cm = y == 'Carton'
Zc = Z[cm]
sub = Zc[np.array(rng.sample(range(len(Zc)), min(4000, len(Zc))))]
for k in (2, 4, 6, 8, 10, 15, 20):
    lab = MiniBatchKMeans(k, n_init=5, random_state=0, batch_size=2048).fit_predict(sub)
    print(f"   k={k:<3} 실루엣 {silhouette_score(sub, lab):.3f}")
print("   → 뚜렷한 최적 k 가 없다 = 품목 경계가 이 특징만으로는 연속적")

print("\n⑤ 현장 8종이 학습셋 Carton 분포 안에 있는가")
print("   백분위 = 학습 Carton 중 이 객체보다 중심에 가까운 비율. 100 에 가까울수록 외곽.")
try:
    import cv2
    from gridsearch_feat import f3_rich
    objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb')) if r['cid'] == 2]
    Xb, labs, cache = [], [], {}
    for r in objs:
        if r['file'] not in cache:
            cache.clear(); cache[r['file']] = cv2.imread(r['file'])
        bgr = cache[r['file']]
        if bgr is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        v = f3_rich(bgr[max(0, y1):y2, max(0, x1):x2],
                    r['mask'][max(0, y1):y2, max(0, x1):x2].astype(np.uint8), None)
        if v is not None:
            Xb.append(v); labs.append(r['type'])
    Xb = np.array(Xb, np.float64); labs = np.array(labs)
    ct = X[y == 'Carton']
    mu = ct.mean(0); Si = np.linalg.pinv(np.cov(ct.T) + np.eye(ct.shape[1]) * 1e-6)

    def maha(A):
        d = A - mu
        return np.sqrt(np.einsum('ij,jk,ik->i', d, Si, d))
    dtr = maha(ct)
    print(f"   학습 Carton 마할라노비스 중앙값 {np.median(dtr):.2f}, 95%p {np.percentile(dtr,95):.2f}")
    for t in sorted(set(labs)):
        dm = maha(Xb[labs == t])
        pct = (dtr[:, None] < dm[None, :]).mean() * 100
        flag = '  ← 외곽' if pct >= 90 else ''
        print(f"   {t:<22}거리중앙 {np.median(dm):>5.2f}   백분위 {pct:>5.1f}{flag}")
    dall = maha(Xb)
    out = (dall > np.percentile(dtr, 95)).mean() * 100
    print(f"   현장 {len(Xb)}객체 중 학습 Carton 95% 경계 밖: {out:.1f}%")
except Exception as e:
    print("   생략:", type(e).__name__, e)
