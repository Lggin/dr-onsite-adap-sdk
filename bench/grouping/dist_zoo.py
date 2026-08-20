"""벡터 비교 거리 함수 비교 — 유클리드·코사인 말고 무엇이 있나.

지금까지 유클리드/코사인/상관/마할라노비스만 봤다. 축의 성격이 제각각(Lab 색, 물리 mm,
히스토그램, CNN 임베딩)이므로 **블록별로 다른 거리를 쓰는 것**이 물리적으로 타당할 수 있다.

  D1 L2        : 유클리드 (기준)
  D2 L1        : 맨해튼 — 이상치 축 하나에 덜 흔들림
  D3 Linf      : 체비셰프 — "하나라도 크게 다르면 다른 물건"
  D4 Lp0.5     : 분수 거리 — 고차원에서 L2보다 낫다는 보고
  D5 cosine    : 코사인
  D6 correl    : 상관 거리
  D7 spearman  : 순위 상관 — 단조 변환에 불변(조명 강건)
  D8 canberra  : Σ|a-b|/(|a|+|b|) — 작은 값의 차이를 크게 봄
  D9 braycurtis: Σ|a-b|/Σ|a+b| — 비율 기반
  D10 mixed    : **블록별 혼합** — 색=CIEDE2000, 크기=상대오차,
                 질감/CNN=코사인 → 가중합

프로토콜 동일: 종류당 밝은 프레임 medoid 1개 등록 → 어두운 프레임 배정,
종류를 4+4 로 갈라 처음 보는 4종에서 채점.
⚠️ 실패 1~2개 차이는 표본(225개)상 무의미하므로 순위를 과신하면 안 된다.
"""
import os, pickle
from collections import Counter
import numpy as np, cv2
from scipy.stats import rankdata
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL
from sklearn.decomposition import PCA

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
D = pickle.load(open(S + '/axis_power.pkl', 'rb'))
BASE, E, TRUTH, FRAME = D['BASE'], D['E'], D['TRUTH'], D['FRAME']
types = sorted(set(TRUTH))


class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = BASE / scale * bw
Ep = PCA(16, random_state=0).fit_transform(E); Ep = Ep / max(Ep.std(), 1e-9)
Z = np.hstack([Z7, Ep * 0.35])

LUM = np.zeros(len(TRUTH))
for f in set(FRAME):
    LUM[FRAME == f] = float(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2LAB)[..., 0].mean())
SINGLE = np.array(['/' + t + '_single/' in f for t, f in zip(TRUTH, FRAME)])
bright = np.zeros(len(TRUTH), bool)
for t in types:
    idx = np.where((TRUTH == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True
HALF = (['blue_printed', 'navy_long', 'pacific_short', 'plain_small'],
        ['navy_small', 'pacific_long', 'swanson', 'red'])
print(f"객체 {len(Z)} / {len(types)}종 / 특징 {Z.shape[1]}차원")

# ── 블록 정의 (mixed 용) ──────────────────────────────────────────────────
# Lab 은 _build_features 에서 L 에 CLUSTER_L_WEIGHT 가 곱해지고 스케일 나눗셈이 된 상태다.
# CIEDE2000 을 쓰려면 원래 Lab 값이 필요하므로 BASE 에서 되돌린다.
LAB_RAW = BASE[:, :3].copy()
LAB_RAW[:, 0] /= max(cfg.CLUSTER_L_WEIGHT, 1e-6)
MM_RAW = BASE[:, 5:7].copy()
TEX_RAW = BASE[:, 3:5].copy()


def ciede2000(lab1, lab2):
    """CIEDE2000 색차(간략 구현). lab: (...,3) OpenCV Lab(0~255) → 표준 범위로 변환."""
    def conv(x):
        L = x[..., 0] * 100.0 / 255.0
        a = x[..., 1] - 128.0
        b = x[..., 2] - 128.0
        return L, a, b
    L1, a1, b1 = conv(np.atleast_2d(lab1))
    L2, a2, b2 = conv(np.atleast_2d(lab2))
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7 + 1e-12)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2 - h1
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lb = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2
    hb = np.where(np.abs(h1 - h2) > 180, (h1 + h2 + 360) / 2, (h1 + h2) / 2)
    T = (1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb))
         + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63)))
    SL = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7 + 1e-12)) * \
        np.sin(np.radians(60 * np.exp(-((hb - 275) / 25) ** 2)))
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                   + RT * (dCp / SC) * (dHp / SH))


def dist(a_idx, P_idx, kind):
    """객체 a 와 프로토타입들 사이 거리 벡터."""
    A = Z[a_idx]; B = Z[P_idx]
    if kind == 'L2':
        return np.linalg.norm(B - A, axis=1)
    if kind == 'L1':
        return np.abs(B - A).sum(1)
    if kind == 'Linf':
        return np.abs(B - A).max(1)
    if kind == 'Lp0.5':
        return (np.abs(B - A) ** 0.5).sum(1) ** 2
    if kind == 'cosine':
        na = A / max(np.linalg.norm(A), 1e-9)
        nb = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
        return 1 - nb @ na
    if kind == 'correl':
        ca = A - A.mean(); cb = B - B.mean(1, keepdims=True)
        na = ca / max(np.linalg.norm(ca), 1e-9)
        nb = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-9)
        return 1 - nb @ na
    if kind == 'spearman':
        ra = rankdata(A); rb = np.apply_along_axis(rankdata, 1, B)
        ca = ra - ra.mean(); cb = rb - rb.mean(1, keepdims=True)
        na = ca / max(np.linalg.norm(ca), 1e-9)
        nb = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-9)
        return 1 - nb @ na
    if kind == 'canberra':
        return (np.abs(B - A) / np.maximum(np.abs(B) + np.abs(A), 1e-9)).sum(1)
    if kind == 'braycurtis':
        return np.abs(B - A).sum(1) / np.maximum(np.abs(B + A).sum(1), 1e-9)
    if kind == 'mixed':
        # 색: CIEDE2000 / 크기: 상대오차 / 질감·CNN: 코사인
        dc = ciede2000(LAB_RAW[a_idx], LAB_RAW[P_idx]).ravel()
        ms = MM_RAW[P_idx]; ma = MM_RAW[a_idx]
        dsz = np.abs(ms[:, 0] - ma[0]) / np.maximum(ms[:, 0] + ma[0], 1e-9) * 2
        tv = np.hstack([TEX_RAW, Ep])
        ta = tv[a_idx]; tb = tv[P_idx]
        nta = ta / max(np.linalg.norm(ta), 1e-9)
        ntb = tb / np.maximum(np.linalg.norm(tb, axis=1, keepdims=True), 1e-9)
        dtx = 1 - ntb @ nta
        # 각 항을 비슷한 눈금으로: 색차 ~0~50, 상대오차 0~1, 코사인 0~2
        return dc / 20.0 + dsz * 3.0 + dtx * 2.0
    raise ValueError(kind)


KINDS = ['L2', 'L1', 'Linf', 'Lp0.5', 'cosine', 'correl', 'spearman',
         'canberra', 'braycurtis', 'mixed']

P_idx = []
P_lab = []
for t in types:
    m = np.where((TRUTH == t) & bright)[0]
    c = Z[m].mean(0)
    P_idx.append(m[int(np.argmin(np.linalg.norm(Z[m] - c, axis=1)))]); P_lab.append(t)
P_idx = np.array(P_idx); P_lab = np.array(P_lab)

print(f"\n{'거리':<12}{'정확도':>9}{'실패':>6}   주요 오분류")
print("-" * 62)
res = []
for kind in KINDS:
    accs, fails, fmap, n = [], 0, Counter(), 0
    for h in HALF:
        te = np.where(~SINGLE & np.isin(TRUTH, h) & ~bright)[0]
        ok = 0
        for i in te:
            d = dist(i, P_idx, kind)
            pr = P_lab[int(np.argmin(d))]
            if pr == TRUTH[i]:
                ok += 1
            else:
                fmap[(TRUTH[i], pr)] += 1
        accs.append(ok / len(te)); fails += len(te) - ok; n += len(te)
    top = ' · '.join(f'{a}→{b}:{c}' for (a, b), c in fmap.most_common(2))
    res.append((float(np.mean(accs)), kind, fails))
    print(f"{kind:<12}{np.mean(accs)*100:>8.1f}%{fails:>6}   {top}")
res.sort(reverse=True)
print(f"\n최고: {res[0][1]} {res[0][0]*100:.1f}% (실패 {res[0][2]})  "
      f"/ 기준 L2 {[r for r in res if r[1]=='L2'][0][0]*100:.1f}%")
print(f"⚠️ 시험 {n}개 — 실패 1~2개 차이는 무의미")
