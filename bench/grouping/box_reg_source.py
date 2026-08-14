"""등록 방식 비교 — 낱개로 등록 vs 쌓인 상태로 등록.

현장 운영 절차를 정하는 실험이다. "박스를 하나 꺼내 깨끗하게 찍어 등록"이 편하지만,
정작 운영은 빽빽하게 쌓인 상태다. 옆 박스에 가려 마스크가 잘리고 그림자가 지므로
특징이 체계적으로 달라질 수 있다 — 그 차이가 실제로 얼마나 되는지 잰다.

시험 대상은 **쌓인 상태 객체 전부**(조명 밝음/어두움 모두). 거리 학습은 앞 실험과 같은
교차검증(3종 학습 → 처음 보는 3종 평가)을 유지해 등록 방식의 효과만 분리한다.
"""
import os, pickle
from collections import defaultdict
import numpy as np, cv2
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb'))
        if r['cid'] == 2 and r['size_mm'] and r['geom']]

def ident(t):
    return t[:-7] if t.endswith('_single') else t

class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()

byfile = defaultdict(list)
for r in objs:
    byfile[r['file']].append(r)

FEAT, LAB, SINGLE, LUM = [], [], [], []
for f, rs in byfile.items():
    img = cv2.imread(f)
    geoms = [(r['geom']['center_3d'], r['geom']['long_axis_3d'], r['geom']['short_axis_3d'],
              r['geom']['long_side_mm'], r['geom']['short_side_mm']) for r in rs]
    fs, _ = CL._build_features([r['mask'] for r in rs], [r['size_mm'] for r in rs],
                               geoms, img, rs[0]['intr'], cfg)
    L = float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean())
    for r, ft in zip(rs, fs):
        if ft is None:
            continue
        FEAT.append(ft); LAB.append(ident(r['type']))
        SINGLE.append(r['type'].endswith('_single')); LUM.append(L)
FEAT = np.stack(FEAT); LAB = np.array(LAB)
SINGLE = np.array(SINGLE); LUM = np.array(LUM)
types = sorted(set(LAB))
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z = FEAT / scale * bw
MATCH_TAU = config.CLUSTER_MATCH_TAU
print("낱개 객체 수:", {t: int(((LAB == t) & SINGLE).sum()) for t in types})
print("쌓인 객체 수:", {t: int(((LAB == t) & ~SINGLE).sum()) for t in types})

# 쌓인 것 중 밝은 절반(등록 후보) 표시
bright = np.zeros(len(Z), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    thr = np.median(LUM[idx])
    bright[idx[LUM[idx] >= thr]] = True


def learn(train_types, lam=0.10):
    D = np.concatenate([Z[LAB == t] - Z[LAB == t].mean(0) for t in train_types])
    S_W = D.T @ D / max(len(D) - len(train_types), 1)
    S_W += lam * np.trace(S_W) / S_W.shape[0] * np.eye(S_W.shape[0])
    w, V = np.linalg.eigh(S_W)
    M = V @ np.diag(w ** -0.5) @ V.T
    rng = np.random.RandomState(0); idx = rng.choice(len(Z), min(400, len(Z)), False)
    db = np.linalg.norm(Z[idx][:, None] - Z[idx][None], axis=2)
    Za = Z[idx] @ M.T
    da = np.linalg.norm(Za[:, None] - Za[None], axis=2)
    return M * (np.median(db[db > 0]) / np.median(da[da > 0]))


def run(Zx, reg_mode, subset):
    """reg_mode: 'single'=낱개 전부 등록, 'stack'=쌓인 밝은 것 medoid 1개"""
    P, PL = [], []
    for t in types:
        if reg_mode == 'single':
            m = np.where((LAB == t) & SINGLE)[0]
        else:
            m = np.where((LAB == t) & bright)[0]
            c = Zx[m].mean(0)
            m = [m[int(np.argmin(np.linalg.norm(Zx[m] - c, axis=1)))]]
        for i in m:
            P.append(Zx[i]); PL.append(t)
    P = np.stack(P); PL = np.array(PL)
    te = np.where(~SINGLE & np.isin(LAB, subset))[0]     # 쌓인 것만 시험
    if reg_mode == 'stack':
        te = te[~bright[te]]                             # 등록에 쓴 밝은 쪽 제외
    d = np.linalg.norm(Zx[te][:, None] - P[None], axis=2)
    pred = np.where(d.min(1) < MATCH_TAU, PL[d.argmin(1)], '기타')
    return float((pred == LAB[te]).mean()), float((pred == '기타').mean()), len(te)


HALF = (['blue_printed', 'navy_long', 'pacific_short'],
        ['navy_small', 'pacific_long', 'swanson'])
print(f"\n{'거리':<14}{'등록 방식':<22}{'정확도':>8}{'기타':>8}{'n':>6}")
for mname in ('고정 가중', '학습된 거리'):
    for reg, rname in (('single', '낱개 등록'), ('stack', '쌓인 상태 등록(밝을때)')):
        accs, oths, ns = [], [], 0
        for tr, te_t in ((HALF[0], HALF[1]), (HALF[1], HALF[0])):
            Zx = Z if mname == '고정 가중' else Z @ learn(tr).T
            a, o, n = run(Zx, reg, te_t)
            accs.append(a); oths.append(o); ns += n
        print(f"{mname:<14}{rname:<22}{np.mean(accs)*100:>7.1f}%{np.mean(oths)*100:>7.1f}%{ns:>6}")
