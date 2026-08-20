"""정답 등록(프로토타입) 경로에서 코사인 vs 유클리드 — 운영 조건 그대로.

8/14 실험(box_eval/box_reg_source)과 같은 프로토콜:
  · 특징 = 실운영 경로(물리 mm 크기 + 워프패치 질감) 7차원
  · 등록 = 밝은 프레임의 medoid 1개  /  시험 = 어두운 프레임 (실제 조명 on/off)
  · 종류를 3+3 으로 갈라 **처음 보는 종류**에서 채점(거리 학습이 낀 조건의 공정성)

이번에 추가로 보는 것
  · 거리: 유클리드(현행) vs 코사인
  · 크기 축 처리: 코사인은 벡터 방향만 보므로 물리 크기(mm)의 스케일이 큰 축을
    어떻게 다루는지가 관건이다 → 크기 포함/제외를 함께 잰다
  · 등록 방식: 쌓인 상태 vs 낱개 (8/14 결론 재확인)
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
Z7 = FEAT / scale * bw                      # 실운영 7차원(색3+질감2+크기2)
Z5 = Z7[:, :5]                              # 크기 제외
print(f"객체 {len(Z7)} / {len(types)}종  (7차원 = 색3+질감2+크기2)")

# 쌓인 것 중 밝은 절반 = 등록 후보
bright = np.zeros(len(Z7), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True


def dists(A, B, kind):
    if kind == 'cosine':
        na = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
        nb = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
        return 1 - na @ nb.T
    return np.linalg.norm(A[:, None] - B[None], axis=2)


def whiten(Z, y_, lam=0.10):
    D_ = np.concatenate([Z[y_ == c] - Z[y_ == c].mean(0) for c in np.unique(y_)])
    Sw = D_.T @ D_ / max(len(D_) - len(np.unique(y_)), 1)
    Sw += lam * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    w, V = np.linalg.eigh(Sw)
    M = V @ np.diag(w ** -0.5) @ V.T
    rng = np.random.RandomState(0)
    i = rng.choice(len(Z), min(300, len(Z)), False)
    d0 = np.linalg.norm(Z[i][:, None] - Z[i][None], axis=2)
    Za = Z[i] @ M.T
    d1 = np.linalg.norm(Za[:, None] - Za[None], axis=2)
    return M * (np.median(d0[d0 > 0]) / np.median(d1[d1 > 0]))


# 임계는 거리 종류마다 눈금이 다르므로, 등록 집합 안에서 분위로 자동 설정한다
def auto_tau(Zx, reg_idx, kind, q=97):
    d = dists(Zx[reg_idx], Zx[reg_idx], kind)
    iu = np.triu_indices(len(reg_idx), 1)
    return float(np.percentile(d[iu], 100 - q)) * 3.0


def run(Zx, kind, reg_mode, subset, tau=None):
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
    te = np.where(~SINGLE & np.isin(LAB, subset))[0]
    if reg_mode == 'stack':
        te = te[~bright[te]]
    d = dists(Zx[te], P, kind)
    j = d.argmin(1)
    if tau is None:
        pred = PL[j]                                  # 임계 없이 최근접만
    else:
        pred = np.where(d.min(1) < tau, PL[j], '기타')
    return float((pred == LAB[te]).mean()), len(te)


HALF = (['blue_printed', 'navy_long', 'pacific_short'],
        ['navy_small', 'pacific_long', 'swanson'])

print(f"\n{'특징':<14}{'거리':<9}{'등록':<12}{'정확도(처음 보는 종류)':>22}")
print("-" * 58)
for zname, Zbase in (('7차원(크기포함)', Z7), ('5차원(색+질감)', Z5)):
    for kind in ('euclid', 'cosine'):
        for reg, rname in (('stack', '쌓인상태'), ('single', '낱개')):
            accs = []
            for te_t in HALF:
                accs.append(run(Zbase, kind, reg, te_t)[0])
            print(f"{zname:<14}{kind:<9}{rname:<12}{np.mean(accs)*100:>21.1f}%")

print(f"\n{'특징':<14}{'거리':<20}{'정확도':>12}   (등록=쌓인상태)")
print("-" * 58)
for zname, Zbase in (('7차원', Z7), ('5차원', Z5)):
    for kind in ('euclid', 'cosine'):
        accs = []
        for tr, te_t in ((HALF[0], HALF[1]), (HALF[1], HALF[0])):
            m = np.isin(LAB, tr)
            ytr = np.array([sorted(tr).index(t) for t in LAB[m]])
            M = whiten(Zbase[m], ytr)
            accs.append(run(Zbase @ M.T, kind, 'stack', te_t)[0])
        print(f"{zname:<14}{'학습된거리+' + kind:<20}{np.mean(accs)*100:>11.1f}%")
