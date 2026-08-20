"""소프트 크기 게이트 — 자르지 않고 벌점으로.

하드 게이트(±25mm 컷)는 정답을 7.7% 탈락시켜 상한이 92.3% 로 고정됐다. 크기 측정이
쌓인 상태에서 짧은 변 ±26~61mm 로 흔들리기 때문이다.

대신 크기 차이를 **거리에 곱하는(또는 더하는) 벌점**으로 쓴다. 크기가 많이 다르면
불리해지되 완전히 배제하지는 않으므로, 크기 측정이 틀려도 색·질감이 복구할 수 있다.

  size_d = sqrt( (Δlong/SL)² + w_short·(Δshort/SS)² )     ← 짧은 변 가중 w_short 탐색
  ① mul   : d = d_cos · (1 + α·size_d)
  ② add   : d = d_cos + λ·size_d
  ③ gauss : d = d_cos · exp(size_d²/2σ²)     (거리 기반 소프트맥스 게이트)

프로토콜은 8/14 와 동일(밝은 프레임 등록 → 어두운 프레임 배정, 처음 보는 3종 채점).
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
ident = lambda t: t[:-7] if t.endswith('_single') else t


class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()

byfile = defaultdict(list)
for r in objs:
    byfile[r['file']].append(r)
FEAT, MM, LAB, SINGLE, LUM = [], [], [], [], []
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
        FEAT.append(ft); MM.append(r['size_mm']); LAB.append(ident(r['type']))
        SINGLE.append(r['type'].endswith('_single')); LUM.append(L)
FEAT = np.stack(FEAT); MM = np.stack(MM).astype(float)
LAB = np.array(LAB); SINGLE = np.array(SINGLE); LUM = np.array(LUM)
types = sorted(set(LAB))
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = FEAT / scale * bw
Z5 = Z7[:, :5]
bright = np.zeros(len(Z7), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True
print(f"객체 {len(Z7)} / {len(types)}종")

SL, SS = 50.0, 50.0        # 크기 정규화 기준(mm) — CLUSTER_SIZE_SCALE_MM 과 동일 취지
HALF = (['blue_printed', 'navy_long', 'pacific_short'],
        ['navy_small', 'pacific_long', 'swanson'])


def cos_d(A, B):
    na = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    nb = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return 1 - na @ nb.T


def registry():
    reg = []
    for t in types:
        m = np.where((LAB == t) & bright)[0]
        c = Z7[m].mean(0)
        reg.append((t, m[int(np.argmin(np.linalg.norm(Z7[m] - c, axis=1)))]))
    return reg


REG = registry()
P_idx = np.array([i for _, i in REG]); P_lab = np.array([t for t, _ in REG])


def size_dist(i, w_short):
    dl = (MM[P_idx, 0] - MM[i, 0]) / SL
    ds = (MM[P_idx, 1] - MM[i, 1]) / SS
    return np.sqrt(dl ** 2 + w_short * ds ** 2)


def evaluate(subset, kind, w_short, p):
    te = np.where(~SINGLE & np.isin(LAB, subset) & ~bright)[0]
    ok = 0
    for i in te:
        dc = cos_d(Z5[[i]], Z5[P_idx])[0]
        sd = size_dist(i, w_short)
        if kind == 'mul':
            d = dc * (1 + p * sd)
        elif kind == 'add':
            d = dc + p * sd
        elif kind == 'gauss':
            d = dc * np.exp(sd ** 2 / (2 * p ** 2))
        elif kind == 'cos_only':
            d = dc
        elif kind == 'size_only':
            d = sd
        ok += (P_lab[int(np.argmin(d))] == LAB[i])
    return ok / max(len(te), 1)


def avg(kind, w_short, p):
    return float(np.mean([evaluate(h, kind, w_short, p) for h in HALF]))


print(f"\n기준선:  색·질감 코사인만 {avg('cos_only',1,0)*100:5.1f}%   "
      f"크기만 {avg('size_only',1,0)*100:5.1f}%   (참고: 현행 7차원 유클리드 91.8%)")

print(f"\n{'방식':<8}{'짧은변 가중':>10}{'파라미터':>10}{'정확도':>9}")
print("-" * 40)
best = None
for kind, ps in (('mul', [0.3, 0.6, 1.0, 1.5, 2.5, 4.0]),
                 ('add', [0.05, 0.1, 0.2, 0.35, 0.6, 1.0]),
                 ('gauss', [0.6, 1.0, 1.5, 2.0, 3.0])):
    for w in (0.0, 0.25, 0.5, 1.0):
        for p in ps:
            a = avg(kind, w, p)
            if best is None or a > best[0]:
                best = (a, kind, w, p)
    # 각 방식의 최고만 출력
    rows = [(avg(kind, w, p), w, p) for w in (0.0, 0.25, 0.5, 1.0) for p in ps]
    a, w, p = max(rows)
    print(f"{kind:<8}{w:>10.2f}{p:>10.2f}{a*100:>8.1f}%")

print(f"\n최고: {best[1]} (짧은변 가중 {best[2]}, 파라미터 {best[3]}) → {best[0]*100:.1f}%")

print(f"\n── 최고 방식의 짧은변 가중 민감도 ({best[1]}, p={best[3]}) ──")
for w in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    print(f"   w_short={w:<5} {avg(best[1], w, best[3])*100:>6.1f}%")
