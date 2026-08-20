"""워프 목적지 규격 3종 비교 — 종횡비 파괴가 실제로 얼마나 해로운가.

현행 `_warp_patch` 는 윗면 3D 사각형을 **무조건 128×128 정사각형**으로 워프한다.
그래서 356×192mm 박스는 긴 변이 1.85배 압축되고, 190×180mm 는 거의 그대로다.
같은 무늬라도 종류마다 다른 모양이 되고, 짧은 변 측정이 흔들리면(±26~61mm)
같은 박스도 프레임마다 다르게 왜곡된다.

  W1 square  : 128×128 정사각형              ← 현행
  W2 aspect  : 긴 변 128, 짧은 변은 비율대로   (예: 128×69)
  W3 canvas  : W2 를 128×128 캔버스 가운데 배치 + 유효영역 마스크
  W4 physical: 물리 해상도 고정(1mm=0.4px)    (예: 142×77)

각 규격에서 방향성 특징(H1~H4)을 뽑아 프로토타입 배정 정확도로 채점한다.
⚠️ 패치 크기가 가변이므로 피처는 크기에 불변인 형태(정규화 히스토그램)만 쓴다.
"""
import os, pickle
from collections import defaultdict
import numpy as np, cv2
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
NB = 12
PATCH = 128
MM_PER_PX = 0.4


def warp(bgr, geom, intr, kind):
    """geom=(center,long_ax,short_ax,long_mm,short_mm). (패치, 유효마스크) 반환."""
    fx, fy, cx, cy = intr
    c = np.asarray(geom[0], float)
    ua = np.asarray(geom[1], float); va = np.asarray(geom[2], float)
    lmm, smm = float(geom[3]), float(geom[4])
    hl, hs = lmm / 2000.0, smm / 2000.0
    if hl <= 0 or hs <= 0:
        return None, None
    pts = [c + a * hl * ua + b * hs * va for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    src = []
    for p in pts:
        if p[2] <= 1e-6:
            return None, None
        src.append([p[0] / p[2] * fx + cx, p[1] / p[2] * fy + cy])
    src = np.array(src, np.float32)
    ar = max(smm / max(lmm, 1e-6), 0.05)          # 짧은변/긴변
    if kind == 'W1_square':
        W, H = PATCH, PATCH
    elif kind in ('W2_aspect', 'W3_canvas'):
        W, H = PATCH, max(8, int(round(PATCH * ar)))
    elif kind == 'W4_physical':
        W = max(16, int(round(lmm * MM_PER_PX)))
        H = max(8, int(round(smm * MM_PER_PX)))
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(bgr, M, (W, H))
    if kind != 'W3_canvas':
        return patch, np.ones(patch.shape[:2], bool)
    canvas = np.zeros((PATCH, PATCH, 3), np.uint8)
    mask = np.zeros((PATCH, PATCH), bool)
    y0 = (PATCH - H) // 2
    canvas[y0:y0 + H] = patch
    mask[y0:y0 + H] = True
    return canvas, mask


def _grad(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy), (np.arctan2(gy, gx) % np.pi)


def h_orient(patch, m):
    g = cv2.equalizeHist(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
    mag, ang = _grad(g.astype(np.float32))
    mag = mag * m
    b = np.minimum((ang / np.pi * NB).astype(int), NB - 1)
    h = np.bincount(b.ravel(), weights=mag.ravel(), minlength=NB)
    return h / max(h.sum(), 1e-9)


def h_rotinv(patch, m):
    h = h_orient(patch, m)
    f = np.abs(np.fft.rfft(h))[:NB // 2 + 1]
    return f / max(f[0], 1e-9)


def h_radial(patch, m, nr=8):
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) * m
    g = g - g[m].mean() if m.any() else g
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    H, W = P.shape
    yy, xx = np.mgrid[:H, :W]
    rr = np.hypot(yy - H // 2, xx - W // 2)
    rmax = min(H, W) // 2
    prof = np.array([P[(rr >= k * rmax / nr) & (rr < (k + 1) * rmax / nr)].mean()
                     if ((rr >= k * rmax / nr) & (rr < (k + 1) * rmax / nr)).any() else 0.0
                     for k in range(nr)])
    return prof / max(prof.sum(), 1e-9)


HFN = {'H1_orient': h_orient, 'H2_rotinv': h_rotinv, 'H4_radial': h_radial}
KINDS = ['W1_square', 'W2_aspect', 'W3_canvas', 'W4_physical']

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

BASE, LAB, SINGLE, LUM = [], [], [], []
FEATS = {(k, h): [] for k in KINDS for h in HFN}
for f, rs in byfile.items():
    img = cv2.imread(f)
    geoms = [(r['geom']['center_3d'], r['geom']['long_axis_3d'], r['geom']['short_axis_3d'],
              r['geom']['long_side_mm'], r['geom']['short_side_mm']) for r in rs]
    fs, _ = CL._build_features([r['mask'] for r in rs], [r['size_mm'] for r in rs],
                               geoms, img, rs[0]['intr'], cfg)
    L = float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean())
    for r, ft, gm in zip(rs, fs, geoms):
        if ft is None:
            continue
        ok = True
        tmp = {}
        for k in KINDS:
            p, m = warp(img, gm, r['intr'], k)
            if p is None:
                ok = False
                break
            for hname, fn in HFN.items():
                tmp[(k, hname)] = fn(p, m)
        if not ok:
            continue
        for key, v in tmp.items():
            FEATS[key].append(v)
        BASE.append(ft); LAB.append(ident(r['type']))
        SINGLE.append(r['type'].endswith('_single')); LUM.append(L)

BASE = np.stack(BASE); LAB = np.array(LAB)
SINGLE = np.array(SINGLE); LUM = np.array(LUM)
FEATS = {k: np.stack(v) for k, v in FEATS.items()}
types = sorted(set(LAB))
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = BASE / scale * bw
bright = np.zeros(len(Z7), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True
print(f"객체 {len(LAB)} / {len(types)}종\n")

REG = []
for t in types:
    m = np.where((LAB == t) & bright)[0]
    c = Z7[m].mean(0)
    REG.append((t, m[int(np.argmin(np.linalg.norm(Z7[m] - c, axis=1)))]))
P_idx = np.array([i for _, i in REG]); P_lab = np.array([t for t, _ in REG])
HALF = (['blue_printed', 'navy_long', 'pacific_short'],
        ['navy_small', 'pacific_long', 'swanson'])
PROBLEM = {('pacific_short', 'swanson'), ('swanson', 'pacific_short'),
           ('blue_printed', 'navy_small'), ('navy_small', 'blue_printed')}


def evaluate(Zx, subset, dist='euclid'):
    te = np.where(~SINGLE & np.isin(LAB, subset) & ~bright)[0]
    if dist == 'cosine':
        n = Zx / np.maximum(np.linalg.norm(Zx, axis=1, keepdims=True), 1e-9)
        d = 1 - n[te] @ n[P_idx].T
    else:
        d = np.linalg.norm(Zx[te][:, None] - Zx[P_idx][None], axis=2)
    pred = P_lab[d.argmin(1)]
    ok = pred == LAB[te]
    pr = sum(1 for a, b in zip(LAB[te][~ok], pred[~ok]) if (a, b) in PROBLEM)
    return float(ok.mean()), int((~ok).sum()), pr


base = np.mean([evaluate(Z7, h)[0] for h in HALF])
bf = sum(evaluate(Z7, h)[1] for h in HALF)
print(f"기준선 (7차원, 방향성 없음): {base*100:.1f}%  실패 {bf}\n")
print(f"{'워프':<13}{'방향성':<12}{'가중':>5}{'거리':<9}{'정확도':>8}{'실패':>6}{'문제쌍':>7}")
print("-" * 62)
best = (base, None, None, None, None)
for k in KINDS:
    rows = []
    for hname in HFN:
        E = FEATS[(k, hname)]
        E = E / max(E.std(), 1e-9)
        for w in (0.5, 1.0, 2.0):
            for dist in ('euclid', 'cosine'):
                Zx = np.hstack([Z7, E * w])
                accs, fails, prs = [], 0, 0
                for h in HALF:
                    a, f, p = evaluate(Zx, h, dist)
                    accs.append(a); fails += f; prs += p
                m = float(np.mean(accs))
                rows.append((m, hname, w, dist, fails, prs))
                if m > best[0]:
                    best = (m, k, hname, w, dist)
    m, hname, w, dist, f, p = max(rows)
    print(f"{k:<13}{hname:<12}{w:>5.1f}{dist:<9}{m*100:>7.1f}%{f:>6}{p:>7}")

print(f"\n최고: {best[1]} + {best[2]} (가중 {best[3]}, {best[4]}) → {best[0]*100:.1f}%")
pickle.dump(dict(feats=FEATS, lab=LAB, single=SINGLE, lum=LUM, base=BASE),
            open(S + '/warp_compare.pkl', 'wb'))
