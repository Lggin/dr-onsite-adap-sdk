"""질감 기술자 비교 — H2(FFT 방향 히스토그램)가 최선인가.

H2 가 이긴 이유는 '회전 불변'이라는 성질 때문인데, 그 성질을 갖는 기술자는 여럿이고
H2 는 그중 가장 단순한 축이다. 더 강력한 후보들을 같은 조건에서 잰다.

  T1 lbp_riu2   : LBP 회전불변 균일패턴 히스토그램 — 조명 단조변화에 완전 불변, 빠름
  T2 gabor      : Gabor 필터뱅크(4주파수×6방향) 에너지 → 방향축 마진화로 회전불변
  T3 glcm       : GLCM Haralick(대비·균질성·에너지·상관) 4방향 평균 → 회전 준불변
  T4 fmellin    : log-polar 변환 후 FFT 크기 — 회전+스케일 동시 불변
  T5 radon      : 방향별 투영(Radon)의 분산 프로파일 → 정렬로 회전불변
  T6 autocorr   : 자기상관 방사 프로파일 — 무늬 주기성
  H2 rotinv     : 기준(방향 히스토그램의 FFT 크기)

워프는 W2(종횡비 보존, 긴변 128px)로 고정. 프로토콜은 동일.
"""
import os, pickle
from collections import defaultdict
import numpy as np, cv2
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
PATCH, NB = 128, 12


def warp_w2(bgr, geom, intr):
    fx, fy, cx, cy = intr
    c = np.asarray(geom[0], float)
    ua = np.asarray(geom[1], float); va = np.asarray(geom[2], float)
    lmm, smm = float(geom[3]), float(geom[4])
    if lmm <= 0 or smm <= 0:
        return None
    hl, hs = lmm / 2000.0, smm / 2000.0
    pts = [c + a * hl * ua + b * hs * va for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    src = []
    for p in pts:
        if p[2] <= 1e-6:
            return None
        src.append([p[0] / p[2] * fx + cx, p[1] / p[2] * fy + cy])
    H = max(8, int(round(PATCH * max(smm / lmm, 0.05))))
    dst = np.array([[0, 0], [PATCH, 0], [PATCH, H], [0, H]], np.float32)
    M = cv2.getPerspectiveTransform(np.array(src, np.float32), dst)
    return cv2.warpPerspective(bgr, M, (PATCH, H))


def _gray(p):
    return cv2.equalizeHist(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY))


# ── H2 기준 ───────────────────────────────────────────────────────────────
def h2_rotinv(p):
    g = _gray(p).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) % np.pi)
    b = np.minimum((ang / np.pi * NB).astype(int), NB - 1)
    h = np.bincount(b.ravel(), weights=mag.ravel(), minlength=NB)
    h = h / max(h.sum(), 1e-9)
    f = np.abs(np.fft.rfft(h))[:NB // 2 + 1]
    return f / max(f[0], 1e-9)


# ── T1 LBP riu2 ───────────────────────────────────────────────────────────
def t1_lbp(p, P=8, R=1):
    """회전불변 균일패턴 LBP. 0/1 전이가 2회 이하인 패턴만 유효, 나머지는 하나로 묶음."""
    g = _gray(p).astype(np.float32)
    h, w = g.shape
    if min(h, w) < 2 * R + 2:
        return np.zeros(P + 2)
    codes = np.zeros((h, w), np.int32)
    for k in range(P):
        th = 2 * np.pi * k / P
        dy, dx = -R * np.sin(th), R * np.cos(th)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        sh = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        codes |= ((sh >= g).astype(np.int32) << k)
    # 회전불변: 순환 시프트 최소값 / 균일성: 전이 횟수
    lut = np.zeros(1 << P, np.int32)
    for c in range(1 << P):
        rots = [((c >> i) | (c << (P - i))) & ((1 << P) - 1) for i in range(P)]
        mn = min(rots)
        bits = [(c >> i) & 1 for i in range(P)]
        trans = sum(bits[i] != bits[(i + 1) % P] for i in range(P))
        lut[c] = bin(mn).count('1') if trans <= 2 else P + 1
    v = lut[codes.ravel()]
    hh = np.bincount(v, minlength=P + 2).astype(float)
    return hh / max(hh.sum(), 1e-9)


# ── T2 Gabor ──────────────────────────────────────────────────────────────
_GABOR = []
for lam in (4.0, 8.0, 16.0, 32.0):
    for th in np.arange(0, np.pi, np.pi / 6):
        _GABOR.append((lam, th, cv2.getGaborKernel((15, 15), 3.0, th, lam, 0.5, 0,
                                                   ktype=cv2.CV_32F)))


def t2_gabor(p):
    g = _gray(p).astype(np.float32) / 255.0
    E = np.zeros((4, 6))
    for i, (lam, th, k) in enumerate(_GABOR):
        r = cv2.filter2D(g, cv2.CV_32F, k)
        E[i // 6, i % 6] = float((r ** 2).mean())
    # 방향축을 정렬(최대 응답 방향을 0으로 순환 시프트) → 회전불변
    out = []
    for row in E:
        s = np.roll(row, -int(np.argmax(row)))
        out.append(s / max(s.sum(), 1e-9))
    return np.concatenate(out)


# ── T3 GLCM Haralick ──────────────────────────────────────────────────────
def t3_glcm(p, levels=16):
    g = (_gray(p) // (256 // levels)).astype(np.int32)
    feats = []
    for dy, dx in ((0, 1), (-1, 1), (-1, 0), (-1, -1)):
        a = g[max(0, -dy):g.shape[0] - max(0, dy), max(0, -dx):g.shape[1] - max(0, dx)]
        b = g[max(0, dy):g.shape[0] - max(0, -dy), max(0, dx):g.shape[1] - max(0, -dx)]
        if a.size == 0:
            feats.append([0, 0, 0, 0]); continue
        M = np.bincount((a * levels + b).ravel(), minlength=levels * levels)
        M = M.reshape(levels, levels).astype(float)
        M = (M + M.T); M /= max(M.sum(), 1e-9)
        i, j = np.mgrid[:levels, :levels]
        contrast = float((M * (i - j) ** 2).sum())
        homog = float((M / (1 + (i - j) ** 2)).sum())
        energy = float((M ** 2).sum())
        mi = float((M * i).sum()); si = np.sqrt(float((M * (i - mi) ** 2).sum())) + 1e-9
        corr = float((M * (i - mi) * (j - mi)).sum() / (si * si))
        feats.append([contrast, homog, energy, corr])
    return np.array(feats).mean(0)          # 4방향 평균 = 회전 준불변


# ── T4 Fourier-Mellin ─────────────────────────────────────────────────────
def t4_fmellin(p, nr=12, na=16):
    g = _gray(p).astype(np.float32)
    g = g - g.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g, s=(PATCH, PATCH))))
    F = np.log1p(F)
    cy, cx = PATCH // 2, PATCH // 2
    lp = cv2.warpPolar(F, (na, nr), (cx, cy), PATCH // 2, cv2.WARP_POLAR_LOG)
    # 각도축 FFT 크기 → 회전(=각도 시프트) 불변
    f = np.abs(np.fft.rfft(lp, axis=0))[:4]
    v = f.ravel()
    return v / max(np.linalg.norm(v), 1e-9)


# ── T5 Radon ──────────────────────────────────────────────────────────────
def t5_radon(p, nth=18):
    g = _gray(p).astype(np.float32)
    g = cv2.resize(g, (96, 96))
    g = g - g.mean()
    prof = []
    for th in np.linspace(0, 180, nth, endpoint=False):
        M = cv2.getRotationMatrix2D((48, 48), th, 1.0)
        r = cv2.warpAffine(g, M, (96, 96), borderMode=cv2.BORDER_REFLECT)
        prof.append(r.sum(0).std())          # 투영의 분산 = 그 방향 구조의 강도
    prof = np.array(prof)
    prof = prof / max(prof.sum(), 1e-9)
    f = np.abs(np.fft.rfft(prof))[:6]        # 회전불변화
    return f / max(f[0], 1e-9)


# ── T6 자기상관 ───────────────────────────────────────────────────────────
def t6_autocorr(p, nr=8):
    g = _gray(p).astype(np.float32)
    g = cv2.resize(g, (96, 96)); g = g - g.mean()
    F = np.fft.fft2(g)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    ac = ac / max(ac.max(), 1e-9)
    yy, xx = np.mgrid[:96, :96]
    rr = np.hypot(yy - 48, xx - 48)
    prof = np.array([ac[(rr >= k * 48 / nr) & (rr < (k + 1) * 48 / nr)].mean()
                     for k in range(nr)])
    return prof


TEX = {'H2_rotinv': h2_rotinv, 'T1_lbp': t1_lbp, 'T2_gabor': t2_gabor,
       'T3_glcm': t3_glcm, 'T4_fmellin': t4_fmellin, 'T5_radon': t5_radon,
       'T6_autocorr': t6_autocorr}

# ── 데이터 준비 ───────────────────────────────────────────────────────────
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
FT = {k: [] for k in TEX}
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
        p = warp_w2(img, gm, r['intr'])
        if p is None or min(p.shape[:2]) < 8:
            continue
        for k, fn in TEX.items():
            FT[k].append(fn(p))
        BASE.append(ft); LAB.append(ident(r['type']))
        SINGLE.append(r['type'].endswith('_single')); LUM.append(L)

BASE = np.stack(BASE); LAB = np.array(LAB)
SINGLE = np.array(SINGLE); LUM = np.array(LUM)
FT = {k: np.stack(v) for k, v in FT.items()}
types = sorted(set(LAB))
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = BASE / scale * bw
bright = np.zeros(len(Z7), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True
print(f"객체 {len(LAB)} / {len(types)}종")
for k, v in FT.items():
    print(f"  {k:<12}{v.shape[1]:>3}차원")

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


def evaluate(Zx, subset, dist):
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


base = np.mean([evaluate(Z7, h, 'euclid')[0] for h in HALF])
print(f"\n기준선(7차원, 질감 추가 없음) {base*100:.1f}%\n")
print(f"{'질감':<13}{'차원':>4}{'가중':>6}{'거리':<9}{'정확도':>8}{'실패':>6}{'문제쌍':>7}")
print("-" * 56)
best = (base, None, None, None)
for k, E0 in FT.items():
    E = E0 / max(E0.std(), 1e-9)
    rows = []
    for w in (0.5, 1.0, 2.0, 3.0):
        for dist in ('euclid', 'cosine'):
            Zx = np.hstack([Z7, E * w])
            accs, fails, prs = [], 0, 0
            for h in HALF:
                a, f, pr = evaluate(Zx, h, dist)
                accs.append(a); fails += f; prs += pr
            m = float(np.mean(accs))
            rows.append((m, w, dist, fails, prs))
            if m > best[0]:
                best = (m, k, w, dist)
    m, w, dist, f, pr = max(rows)
    print(f"{k:<13}{E0.shape[1]:>4}{w:>6.1f}{dist:<9}{m*100:>7.1f}%{f:>6}{pr:>7}")

print(f"\n최고: {best[1]} (가중 {best[2]}, {best[3]}) → {best[0]*100:.1f}%")
pickle.dump(dict(FT=FT, Z7=Z7, LAB=LAB, SINGLE=SINGLE, LUM=LUM),
            open(S + '/tex_zoo.pkl', 'wb'))
