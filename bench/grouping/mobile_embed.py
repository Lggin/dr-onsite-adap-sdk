"""경량 백본 비교 — 정확도 vs 속도 곡선을 그린다.

ResNet50+회전TTA 가 99.0% 를 냈지만 2048차원 × 4회 forward 라 Thor(ARM) 실시간에 부담이다.
경량 백본이 어디까지 따라오는지, 차원 축소로 얼마나 줄일 수 있는지 잰다.

측정: 정확도(프로토타입 배정) + 객체당 임베딩 시간 + 차원
⚠️ 속도는 4090 기준이다. Thor(ARM)는 훨씬 느리므로 **상대 비교용**으로만 본다.
"""
import os, pickle, time
from collections import defaultdict
import numpy as np, cv2, torch
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL
from sklearn.decomposition import PCA

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


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


def to_tensor(patch, size):
    im = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


def build_models():
    import torchvision.models as tvm
    out = {}
    specs = [
        ('mobilenet_v2', lambda: tvm.mobilenet_v2(
            weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1), 'classifier'),
        ('mobilenet_v3_small', lambda: tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1), 'classifier'),
        ('shufflenet_v2_x1_0', lambda: tvm.shufflenet_v2_x1_0(
            weights=tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1), 'fc'),
        ('resnet18', lambda: tvm.resnet18(
            weights=tvm.ResNet18_Weights.IMAGENET1K_V1), 'fc'),
        ('resnet50', lambda: tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V2), 'fc'),
    ]
    for name, fn, head in specs:
        try:
            m = fn()
            setattr(m, head, torch.nn.Identity())
            m.eval().to(DEV)
            out[name] = (m, 224)
            print(f"  {name} 로드")
        except Exception as e:
            print(f"  {name} 실패: {type(e).__name__}")
    return out


@torch.no_grad()
def embed(model, size, patches, rot=True, bs=64):
    outs = []
    rots = [0, 1, 2, 3] if rot else [0]
    for i in range(0, len(patches), bs):
        chunk = patches[i:i + bs]
        acc = None
        for k in rots:
            xs = torch.stack([to_tensor(np.rot90(p, k).copy(), size) for p in chunk])
            v = model(xs.to(DEV)).float().cpu().numpy()
            acc = v if acc is None else acc + v
        outs.append(acc / len(rots))
    return np.concatenate(outs)


@torch.no_grad()
def bench(model, size, patches, rot=True, n=64):
    xs = torch.stack([to_tensor(p, size) for p in patches[:n]]).to(DEV)
    r = 4 if rot else 1
    for _ in range(2):
        for _ in range(r):
            model(xs)
    torch.cuda.synchronize() if DEV == 'cuda' else None
    t0 = time.time()
    for _ in range(3):
        for _ in range(r):
            model(xs)
    torch.cuda.synchronize() if DEV == 'cuda' else None
    return (time.time() - t0) / (3 * n) * 1000        # 객체당 ms


# ── 데이터 ────────────────────────────────────────────────────────────────
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
BASE, LAB, SINGLE, LUM, PATCHES = [], [], [], [], []
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
        BASE.append(ft); PATCHES.append(p); LAB.append(ident(r['type']))
        SINGLE.append(r['type'].endswith('_single')); LUM.append(L)
BASE = np.stack(BASE); LAB = np.array(LAB)
SINGLE = np.array(SINGLE); LUM = np.array(LUM)
types = sorted(set(LAB))
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = BASE / scale * bw
bright = np.zeros(len(Z7), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True
REG = []
for t in types:
    m = np.where((LAB == t) & bright)[0]
    c = Z7[m].mean(0)
    REG.append((t, m[int(np.argmin(np.linalg.norm(Z7[m] - c, axis=1)))]))
P_idx = np.array([i for _, i in REG]); P_lab = np.array([t for t, _ in REG])
# 8종 → 4+4 분할. 각 조에 '크기같고색다름' / '색같고크기다름' 쌍이 섞이도록 배치.
HALF = (['blue_printed', 'navy_long', 'pacific_short', 'plain_small'],
        ['navy_small', 'pacific_long', 'swanson', 'red'])
PROBLEM = {('pacific_short', 'swanson'), ('swanson', 'pacific_short'),
           ('blue_printed', 'navy_small'), ('navy_small', 'blue_printed'),
           ('plain_small', 'navy_small'), ('navy_small', 'plain_small'),
           ('plain_small', 'blue_printed'), ('blue_printed', 'plain_small'),
           ('red', 'pacific_short'), ('pacific_short', 'red'),
           ('red', 'swanson'), ('swanson', 'red'),
           ('red', 'pacific_long'), ('pacific_long', 'red')}


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


def best_of(E0):
    E = E0 / max(E0.std(), 1e-9)
    rows = []
    for w in (0.5, 1.0, 2.0):
        for dist in ('euclid', 'cosine'):
            Zx = np.hstack([Z7, E * w])
            accs, fails, prs = [], 0, 0
            for h in HALF:
                a, f, pr = evaluate(Zx, h, dist)
                accs.append(a); fails += f; prs += pr
            rows.append((float(np.mean(accs)), w, dist, fails, prs))
    return max(rows)


print(f"객체 {len(LAB)} / {len(types)}종  (device {DEV})")
# 기준선: 방향성/임베딩 없는 현행 7차원
_b = [evaluate(Z7, h, 'euclid') for h in HALF]
print(f"기준선(7차원 유클리드): {np.mean([x[0] for x in _b])*100:.1f}%  "
      f"실패 {sum(x[1] for x in _b)}  문제쌍 {sum(x[2] for x in _b)}")
# H2 방향 히스토그램(손수 특징 최고)도 비교군으로
NB = 12
def _h2(p):
    g = cv2.equalizeHist(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) % np.pi)
    b = np.minimum((ang / np.pi * NB).astype(int), NB - 1)
    h = np.bincount(b.ravel(), weights=mag.ravel(), minlength=NB)
    h = h / max(h.sum(), 1e-9)
    f = np.abs(np.fft.rfft(h))[:NB // 2 + 1]
    return f / max(f[0], 1e-9)
_H2 = np.stack([_h2(p) for p in PATCHES])
_a, _w, _d, _f, _pr = best_of(_H2)
print(f"H2 방향히스토그램(손수 최고): {_a*100:.1f}%  실패 {_f}  문제쌍 {_pr}  ({_d})")
models = build_models()
print(f"\n{'백본':<20}{'차원':>6}{'객체당ms':>9}{'정확도':>8}{'실패':>6}{'문제쌍':>7}{'거리':>8}")
print("-" * 66)
store = {}
for name, (m, size) in models.items():
    E = embed(m, size, PATCHES, rot=True)
    ms = bench(m, size, PATCHES, rot=True)
    acc, w, dist, f, pr = best_of(E)
    store[name] = E
    print(f"{name:<20}{E.shape[1]:>6}{ms:>9.2f}{acc*100:>7.1f}%{f:>6}{pr:>7}{dist:>8}")
    del m
torch.cuda.empty_cache()

print(f"\n── PCA 차원 축소 (회전TTA 유지) ──")
print(f"{'백본':<20}{'차원':>6}{'정확도':>8}{'실패':>6}")
for name, E0 in store.items():
    for k in (16, 32, 64):
        if E0.shape[1] <= k:
            continue
        Ep = PCA(k, random_state=0).fit_transform(E0)
        acc, w, dist, f, pr = best_of(Ep)
        print(f"{name:<20}{k:>6}{acc*100:>7.1f}%{f:>6}")
pickle.dump(store, open(S + '/mobile_emb.pkl', 'wb'))
