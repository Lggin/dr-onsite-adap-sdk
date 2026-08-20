"""CNN 입력 규격 비교 — 정사각 강제 vs 종횡비 보존.

실험 6에서 워프 목적지를 종횡비 보존으로 바꿔 개선했는데(95.4%→96.4%), CNN 에 넣을 때
다시 224×224 정사각으로 리사이즈하고 있었다. 종횡비 보존이 무효화되는 셈이다.
실측: 같은 패치를 종횡비 유지(128×69) vs 정사각(224×224)으로 넣으면 출력 코사인 유사도
0.590 — 상당히 다른 벡터가 나온다.

ResNet 은 GAP 덕분에 입력 크기가 달라도 항상 2048차원을 뱉으므로 여러 규격이 가능하다.

  I1 square     : 224×224 정사각 강제                        ← 현행
  I2 pad        : 종횡비 유지해 긴 변 224 → 224×224 캔버스에 중앙 배치(회색 패딩)
  I3 native     : 종횡비 유지 그대로 (긴 변 224). 크기가 달라 배치 못 묶음 → 개별 forward
  I4 short224   : 짧은 변을 224 로 (긴 변이 길어짐)
  I5 square160  : 160×160 정사각 (작은 입력이 나은지)

⚠️ I3 는 객체마다 크기가 달라 배치 처리가 불가하므로 느리다. 속도도 함께 잰다.
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


def _norm(im):
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


def prep(patch, kind):
    h, w = patch.shape[:2]
    if kind == 'I1_square':
        return _norm(cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA))
    if kind == 'I5_square160':
        return _norm(cv2.resize(patch, (160, 160), interpolation=cv2.INTER_AREA))
    if kind == 'I2_pad':
        s = 224 / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        r = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((224, 224, 3), 128, np.uint8)      # 회색 패딩(정규화 후 ≈0)
        y0, x0 = (224 - nh) // 2, (224 - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = r
        return _norm(canvas)
    if kind == 'I3_native':
        s = 224 / max(h, w)
        nw, nh = max(16, int(round(w * s))), max(16, int(round(h * s)))
        return _norm(cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA))
    if kind == 'I4_short224':
        s = 224 / max(min(h, w), 1)
        nw, nh = max(16, int(round(w * s))), max(16, int(round(h * s)))
        return _norm(cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA))
    raise ValueError(kind)


KINDS = ['I1_square', 'I2_pad', 'I3_native', 'I4_short224', 'I5_square160']


@torch.no_grad()
def embed(model, patches, kind, rot=True, bs=64):
    """고정 크기면 배치, 가변이면 개별 forward."""
    rots = [0, 1, 2, 3] if rot else [0]
    fixed = kind in ('I1_square', 'I2_pad', 'I5_square160')
    out = []
    if fixed:
        for i in range(0, len(patches), bs):
            chunk = patches[i:i + bs]
            acc = None
            for k in rots:
                xs = torch.stack([prep(np.rot90(p, k).copy(), kind) for p in chunk])
                v = model(xs.to(DEV)).float().cpu().numpy()
                acc = v if acc is None else acc + v
            out.append(acc / len(rots))
        return np.concatenate(out)
    for p in patches:                       # 가변 크기 → 하나씩
        acc = None
        for k in rots:
            x = prep(np.rot90(p, k).copy(), kind).unsqueeze(0)
            v = model(x.to(DEV)).float().cpu().numpy()
            acc = v if acc is None else acc + v
        out.append(acc / len(rots))
    return np.concatenate(out)


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
HALF = (['blue_printed', 'navy_long', 'pacific_short', 'plain_small'],
        ['navy_small', 'pacific_long', 'swanson', 'red'])

ar = np.array([p.shape[1] / max(p.shape[0], 1) for p in PATCHES])
print(f"객체 {len(LAB)} / {len(types)}종")
print(f"패치 종횡비: 중앙 {np.median(ar):.2f}  범위 {ar.min():.2f}~{ar.max():.2f}")


def ev(Zx, dist='euclid'):
    accs, fails, n = [], 0, 0
    for h in HALF:
        te = np.where(~SINGLE & np.isin(LAB, h) & ~bright)[0]
        if dist == 'cosine':
            nn = Zx / np.maximum(np.linalg.norm(Zx, axis=1, keepdims=True), 1e-9)
            d = 1 - nn[te] @ nn[P_idx].T
        else:
            d = np.linalg.norm(Zx[te][:, None] - Zx[P_idx][None], axis=2)
        pred = P_lab[d.argmin(1)]
        ok = pred == LAB[te]
        accs.append(ok.mean()); fails += int((~ok).sum()); n += len(te)
    return float(np.mean(accs)), fails, n


import torchvision.models as tvm
model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Identity(); model.eval().to(DEV)

print(f"\n{'입력규격':<16}{'추출초':>7}{'단독PCA16':>11}{'+7차원':>10}{'실패':>6}")
print("-" * 54)
best = None
for kind in KINDS:
    t0 = time.time()
    E = embed(model, PATCHES, kind, rot=True)
    el = time.time() - t0
    Ep = PCA(16, random_state=0).fit_transform(E)
    Ep = Ep / max(Ep.std(), 1e-9)
    a_alone = ev(Ep)[0]
    rows = []
    for w in (0.2, 0.35, 0.5, 1.0):
        for dist in ('euclid', 'cosine'):
            rows.append((*ev(np.hstack([Z7, Ep * w]), dist), w, dist))
    a_comb, f_comb, n, w, dist = max(rows)
    print(f"{kind:<16}{el:>7.1f}{a_alone*100:>10.1f}%{a_comb*100:>9.1f}%{f_comb:>6}"
          f"   (w={w}, {dist})")
    if best is None or a_comb > best[0]:
        best = (a_comb, kind, w, dist, f_comb)
print(f"\n최고: {best[1]} (w={best[2]}, {best[3]}) → {best[0]*100:.1f}%  실패 {best[4]}")
