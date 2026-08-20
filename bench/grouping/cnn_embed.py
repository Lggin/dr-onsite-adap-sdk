"""CNN 임베딩으로 상한 측정 — 손수 만든 특징(H2)의 한계는 어디인가.

H2(방향 히스토그램 FFT)가 96.4% 로 여섯 후보를 이겼지만, 그게 문제의 상한인지
손수 만든 특징의 한계인지는 알 수 없다. 사전학습 CNN 은 표현력이 압도적이므로
"이 데이터에서 도달 가능한 수준"을 알려준다.

  · DINOv2 (자기지도) — 로컬 가중치 사용
  · ResNet50 (ImageNet 지도학습)
회전 문제: CNN 임베딩은 회전 불변이 아니다. 박스가 뒤집혀 있으면 다른 벡터가 나온다.
  → 0/90/180/270도 회전 임베딩의 **평균**으로 회전 불변화(TTA)한다. 원본도 함께 잰다.

⚠️ CNN 이 크게 이기면 손수 특징의 한계가 드러난 것이고, 비슷하면 H2 로 충분하다는 근거다.
⚠️ 실전 적용에는 Thor(ARM) 추론 비용을 따로 따져야 한다.
"""
import os, pickle
from collections import defaultdict
import numpy as np, cv2, torch
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

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
    """가변 크기 패치를 정사각 입력으로 — 종횡비는 이미 특징에 있으므로 리사이즈."""
    im = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def load_models():
    out = {}
    # DINOv2 (로컬 가중치)
    try:
        import timm
        m = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                              pretrained=False, num_classes=0)
        sd = torch.load(os.path.expanduser('~/lee4/dino_vitb14_reg.pth'),
                        map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        missing = m.load_state_dict(sd, strict=False)
        m.eval().to(DEV)
        out['dinov2'] = (m, 224)
        print(f"  dinov2 로드 (missing {len(missing.missing_keys)}, "
              f"unexpected {len(missing.unexpected_keys)})")
    except Exception as e:
        print("  dinov2 실패:", type(e).__name__, e)
    # ResNet50 (ImageNet)
    try:
        import torchvision.models as tvm
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        m.eval().to(DEV)
        out['resnet50'] = (m, 224)
        print("  resnet50 로드")
    except Exception as e:
        print("  resnet50 실패:", type(e).__name__, e)
    return out


@torch.no_grad()
def embed(model, size, patches, rot_tta=True, bs=64):
    """rot_tta: 0/90/180/270도 임베딩 평균 → 회전 불변화."""
    outs = []
    for i in range(0, len(patches), bs):
        chunk = patches[i:i + bs]
        acc = None
        rots = [0, 1, 2, 3] if rot_tta else [0]
        for k in rots:
            xs = torch.stack([to_tensor(np.rot90(p, k).copy(), size) for p in chunk])
            v = model(xs.to(DEV)).float().cpu().numpy()
            acc = v if acc is None else acc + v
        outs.append(acc / len(rots))
    return np.concatenate(outs)


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
print(f"객체 {len(LAB)} / {len(types)}종  (device {DEV})")

models = load_models()
EMB = {}
for name, (m, size) in models.items():
    for tta in (False, True):
        key = f"{name}{'_rot' if tta else ''}"
        EMB[key] = embed(m, size, PATCHES, rot_tta=tta)
        print(f"  {key:<16}{EMB[key].shape}")
    del m
torch.cuda.empty_cache()

# H2 도 함께(비교 기준)
NB = 12
def h2(p):
    g = cv2.equalizeHist(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) % np.pi)
    b = np.minimum((ang / np.pi * NB).astype(int), NB - 1)
    h = np.bincount(b.ravel(), weights=mag.ravel(), minlength=NB)
    h = h / max(h.sum(), 1e-9)
    f = np.abs(np.fft.rfft(h))[:NB // 2 + 1]
    return f / max(f[0], 1e-9)
EMB['H2_rotinv'] = np.stack([h2(p) for p in PATCHES])

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
print(f"\n기준선(7차원) {base*100:.1f}%\n")
print(f"{'임베딩':<18}{'차원':>5}{'모드':<10}{'가중':>5}{'거리':<9}{'정확도':>8}{'실패':>6}{'문제쌍':>7}")
print("-" * 70)
for key, E0 in EMB.items():
    E = E0 / max(E0.std(), 1e-9)
    rows = []
    # 단독 사용(임베딩만) + 7차원 결합 둘 다
    for mode in ('embed_only', 'with_z7'):
        for w in (0.5, 1.0, 2.0):
            for dist in ('euclid', 'cosine'):
                Zx = (E * w) if mode == 'embed_only' else np.hstack([Z7, E * w])
                accs, fails, prs = [], 0, 0
                for h in HALF:
                    a, f, pr = evaluate(Zx, h, dist)
                    accs.append(a); fails += f; prs += pr
                rows.append((float(np.mean(accs)), mode, w, dist, fails, prs))
    m, mode, w, dist, f, pr = max(rows)
    print(f"{key:<18}{E0.shape[1]:>5}{mode:<10}{w:>5.1f}{dist:<9}{m*100:>7.1f}%{f:>6}{pr:>7}")
pickle.dump(EMB, open(S + '/cnn_emb.pkl', 'wb'))
