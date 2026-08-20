"""축별 판별력 분석 — 각 축이 '품목'을 보는가 '촬영 조건'을 보는가.

세 가지 분산을 비교한다.
  · between  : 종류 간 분산      → 품목 신호(클수록 좋다)
  · within   : 종류 내 분산      → 잡음(작을수록 좋다)
  · frame    : 같은 종류·같은 프레임 안의 분산  → 순수 개체차·측정오차
  · session  : 같은 종류이지만 프레임이 다를 때의 분산  → **촬영 조건 영향**

  F = between / within        높을수록 품목 판별에 유용
  R = session / frame         높을수록 그 축이 촬영 조건(조명·각도)에 흔들린다

사람 눈에 이유가 안 보이는 과분할은 R 이 큰 축이 원인일 것이다.
"""
import os, pickle
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


def prep(patch):
    h, w = patch.shape[:2]
    s = 224 / max(h, w)
    im = cv2.resize(patch, (max(16, int(round(w * s))), max(16, int(round(h * s)))),
                    interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


@torch.no_grad()
def embed(model, patches):
    out = []
    for p in patches:
        acc = None
        for k in (0, 1, 2, 3):
            v = model(prep(np.rot90(p, k).copy()).unsqueeze(0).to(DEV)).float().cpu().numpy()
            acc = v if acc is None else acc + v
        out.append(acc / 4)
    return np.concatenate(out)


class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()
objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb'))
        if r['cid'] == 2 and r['size_mm'] and r['geom']]
byfile = defaultdict(list)
for r in objs:
    byfile[r['file']].append(r)
BASE, PATCHES, TRUTH, FRAME, CROPS = [], [], [], [], []
for f, rs in byfile.items():
    img = cv2.imread(f)
    geoms = [(r['geom']['center_3d'], r['geom']['long_axis_3d'], r['geom']['short_axis_3d'],
              r['geom']['long_side_mm'], r['geom']['short_side_mm']) for r in rs]
    fs, _ = CL._build_features([r['mask'] for r in rs], [r['size_mm'] for r in rs],
                               geoms, img, rs[0]['intr'], cfg)
    for r, ft, gm in zip(rs, fs, geoms):
        if ft is None:
            continue
        p = warp_w2(img, gm, r['intr'])
        if p is None or min(p.shape[:2]) < 8:
            continue
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        CROPS.append(cv2.resize(img[max(0, y1):y2, max(0, x1):x2], (96, 96)))
        BASE.append(ft); PATCHES.append(p); FRAME.append(f)
        TRUTH.append(r['type'][:-7] if r['type'].endswith('_single') else r['type'])
BASE = np.stack(BASE); TRUTH = np.array(TRUTH); FRAME = np.array(FRAME)

import torchvision.models as tvm
model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Identity(); model.eval().to(DEV)
E = embed(model, PATCHES)
pca = PCA(16, random_state=0).fit(E)
Ep = pca.transform(E); Ep = Ep / max(Ep.std(), 1e-9)
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z = np.hstack([BASE / scale * bw, Ep * 0.35])
AX = ['L', 'a', 'b', 'tex_hp', 'tex_edge', 'size_long', 'size_short'] + \
     [f'cnn{i:02d}' for i in range(16)]
Zs = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
print(f"객체 {len(Z)} / 종류 {len(set(TRUTH))} / 프레임 {len(set(FRAME))}\n")

rows = []
for k, nm in enumerate(AX):
    v = Zs[:, k]
    # between / within (종류 기준)
    gm_ = [v[TRUTH == t] for t in sorted(set(TRUTH))]
    within = np.mean([g.var() for g in gm_ if len(g) > 1])
    between = np.var([g.mean() for g in gm_])
    # frame(같은 종류·같은 프레임 내) vs session(같은 종류·프레임 간)
    fr, se = [], []
    for t in sorted(set(TRUTH)):
        m = TRUTH == t
        fmeans = []
        for f in set(FRAME[m]):
            g = v[m & (FRAME == f)]
            if len(g) > 1:
                fr.append(g.var())
            if len(g) > 0:
                fmeans.append(g.mean())
        if len(fmeans) > 1:
            se.append(np.var(fmeans))
    frame_v = np.mean(fr) if fr else np.nan
    sess_v = np.mean(se) if se else np.nan
    rows.append((nm, between / max(within, 1e-9), sess_v / max(frame_v, 1e-9),
                 between, within, frame_v, sess_v))

print(f"{'축':<11}{'F(품목판별)':>12}{'R(촬영영향)':>12}{'판정':>18}")
print("-" * 56)
for nm, F, R, b, w, fv, sv in sorted(rows, key=lambda r: -r[1]):
    if F >= 1.0 and R < 1.5:
        j = '품목 신호 ✔'
    elif F >= 1.0:
        j = '품목+촬영 혼재'
    elif R >= 1.5:
        j = '촬영 조건 ✘'
    else:
        j = '약함'
    print(f"{nm:<11}{F:>12.2f}{R:>12.2f}{j:>18}")

noisy = [nm for nm, F, R, *_ in rows if F < 1.0 and R >= 1.5]
print(f"\n촬영 조건에 흔들리고 품목 판별력 낮은 축: {noisy}")
pickle.dump(dict(Z=Z, Zs=Zs, AX=AX, TRUTH=TRUTH, FRAME=FRAME, BASE=BASE,
                 Ep=Ep, E=E, pca=pca, CROPS=CROPS, rows=rows),
            open(S + '/axis_power.pkl', 'wb'))
