"""그룹화 시간 측정 — 프레임당 40 객체를 50~100ms 안에 처리할 수 있는가.

8/21 실험에서 정확도가 91.1% → 99%대로 올랐지만 그 대가는 **CNN 임베딩 추론**이다.
게다가 이긴 입력 규격(I3 native = 종횡비 보존)은 크기가 객체마다 달라 **배치로 묶을 수
없고**, 회전 4방향 TTA 까지 필요하다. 즉 40 객체면 ResNet50 forward 가 **160 회**다.
운영 목표(40 객체 / 50~100ms)에 들어가는지 실측한다.

측정 대상 — 정확도는 `~/공유_서브클래스_그룹화_실험정리.md` 의 값을 함께 적는다.

  H_handcraft   현행 배포(손수 7차원, assign_to_prototypes)          8종 91.1%
  R50_nat4      ResNet50 · 종횡비 보존 · 4회전 TTA · 개별 forward     8종 99%대  ← 목표 구성
  R50_nat1      같은 것에서 TTA 만 제거(회전 불변 상실)
  R50_sq4_batch ResNet50 · 정사각 224 강제 · 4회전 · 전부 배치        정확도 손해(실험 8)
  R18_nat4      ResNet18 로 교체(경량화 후보)

⚠️ 검출·윗면기하 시간은 **포함하지 않는다**. 그룹화는 이미 파이프라인이 뽑아놓은
마스크·윗면기하를 재사용하므로, 여기서 재는 것은 그 위에 얹히는 증분이다.
⚠️ 40 객체는 한 프레임 실측이 아니라 **한 프레임의 객체를 40 개까지 복제**해 만든다.
패치 내용은 시간에 거의 영향이 없고(같은 해상도 워프), 목표치가 최대 객체 수 기준이다.
"""
import os
import pickle
import statistics
import time

import cv2
import numpy as np
import torch

os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config                          # noqa: E402
from dr_onsite_adap.runtime import clustering as CL        # noqa: E402

S = os.environ.get("GROUPING_WORK", os.path.expanduser("~/grouping_work"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 128            # 워프 긴 변(현행과 동일)
CNN = 224              # CNN 입력 긴 변
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NS = [int(x) for x in os.environ.get("TIMING_NS", "1,10,20,40").split(",")]
REP = int(os.environ.get("TIMING_REP", "15"))
R50_W = os.path.expanduser(os.environ.get(
    "R50_WEIGHTS", "~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"))
R18_W = os.path.expanduser(os.environ.get(
    "R18_WEIGHTS", "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"))


# ─────────────────────────────── 전처리 ───────────────────────────────
def warp_w2(bgr, geom, intr):
    """윗면을 종횡비 보존 워프(긴 변 PATCH). 실험 6 에서 이긴 규격."""
    fx, fy, cx, cy = intr
    c = np.asarray(geom[0], float)
    ua, va = np.asarray(geom[1], float), np.asarray(geom[2], float)
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
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def prep_native(patch):
    h, w = patch.shape[:2]
    s = CNN / max(h, w)
    return _norm(cv2.resize(patch, (max(8, int(round(w * s))), max(8, int(round(h * s)))),
                            interpolation=cv2.INTER_AREA))


def prep_square(patch):
    return _norm(cv2.resize(patch, (CNN, CNN), interpolation=cv2.INTER_AREA))


def prep_pad(patch):
    """종횡비 보존 리사이즈 + 회색 패딩으로 정사각 캔버스(배치용, I2 계열)."""
    h, w = patch.shape[:2]
    s = CNN / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((CNN, CNN, 3), 128, np.uint8)
    y0, x0 = (CNN - nh) // 2, (CNN - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return _norm(canvas)


# ─────────────────────────────── 임베딩 ───────────────────────────────
def embed_native(model, patches, rots, half):
    """종횡비 보존 → 크기가 달라 배치 불가. 객체×회전 개별 forward."""
    out = []
    with torch.no_grad():
        for p in patches:
            acc = 0
            for k in range(rots):
                x = prep_native(np.rot90(p, k).copy()).unsqueeze(0).to(DEV)
                acc = acc + model(x.half() if half else x).float()
            out.append((acc / rots).squeeze(0))
    return torch.stack(out)


def embed_square_batched(model, patches, rots, half, bs=64):
    """정사각 강제(찌그러뜨림) → 전부 한 배치로 묶을 수 있다."""
    xs = [prep_square(np.rot90(p, k).copy()) for p in patches for k in range(rots)]
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            x = torch.stack(xs[i:i + bs]).to(DEV)
            out.append(model(x.half() if half else x).float())
    E = torch.cat(out).reshape(len(patches), rots, -1).mean(1)
    return E


def embed_pad_batched(model, patches, rots, half, bs=64):
    """종횡비 보존(패딩) + 배치 — 정확도 손해 없이(찌그러뜨림 없이) 배치화되는가."""
    xs = [prep_pad(np.rot90(p, k).copy()) for p in patches for k in range(rots)]
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            x = torch.stack(xs[i:i + bs]).to(DEV)
            out.append(model(x.half() if half else x).float())
    E = torch.cat(out).reshape(len(patches), rots, -1).mean(1)
    return E


def load_backbone(kind):
    import torchvision.models as tvm
    if kind == "r50":
        m, w = tvm.resnet50(weights=None), R50_W
    else:
        m, w = tvm.resnet18(weights=None), R18_W
    sd = torch.load(w, map_location="cpu")
    m.load_state_dict(sd)
    m.fc = torch.nn.Identity()
    return m.eval().to(DEV)


# ─────────────────────────────── 측정 도구 ───────────────────────────────
def bench(fn, rep=REP, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(rep):
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts), max(ts)


# ─────────────────────────────── 데이터 ───────────────────────────────
objs = pickle.load(open(f"{S}/box_objs.pkl", "rb"))
usable = [o for o in objs if o.get("geom") and o.get("size_mm") and o.get("intr")]
by_file = {}
for o in usable:
    by_file.setdefault(o["file"], []).append(o)
best = max(by_file.values(), key=len)
FRAME = cv2.imread(best[0]["file"])
INTR = best[0]["intr"]
print(f"장치 {DEV} / torch {torch.__version__}"
      + (f" / {torch.cuda.get_device_name(0)}" if DEV == "cuda" else ""))
print(f"프레임 {os.path.basename(best[0]['file'])} {FRAME.shape[1]}x{FRAME.shape[0]} "
      f"— 윗면기하 확보 객체 {len(best)}개 (전체 {len(usable)}/{len(objs)})")


def geom_of(o):
    g = o["geom"]
    return (g["center_3d"], g["long_axis_3d"], g["short_axis_3d"],
            g["long_side_mm"], g["short_side_mm"])


def make_scene(n):
    """한 프레임의 객체를 순환 복제해 n 개 장면을 만든다."""
    pool = best
    sel = [pool[i % len(pool)] for i in range(n)]
    masks = [o["mask"] for o in sel]
    cids = [o["cid"] for o in sel]
    sizes = [o["size_mm"] for o in sel]
    geoms = [geom_of(o) for o in sel]
    return masks, cids, sizes, geoms


# 정답 뱅크(현행 경로용) — 8 종 등록 상태를 재현
_m, _c, _s, _g = make_scene(min(8, len(best)))
BANK = {}
for i in range(len(_m)):
    f = CL.prototype_feature(_m[i], FRAME, _s[i], geom=_g[i], intrinsics=INTR)
    if f is not None:
        CL.add_prototype(BANK, _c[i], f"proto{i}", f, _s[i][0], _s[i][1])
print(f"정답 뱅크 {CL.bank_count(BANK)}개 등록")

R50 = load_backbone("r50")
R18 = load_backbone("r18")
R50H = load_backbone("r50").half()
PROJ = torch.randn(2048, 16, device=DEV)       # PCA 16 투영(비용만 재현)


# ─────────────────────────────── 조건 ───────────────────────────────
def run_handcraft(sc):
    masks, cids, sizes, geoms = sc
    return CL.assign_to_prototypes(masks, cids, FRAME, sizes, BANK,
                                   geoms=geoms, intrinsics=INTR)


def _cnn_common(sc, model, embed, rots, half, proj=True):
    masks, cids, sizes, geoms = sc
    patches = [w for w in (warp_w2(FRAME, g, INTR) for g in geoms) if w is not None]
    E = embed(model, patches, rots, half)
    if proj and E.shape[1] == 2048:
        E = E @ PROJ
    hand = CL.assign_to_prototypes(masks, cids, FRAME, sizes, BANK,
                                   geoms=geoms, intrinsics=INTR)
    Z = E.cpu().numpy()
    _ = np.linalg.norm(Z[:, None, :] - Z[None, :len(BANK) or 1, :], axis=2)
    return hand


CONDS = [
    ("H_handcraft", "현행 손수 7차원", lambda sc: run_handcraft(sc)),
    ("R50_nat4", "R50 종횡비보존 4회전(목표)",
     lambda sc: _cnn_common(sc, R50, embed_native, 4, False)),
    ("R50_nat4_fp16", "R50 종횡비보존 4회전 fp16",
     lambda sc: _cnn_common(sc, R50H, embed_native, 4, True)),
    ("R50_nat1", "R50 종횡비보존 TTA없음",
     lambda sc: _cnn_common(sc, R50, embed_native, 1, False)),
    ("R50_sq4_batch", "R50 정사각 4회전 배치",
     lambda sc: _cnn_common(sc, R50, embed_square_batched, 4, False)),
    ("R50_pad4_batch", "R50 종횡비보존(패딩) 4회전 배치",
     lambda sc: _cnn_common(sc, R50, embed_pad_batched, 4, False)),
    ("R18_nat4", "R18 종횡비보존 4회전",
     lambda sc: _cnn_common(sc, R18, embed_native, 4, False)),
]

print(f"\n{'조건':<16}{'설명':<26}" + "".join(f"{f'N={n}':>12}" for n in NS))
rows = {}
for tag, desc, fn in CONDS:
    cells = []
    for n in NS:
        sc = make_scene(n)
        med, lo, hi = bench(lambda: fn(sc))
        cells.append(med)
        rows[(tag, n)] = (med, lo, hi)
    print(f"{tag:<16}{desc:<26}" + "".join(f"{c:>11.1f}ms" for c in cells), flush=True)

# ─────────────────── N=40 단계별 분해 (목표 구성) ───────────────────
NMAX = max(NS)
sc = make_scene(NMAX)
masks, cids, sizes, geoms = sc
patches = [w for w in (warp_w2(FRAME, g, INTR) for g in geoms) if w is not None]
print(f"\nN={NMAX} 단계별 분해 — 목표 구성(R50 종횡비보존 4회전)")
stages = [
    ("워프(CPU)", lambda: [warp_w2(FRAME, g, INTR) for g in geoms]),
    ("CNN 전처리(CPU)", lambda: [prep_native(np.rot90(p, k).copy())
                                 for p in patches for k in range(4)]),
    ("CNN forward(GPU)", lambda: embed_native(R50, patches, 4, False)),
    ("손수 특징+배정(CPU)", lambda: run_handcraft(sc)),
]
for nm, f in stages:
    med, lo, hi = bench(f, rep=max(5, REP // 2))
    print(f"  {nm:<22}{med:>9.1f}ms  (최소 {lo:.1f} / 최대 {hi:.1f})")

print(f"\nN={NMAX} 단계별 분해 — 배치화(패딩, 종횡비 보존) 4회전")
stages_pad = [
    ("워프(CPU)", lambda: [warp_w2(FRAME, g, INTR) for g in geoms]),
    ("CNN 전처리(CPU, 배치용)", lambda: [prep_pad(np.rot90(p, k).copy())
                                        for p in patches for k in range(4)]),
    ("CNN forward(GPU, 배치)", lambda: embed_pad_batched(R50, patches, 4, False)),
    ("손수 특징+배정(CPU)", lambda: run_handcraft(sc)),
]
for nm, f in stages_pad:
    med, lo, hi = bench(f, rep=max(5, REP // 2))
    print(f"  {nm:<22}{med:>9.1f}ms  (최소 {lo:.1f} / 최대 {hi:.1f})")

# ─────────────────── 이식 예측용 단위 측정 ───────────────────
print("\n이식 예측용 단위 측정 (Orin 환산 기준)")
x1 = torch.randn(1, 3, CNN, CNN, device=DEV)
xb = torch.randn(64, 3, CNN, CNN, device=DEV)
with torch.no_grad():
    med, lo, hi = bench(lambda: R50(x1), rep=30)
    print(f"  R50 bs1 224² fp32      {med:>8.2f}ms  (최소 {lo:.2f})")
    med, lo, hi = bench(lambda: R50H(x1.half()), rep=30)
    print(f"  R50 bs1 224² fp16      {med:>8.2f}ms  (최소 {lo:.2f})")
    med, lo, hi = bench(lambda: R50(xb), rep=10)
    print(f"  R50 bs64 224² fp32     {med:>8.2f}ms  → 장당 {med/64:.2f}ms")
    med, lo, hi = bench(lambda: R18(x1), rep=30)
    print(f"  R18 bs1 224² fp32      {med:>8.2f}ms  (최소 {lo:.2f})")
a = torch.randn(2048, 2048, device=DEV)
med, _, _ = bench(lambda: a @ a, rep=20)
print(f"  matmul 2048³ fp32      {med:>8.2f}ms  → {2*2048**3/med/1e9:.1f} GFLOP/s")
big = np.random.rand(720, 1280, 3).astype(np.float32)
med, _, _ = bench(lambda: cv2.GaussianBlur(big, (0, 0), 4.0), rep=20)
print(f"  CPU 기준 블러 720p     {med:>8.2f}ms")
