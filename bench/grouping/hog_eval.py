"""방향성 특징 평가 — 기존 7차원에 붙였을 때 실제로 좋아지는가.

관심사는 두 가지.
  ① 전체 정확도가 오르는가 (현행 91.8%)
  ② **문제 쌍**(pacific_short↔swanson, blue_printed↔navy_small)이 풀리는가
     — 전체가 그대로여도 문제 쌍만 풀리면 다른 실패가 생긴 것이므로 따로 본다.

프로토콜은 동일: 밝은 프레임 medoid 등록 → 어두운 프레임 배정, 처음 보는 3종 채점.
"""
import os, pickle
import numpy as np
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
D = pickle.load(open(S + '/hog_feats.pkl', 'rb'))
BASE, MM, LAB, SINGLE, LUM = D['base'], D['mm'], D['lab'], D['single'], D['lum']
EXTRA = D['extra']
types = sorted(set(LAB))


class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = BASE / scale * bw
bright = np.zeros(len(Z7), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True

HALF = (['blue_printed', 'navy_long', 'pacific_short'],
        ['navy_small', 'pacific_long', 'swanson'])
PROBLEM = {('pacific_short', 'swanson'), ('swanson', 'pacific_short'),
           ('blue_printed', 'navy_small'), ('navy_small', 'blue_printed')}


def registry():
    reg = []
    for t in types:
        m = np.where((LAB == t) & bright)[0]
        c = Z7[m].mean(0)
        reg.append((t, m[int(np.argmin(np.linalg.norm(Z7[m] - c, axis=1)))]))
    return reg


REG = registry()
P_idx = np.array([i for _, i in REG]); P_lab = np.array([t for t, _ in REG])


def build(extra_key, w_extra):
    """7차원 + (방향성 특징 × 가중). 방향성은 히스토그램이라 L1 정규화돼 있으므로
    스케일을 맞춰 붙인다."""
    if extra_key is None:
        return Z7
    E = EXTRA[extra_key]
    E = E / max(E.std(), 1e-9)               # 전체 표준편차로 스케일 통일
    return np.hstack([Z7, E * w_extra])


def evaluate(Zx, subset, dist='euclid'):
    te = np.where(~SINGLE & np.isin(LAB, subset) & ~bright)[0]
    if dist == 'cosine':
        n = Zx / np.maximum(np.linalg.norm(Zx, axis=1, keepdims=True), 1e-9)
        d = 1 - n[te] @ n[P_idx].T
    else:
        d = np.linalg.norm(Zx[te][:, None] - Zx[P_idx][None], axis=2)
    pred = P_lab[d.argmin(1)]
    ok = pred == LAB[te]
    probs = sum(1 for a, b in zip(LAB[te][~ok], pred[~ok]) if (a, b) in PROBLEM)
    return float(ok.mean()), int((~ok).sum()), probs, len(te)


print(f"{'특징':<16}{'가중':>6}{'거리':<9}{'정확도':>9}{'실패':>6}{'문제쌍':>7}")
print("-" * 56)
base_acc = np.mean([evaluate(Z7, h)[0] for h in HALF])
bf = sum(evaluate(Z7, h)[1] for h in HALF)
bp = sum(evaluate(Z7, h)[2] for h in HALF)
print(f"{'(현행 7차원)':<16}{'-':>6}{'euclid':<9}{base_acc*100:>8.1f}%{bf:>6}{bp:>7}")

best = (base_acc, None, None, None)
for key in EXTRA:
    for w in (0.3, 0.6, 1.0, 2.0, 3.0):
        for dist in ('euclid', 'cosine'):
            Zx = build(key, w)
            accs, fails, probs = [], 0, 0
            for h in HALF:
                a, f, p, _ = evaluate(Zx, h, dist)
                accs.append(a); fails += f; probs += p
            m = float(np.mean(accs))
            if m > best[0]:
                best = (m, key, w, dist)
    # 각 특징의 최고 설정만 출력
    rows = []
    for w in (0.3, 0.6, 1.0, 2.0, 3.0):
        for dist in ('euclid', 'cosine'):
            Zx = build(key, w)
            accs, fails, probs = [], 0, 0
            for h in HALF:
                a, f, p, _ = evaluate(Zx, h, dist)
                accs.append(a); fails += f; probs += p
            rows.append((float(np.mean(accs)), w, dist, fails, probs))
    m, w, dist, f, p = max(rows)
    print(f"{key:<16}{w:>6.1f}{dist:<9}{m*100:>8.1f}%{f:>6}{p:>7}")

print(f"\n최고: {best[1]} (가중 {best[2]}, {best[3]}) → {best[0]*100:.1f}%  "
      f"(현행 {base_acc*100:.1f}%)")

# 문제 쌍만 따로 — 두 종류만 놓고 이진 판별
print(f"\n── 문제 쌍 이진 판별 (그 두 종류만 등록·시험) ──")
print(f"{'특징':<16}{'pacific_short vs swanson':>26}{'blue vs navy_small':>22}")
for key in [None] + list(EXTRA):
    Zx = build(key, 1.0)
    out = []
    for pair in (('pacific_short', 'swanson'), ('blue_printed', 'navy_small')):
        P2, L2 = [], []
        for t in pair:
            m = np.where((LAB == t) & bright)[0]
            c = Zx[m].mean(0)
            P2.append(Zx[m][int(np.argmin(np.linalg.norm(Zx[m] - c, axis=1)))]); L2.append(t)
        P2 = np.stack(P2); L2 = np.array(L2)
        te = np.where(~SINGLE & np.isin(LAB, pair) & ~bright)[0]
        d = np.linalg.norm(Zx[te][:, None] - P2[None], axis=2)
        out.append(float((L2[d.argmin(1)] == LAB[te]).mean()))
    nm = key or '(현행 7차원)'
    print(f"{nm:<16}{out[0]*100:>25.1f}%{out[1]*100:>21.1f}%")
