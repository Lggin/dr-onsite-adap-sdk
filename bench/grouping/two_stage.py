"""2단계 그룹화 — ① 물리 크기로 먼저 가르고 ② 같은 크기 안에서 색·질감 코사인.

착안: 두 신호의 성격이 완전히 다르다.
  · 크기(mm)  = 조명에 **완전 불변**, 물리 단위가 명확(± 몇 mm 로 임계를 말할 수 있음)
  · 색·질감   = 조명에 흔들림, 절대값보다 **패턴**이 중요 → 코사인이 유리
같은 거리 공간에 섞으면 서로의 장점을 깎는다(실측: 크기가 있으면 코사인 이점 소멸).

비교 대상
  1단계(현행)  : 7차원 한 번에 유클리드
  1단계+학습거리: 위 + 화이트닝 (8/14 최선, 94.3%)
  2단계        : 크기 게이트 → 색·질감 코사인
  2단계+학습거리: 크기 게이트 → 화이트닝된 색·질감

프로토콜은 8/14 와 동일: 밝은 프레임 등록 → 어두운 프레임 배정,
종류를 3+3 으로 갈라 **처음 보는 종류**에서 채점.
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
        FEAT.append(ft); MM.append(r['size_mm'])
        LAB.append(ident(r['type'])); SINGLE.append(r['type'].endswith('_single'))
        LUM.append(L)
FEAT = np.stack(FEAT); MM = np.stack(MM).astype(float)
LAB = np.array(LAB); SINGLE = np.array(SINGLE); LUM = np.array(LUM)
types = sorted(set(LAB))
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = FEAT / scale * bw
Z5 = Z7[:, :5]                              # 색3 + 질감2
print(f"객체 {len(Z7)} / {len(types)}종")

bright = np.zeros(len(Z7), bool)
for t in types:
    idx = np.where((LAB == t) & ~SINGLE)[0]
    bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True


def cos_d(A, B):
    na = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    nb = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return 1 - na @ nb.T


def euc_d(A, B):
    return np.linalg.norm(A[:, None] - B[None], axis=2)


def whiten(Z, y_, lam=0.10):
    D_ = np.concatenate([Z[y_ == c] - Z[y_ == c].mean(0) for c in np.unique(y_)])
    Sw = D_.T @ D_ / max(len(D_) - len(np.unique(y_)), 1)
    Sw += lam * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    w, V = np.linalg.eigh(Sw)
    M = V @ np.diag(w ** -0.5) @ V.T
    rng = np.random.RandomState(0)
    i = rng.choice(len(Z), min(300, len(Z)), False)
    d0 = euc_d(Z[i], Z[i]); Za = Z[i] @ M.T; d1 = euc_d(Za, Za)
    return M * (np.median(d0[d0 > 0]) / np.median(d1[d1 > 0]))


def registry(idx_of):
    """종류별 등록(밝은 쪽 medoid 1개) 인덱스."""
    reg = []
    for t in types:
        m = idx_of(t)
        c = Z7[m].mean(0)
        reg.append((t, m[int(np.argmin(np.linalg.norm(Z7[m] - c, axis=1)))]))
    return reg


def evaluate(subset, mode, size_tol=25.0, W=None):
    """mode: 'flat' | 'flat_w' | 'two' | 'two_w'"""
    reg = registry(lambda t: np.where((LAB == t) & bright)[0])
    te = np.where(~SINGLE & np.isin(LAB, subset) & ~bright)[0]
    P_idx = np.array([i for _, i in reg]); P_lab = np.array([t for t, _ in reg])
    ok = 0
    for i in te:
        if mode in ('flat', 'flat_w'):
            Zx = Z7 @ W.T if mode == 'flat_w' else Z7
            d = euc_d(Zx[[i]], Zx[P_idx])[0]
            pred = P_lab[int(np.argmin(d))]
        else:
            # ── 1단계: 물리 크기 게이트 ──
            dl = np.abs(MM[P_idx, 0] - MM[i, 0])
            ds = np.abs(MM[P_idx, 1] - MM[i, 1])
            cand = np.where((dl <= size_tol) & (ds <= size_tol))[0]
            if len(cand) == 0:                      # 크기로 맞는 게 없으면 가장 가까운 크기
                cand = np.array([int(np.argmin(dl + ds))])
            elif len(cand) == 1:
                pred = P_lab[cand[0]]
                ok += (pred == LAB[i]); continue
            # ── 2단계: 후보 안에서 색·질감 코사인 ──
            Zc = Z5 @ W.T if mode == 'two_w' else Z5
            d = cos_d(Zc[[i]], Zc[P_idx[cand]])[0]
            pred = P_lab[cand[int(np.argmin(d))]]
        ok += (pred == LAB[i])
    return ok / max(len(te), 1), len(te)


HALF = (['blue_printed', 'navy_long', 'pacific_short'],
        ['navy_small', 'pacific_long', 'swanson'])

print(f"\n{'방식':<34}{'정확도':>10}")
print("-" * 46)
res = {}
for mode, nm in (('flat', '1단계: 7차원 유클리드 (현행)'),
                 ('flat_w', '1단계: 7차원 + 학습된거리'),
                 ('two', '2단계: 크기게이트 → 색·질감 코사인'),
                 ('two_w', '2단계: 크기게이트 → 학습된 색·질감')):
    accs = []
    for tr, te_t in ((HALF[0], HALF[1]), (HALF[1], HALF[0])):
        W = None
        if mode.endswith('_w'):
            m = np.isin(LAB, tr)
            ytr = np.array([sorted(tr).index(t) for t in LAB[m]])
            W = whiten((Z7 if mode == 'flat_w' else Z5)[m], ytr)
        accs.append(evaluate(te_t, mode, W=W)[0])
    res[mode] = float(np.mean(accs))
    print(f"{nm:<34}{np.mean(accs)*100:>9.1f}%")

print(f"\n── 크기 허용오차(mm) 스윕: 2단계 ──")
for tol in (10, 15, 20, 25, 30, 40, 60):
    accs = []
    for te_t in HALF:
        accs.append(evaluate(te_t, 'two', size_tol=tol)[0])
    print(f"   ±{tol:>3}mm   {np.mean(accs)*100:>6.1f}%")
