"""정답 라벨(6종)로 그룹화 성능 측정 — 실제 조명 변화 기준.

어제까지의 측정은 ①정답이 없어 의사-SKU(과분할 그룹)로 채점했고 ②조명이 HSV 합성이었다.
이제 둘 다 진짜다.

분할: 종류별로 프레임을 밝기(평균 L)로 반 나눠 **밝은 쪽 = 등록(정답 뱅크), 어두운 쪽 = 시험**.
      실제 조명 on/off 로 찍었으므로 이 분할이 곧 "등록해둔 뒤 조명이 바뀐 상황"이다.
특징: 실운영 경로 그대로 — 물리 mm 크기 + 워프패치 질감(둘 다 오늘 처음 진짜 값).
"""
import os, pickle
from collections import defaultdict
import numpy as np, cv2
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb'))
        if r['cid'] == 2 and r['size_mm'] and r['geom']]      # Carton + 기하 성공
print(f"대상 객체 {len(objs)}개")

def ident(t):                      # blue_printed_single → blue_printed
    return t[:-7] if t.endswith('_single') else t

byfile = defaultdict(list)
for r in objs:
    byfile[r['file']].append(r)

class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()

# ── 프레임별 특징 추출(실운영 경로: 물리 mm + 워프패치) ──────────────────────
FEAT, LAB, SRC, LUM = [], [], [], []
for f, rs in byfile.items():
    img = cv2.imread(f)
    masks = [r['mask'] for r in rs]
    sizes = [r['size_mm'] for r in rs]
    geoms = [(r['geom']['center_3d'], r['geom']['long_axis_3d'], r['geom']['short_axis_3d'],
              r['geom']['long_side_mm'], r['geom']['short_side_mm']) for r in rs]
    intr = rs[0]['intr']
    fs, phys = CL._build_features(masks, sizes, geoms, img, intr, cfg)
    L = float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean())
    for r, ft in zip(rs, fs):
        if ft is None:
            continue
        FEAT.append(ft); LAB.append(ident(r['type'])); SRC.append(f); LUM.append(L)
FEAT = np.stack(FEAT); LAB = np.array(LAB); SRC = np.array(SRC); LUM = np.array(LUM)
print(f"특징 {FEAT.shape} | 물리크기 사용 {sum(phys)}/{len(phys)} (마지막 프레임 기준)")
types = sorted(set(LAB))
print("종류별 객체 수:", {t: int((LAB == t).sum()) for t in types})

ss = config.CLUSTER_SIZE_SCALE_MM        # 물리 mm 를 쓰므로 고정 스케일
scale, bw = CL._scale_bw(ss, cfg)
Z = FEAT / scale * bw
one = np.ones(Z.shape[1])

# ── 등록/시험 분할: 종류별 밝기 상위 절반 = 등록 ────────────────────────────
is_reg = np.zeros(len(Z), bool)
for t in types:
    idx = np.where(LAB == t)[0]
    files = sorted(set(SRC[idx]), key=lambda f: -LUM[SRC == f][0])
    bright = set(files[:max(1, len(files) // 2)])
    for i in idx:
        if SRC[i] in bright:
            is_reg[i] = True
print(f"\n등록(밝은) {is_reg.sum()}개 / 시험(어두운) {(~is_reg).sum()}개")
print(f"평균 밝기  등록 {LUM[is_reg].mean():.1f}  시험 {LUM[~is_reg].mean():.1f}")

M = np.load(S + '/metric_M.npy')

def evaluate(Zx, tag):
    P, PL = [], []
    for t in types:                       # 종류당 medoid 1개 등록
        m = np.where(is_reg & (LAB == t))[0]
        c = Zx[m].mean(0)
        P.append(Zx[m][int(np.argmin(np.linalg.norm(Zx[m] - c, axis=1)))]); PL.append(t)
    P = np.stack(P); PL = np.array(PL)
    te = np.where(~is_reg)[0]
    d = np.linalg.norm(Zx[te][:, None] - P[None], axis=2)
    j = d.argmin(1); near = d.min(1)
    pred = np.where(near < cfg.CLUSTER_MATCH_TAU, PL[j], '기타')
    acc = (pred == LAB[te]).mean()
    other = (pred == '기타').mean()
    print(f"\n=== {tag} ===  정확도 {acc*100:.1f}%  '기타' 이탈 {other*100:.1f}%")
    print(f"{'종류':<18}{'n':>5}{'정확':>8}{'기타':>8}  주요 오배정")
    for t in types:
        sel = LAB[te] == t
        if not sel.any():
            continue
        p = pred[sel]
        wrong = [(x, int((p == x).sum())) for x in set(p) if x != t and x != '기타']
        wrong.sort(key=lambda z: -z[1])
        print(f"{t:<18}{sel.sum():>5}{(p==t).mean()*100:>7.0f}%{(p=='기타').mean()*100:>7.0f}%"
              f"  {', '.join(f'{a}:{b}' for a, b in wrong[:2])}")
    return acc

evaluate(Z, "고정 가중 (현재 운영)")
evaluate(Z @ M.T, "학습된 거리 (어제, 합성변형 기반)")
np.save(S + '/box_Z.npy', Z); np.save(S + '/box_isreg.npy', is_reg)
with open(S + '/box_meta.pkl', 'wb') as fp:
    pickle.dump(dict(LAB=LAB, SRC=SRC, LUM=LUM, types=types), fp)
