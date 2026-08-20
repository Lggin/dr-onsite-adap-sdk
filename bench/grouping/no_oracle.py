"""정답 없이 재현 가능한 설정 검증.

앞선 실험에서 'F(품목 판별력) ≥ 2.0 인 축만 남기면' 군집이 30→19개로 줄고 ARI 0.658→0.852
로 올랐다. 그런데 **F 는 촬영 종류(정답)로 계산**한 것이라 실전에서는 쓸 수 없다.

대신 정답이 필요 없는 두 조작으로 같은 효과가 나는지 본다.
  ① PCA 차원 축소 — 분산이 작은 하위 축은 정의상 정보가 적다(비지도)
  ② size_short 제외/축소 — 쌓인 박스는 짧은 변이 이웃에 가려 부정확하다는
     물리적 근거가 있다(8/14 실측 ±26~61mm)

두 지표로 잰다.
  · 비지도 군집: 군집 수·순도·ARI
  · 프로토타입 배정(실제 운용 방식): 밝은 프레임 등록 → 어두운 프레임 배정,
    종류를 4+4 로 갈라 **처음 보는 4종**에서 채점
"""
import os, pickle
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import cv2
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
D = pickle.load(open(S + '/axis_power.pkl', 'rb'))
BASE, E, TRUTH, FRAME = D['BASE'], D['E'], D['TRUTH'], D['FRAME']
AXF = {nm: f for nm, f, *_ in D['rows']}


class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z7 = BASE / scale * bw
types = sorted(set(TRUTH))
y = np.array([types.index(t) for t in TRUTH])

# 프레임 밝기 → 등록/시험 분할 (운용 프로토콜 재현)
LUM = np.zeros(len(TRUTH))
for f in set(FRAME):
    img = cv2.imread(f)
    LUM[FRAME == f] = float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean())
SINGLE = np.array(['_single' in f or False for f in FRAME])   # 낱개 폴더 여부
SINGLE = np.array(['/' + t + '_single/' in f for t, f in zip(TRUTH, FRAME)])
bright = np.zeros(len(TRUTH), bool)
for t in types:
    idx = np.where((TRUTH == t) & ~SINGLE)[0]
    if len(idx):
        bright[idx[LUM[idx] >= np.median(LUM[idx])]] = True
HALF = (['blue_printed', 'navy_long', 'pacific_short', 'plain_small'],
        ['navy_small', 'pacific_long', 'swanson', 'red'])
print(f"객체 {len(TRUTH)} / {len(types)}종 / 등록후보 {bright.sum()} / 낱개 {SINGLE.sum()}")


def cluster_scores(Zx, thr):
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                  linkage='complete').fit_predict(Zx)
    pur = sum(Counter(TRUTH[lab == g]).most_common(1)[0][1] for g in set(lab)) / len(TRUTH)
    return len(set(lab)), pur, adjusted_rand_score(y, lab)


def proto_acc(Zx):
    """운용 방식: 종류별 밝은 medoid 1개 등록 → 어두운 것 배정, 처음 보는 4종 채점."""
    P, PL = [], []
    for t in types:
        m = np.where((TRUTH == t) & bright)[0]
        if not len(m):
            continue
        c = Zx[m].mean(0)
        P.append(Zx[m][int(np.argmin(np.linalg.norm(Zx[m] - c, axis=1)))]); PL.append(t)
    P = np.stack(P); PL = np.array(PL)
    accs = []
    for h in HALF:
        te = np.where(~SINGLE & np.isin(TRUTH, h) & ~bright)[0]
        if not len(te):
            continue
        d = np.linalg.norm(Zx[te][:, None] - P[None], axis=2)
        accs.append((PL[d.argmin(1)] == TRUTH[te]).mean())
    return float(np.mean(accs))


print(f"\n{'설정':<34}{'축':>4}{'군집@2.0':>10}{'순도':>7}{'ARI':>7}{'배정정확도':>10}")
print("-" * 74)
results = {}
for npca in (3, 5, 8, 16):
    Ep = PCA(npca, random_state=0).fit_transform(E)
    Ep = Ep / max(Ep.std(), 1e-9)
    for drop_short in (False, True):
        base = np.delete(Z7, 6, axis=1) if drop_short else Z7
        Zx = np.hstack([base, Ep * 0.35])
        k, pur, ari = cluster_scores(Zx, 2.0)
        pa = proto_acc(Zx)
        nm = f"PCA{npca}" + (" + size_short 제외" if drop_short else "")
        results[nm] = (Zx.shape[1], k, pur, ari, pa)
        print(f"{nm:<34}{Zx.shape[1]:>4}{k:>10}{pur*100:>6.0f}%{ari:>7.3f}{pa*100:>9.1f}%")

# 비교 기준 두 개
Ep16 = PCA(16, random_state=0).fit_transform(E); Ep16 /= max(Ep16.std(), 1e-9)
Zfull = np.hstack([Z7, Ep16 * 0.35])
AXn = ['L', 'a', 'b', 'tex_hp', 'tex_edge', 'size_long', 'size_short'] + \
      [f'cnn{i:02d}' for i in range(16)]
keep = [i for i, nm in enumerate(AXn) if AXF[nm] >= 2.0]
Zorc = Zfull[:, keep]
print("-" * 74)
for nm, Zx in (('(기준) 전체 23축', Zfull), ('(참고·정답사용) F≥2.0 11축', Zorc)):
    k, pur, ari = cluster_scores(Zx, 2.0)
    print(f"{nm:<34}{Zx.shape[1]:>4}{k:>10}{pur*100:>6.0f}%{ari:>7.3f}{proto_acc(Zx)*100:>9.1f}%")
