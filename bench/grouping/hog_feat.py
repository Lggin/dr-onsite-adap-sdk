"""방향성 히스토그램 질감 특징 — pacific_short ↔ swanson 을 가를 수 있는가.

현재 질감 2차원(하이패스 std, 에지 밀도)은 무늬의 **양**만 재고 **생김새**를 못 본다.
그래서 크기·색이 비슷한 크래프트 박스끼리(Pacific 로고 vs Swanson 로고) 구분이 안 된다.

워프 패치(128px 정면화)에서 방향성 특징을 뽑는다:
  H1 orient   : 그래디언트 방향 히스토그램 (가중=크기). 획이 어느 방향으로 흐르는가
  H2 rotinv   : 위 히스토그램의 FFT 크기 — **회전 불변**(박스가 돌아가 있어도 같음)
  H3 blockhog : 패치를 3×3 블록으로 나눈 미니 HOG — 무늬의 **공간 배치**까지
                (예: swanson 은 상하단에 검은 띠 → 위/아래 블록의 에지가 강함)
  H4 radial   : FFT 파워의 방사 프로파일 — 무늬의 굵기·주기성

각각을 기존 특징에 붙여 프로토타입 배정 정확도를 잰다.
"""
import os, pickle
from collections import defaultdict
import numpy as np, cv2
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
NB = 12                      # 방향 bin 수


def _grad(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi)        # 0~π (획의 방향은 180도 주기)
    return mag, ang


def h_orient(patch):
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)                    # 조명 정규화
    mag, ang = _grad(g.astype(np.float32))
    b = np.minimum((ang / np.pi * NB).astype(int), NB - 1)
    h = np.bincount(b.ravel(), weights=mag.ravel(), minlength=NB)
    return h / max(h.sum(), 1e-9)


def h_rotinv(patch):
    """방향 히스토그램의 FFT 크기 — 회전(=순환 시프트)에 불변."""
    h = h_orient(patch)
    f = np.abs(np.fft.rfft(h))[:NB // 2 + 1]
    return f / max(f[0], 1e-9)                 # DC 로 정규화


def h_blockhog(patch, nb=6):
    """3×3 블록 × nb 방향 = 54차원. 무늬의 공간 배치를 담는다."""
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    mag, ang = _grad(g.astype(np.float32))
    H, W = g.shape
    out = []
    for by in range(3):
        for bx in range(3):
            ys, ye = by * H // 3, (by + 1) * H // 3
            xs, xe = bx * W // 3, (bx + 1) * W // 3
            m, a = mag[ys:ye, xs:xe], ang[ys:ye, xs:xe]
            b = np.minimum((a / np.pi * nb).astype(int), nb - 1)
            h = np.bincount(b.ravel(), weights=m.ravel(), minlength=nb)
            out.append(h / max(h.sum(), 1e-9))
    return np.concatenate(out)


def h_radial(patch, nr=8):
    """FFT 파워의 방사 프로파일 — 무늬 굵기/주기성. 회전에 준불변."""
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) * np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None]
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    H, W = P.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    rr = np.hypot(yy - cy, xx - cx)
    rmax = min(cy, cx)
    prof = []
    for k in range(nr):
        m = (rr >= k * rmax / nr) & (rr < (k + 1) * rmax / nr)
        prof.append(P[m].mean() if m.any() else 0.0)
    prof = np.array(prof)
    return prof / max(prof.sum(), 1e-9)


EXTRA = {'H1_orient': h_orient, 'H2_rotinv': h_rotinv,
         'H3_blockhog': h_blockhog, 'H4_radial': h_radial}

if __name__ == '__main__':
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
    BASE, LAB, SINGLE, LUM, MM = [], [], [], [], []
    EX = {k: [] for k in EXTRA}
    for f, rs in byfile.items():
        img = cv2.imread(f)
        geoms = [(r['geom']['center_3d'], r['geom']['long_axis_3d'],
                  r['geom']['short_axis_3d'], r['geom']['long_side_mm'],
                  r['geom']['short_side_mm']) for r in rs]
        fs, _ = CL._build_features([r['mask'] for r in rs], [r['size_mm'] for r in rs],
                                   geoms, img, rs[0]['intr'], cfg)
        L = float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean())
        for r, ft, gm in zip(rs, fs, geoms):
            if ft is None:
                continue
            p = CL._warp_patch(img, gm, r['intr'])
            if p is None:
                continue
            BASE.append(ft); MM.append(r['size_mm']); LAB.append(ident(r['type']))
            SINGLE.append(r['type'].endswith('_single')); LUM.append(L)
            for k, fn in EXTRA.items():
                EX[k].append(fn(p))
    D = dict(base=np.stack(BASE), mm=np.stack(MM).astype(float),
             lab=np.array(LAB), single=np.array(SINGLE), lum=np.array(LUM),
             extra={k: np.stack(v) for k, v in EX.items()})
    print(f"객체 {len(D['lab'])}")
    for k, v in D['extra'].items():
        print(f"  {k:<12}{v.shape}")
    pickle.dump(D, open(S + '/hog_feats.pkl', 'wb'))
    print("저장:", S + '/hog_feats.pkl')
