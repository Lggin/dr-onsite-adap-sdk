"""박스 유사도 그리드 서치 ① — 피처 추출.

box_types 475개(정답 6종 + depth)에 대해 **세 가지 피처 세트**를 뽑는다.
정답이 있으므로 "어느 조합이 좋은가"를 점수로 고를 수 있다.

  F1 basic   : 지금 EDA 방식 그대로 (bbox 크롭, Lab 중앙값 3 + 질감 2)      ← 대조군
  F2 mask    : 같은 5차원이되 **마스크 안 픽셀만** 사용                      ← 배경 오염 제거
  F3 rich    : 마스크 + 크기정규화 질감 + 색분포 + 형태 + 물리크기            ← 확장

F1↔F2 차이가 곧 "배경 오염이 얼마나 망치고 있었나"다.
"""
import os, pickle
import numpy as np, cv2

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
PATCH = 128          # 질감을 재기 전 정규화할 크기(픽셀 크기 의존성 제거)


def _tex(gray):
    """로그 하이패스 std + Canny 에지밀도 + LBP 엔트로피."""
    gl = np.log1p(gray.astype(np.float32))
    hp = gl - cv2.GaussianBlur(gl, (0, 0), 3.0)
    t1 = float(hp.std())
    t2 = float(cv2.Canny(gray, 40, 120).mean())
    # LBP: 각 픽셀을 이웃 8개와 비교한 이진 패턴 → 히스토그램 엔트로피(무늬 복잡도)
    g = gray.astype(np.int16)
    code = np.zeros(g.shape, np.uint8)
    for k, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                  (1, 1), (1, 0), (1, -1), (0, -1)]):
        sh = np.roll(np.roll(g, dy, 0), dx, 1)
        code |= ((sh >= g).astype(np.uint8) << k)
    h = np.bincount(code.ravel(), minlength=256).astype(np.float64)
    h /= max(h.sum(), 1)
    t3 = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    return t1, t2, t3


def f1_basic(bgr, mask, size_mm):
    """대조군 — 지금 EDA 방식(bbox 크롭 전체, 배경 포함)."""
    if bgr.size == 0 or min(bgr.shape[:2]) < 8:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = [float(np.median(lab[..., i])) for i in range(3)]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gl = np.log1p(g.astype(np.float32))
    t1 = float((gl - cv2.GaussianBlur(gl, (0, 0), 3.0)).std())
    t2 = float(cv2.Canny(g, 40, 120).mean())
    return [L, a, b, t1, t2]


def f2_mask(bgr, mask, size_mm):
    """같은 5차원이되 마스크 안 픽셀만."""
    if bgr.size == 0 or mask.sum() < 40:
        return None
    m = mask.astype(bool)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = [float(np.median(lab[..., i][m])) for i in range(3)]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 마스크 밖을 마스크 내부 평균으로 채워 경계가 가짜 에지를 만들지 않게 한다
    filled = np.where(m, g, int(g[m].mean())).astype(np.uint8)
    gl = np.log1p(filled.astype(np.float32))
    hp = (gl - cv2.GaussianBlur(gl, (0, 0), 3.0))[m]
    t1 = float(hp.std())
    e = cv2.Canny(filled, 40, 120)
    t2 = float(e[m].mean())
    return [L, a, b, t1, t2]


def f3_rich(bgr, mask, size_mm):
    """마스크 + 크기정규화 질감 + 색분포 + 형태 + 물리크기."""
    if bgr.size == 0 or mask.sum() < 40:
        return None
    m = mask.astype(bool)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[m]                                   # (N,3)
    med = np.median(px, 0)
    # 색 분포: 사분위 폭(IQR) — 단색인가 무늬가 있는가
    iqr = np.percentile(px, 75, 0) - np.percentile(px, 25, 0)
    chroma = float(np.hypot(med[1] - 128, med[2] - 128))   # 채도(무채색이면 0)
    # 질감: 마스크 영역을 128px 로 정규화한 뒤 측정(픽셀 크기 의존 제거)
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sub = g[y0:y1, x0:x1]
    sm = m[y0:y1, x0:x1]
    filled = np.where(sm, sub, int(sub[sm].mean()) if sm.any() else 0).astype(np.uint8)
    norm = cv2.resize(filled, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    t1, t2, t3 = _tex(norm)
    # 형태: 종횡비, 충진도(마스크/볼록껍질 — 가려짐·찌그러짐 지표)
    cnts, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity, aspect = 1.0, 1.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ar = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = float(ar / hull) if hull > 0 else 1.0
        (_, (w, h), _) = cv2.minAreaRect(c)
        aspect = float(max(w, h) / max(min(w, h), 1))
    # 물리 크기(있으면). 없으면 픽셀 대각선으로 대체 — 스케일이 다르므로 별도 축.
    if size_mm:
        s_long, s_short = float(size_mm[0]), float(size_mm[1])
    else:
        s_long, s_short = float(max(y1 - y0, x1 - x0)), float(min(y1 - y0, x1 - x0))
    return [float(med[0]), float(med[1]), float(med[2]),
            float(iqr[0]), float(iqr[1]), float(iqr[2]), chroma,
            t1, t2, t3, solidity, aspect, s_long, s_short]


FEATSETS = {'F1_basic': f1_basic, 'F2_mask': f2_mask, 'F3_rich': f3_rich}
NAMES = {
    'F1_basic': ['L', 'a', 'b', 'tex_hp', 'tex_edge'],
    'F2_mask': ['L', 'a', 'b', 'tex_hp', 'tex_edge'],
    'F3_rich': ['L', 'a', 'b', 'L_iqr', 'a_iqr', 'b_iqr', 'chroma',
                'tex_hp', 'tex_edge', 'tex_lbp', 'solidity', 'aspect',
                'size_long', 'size_short'],
}

if __name__ == '__main__':
    objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb')) if r['cid'] == 2]
    print(f"대상 {len(objs)}개 (box_types Carton)")
    out = {k: [] for k in FEATSETS}
    labels, srcs = [], []
    cache = {}
    for r in objs:
        if r['file'] not in cache:
            cache.clear()
            cache[r['file']] = cv2.imread(r['file'])
        bgr_full = cache[r['file']]
        if bgr_full is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        crop = bgr_full[max(0, y1):y2, max(0, x1):x2]
        mk = r['mask'][max(0, y1):y2, max(0, x1):x2].astype(np.uint8)
        ok = True
        vals = {}
        for k, fn in FEATSETS.items():
            v = fn(crop, mk, r['size_mm'])
            if v is None:
                ok = False
                break
            vals[k] = v
        if not ok:
            continue
        for k, v in vals.items():
            out[k].append(v)
        t = r['type'][:-7] if r['type'].endswith('_single') else r['type']
        labels.append(t)
        srcs.append(r['file'])
    res = {k: np.array(v) for k, v in out.items()}
    for k, v in res.items():
        print(f"  {k:<10} {v.shape}")
    pickle.dump(dict(feats=res, labels=np.array(labels), srcs=np.array(srcs),
                     names=NAMES), open(S + '/grid_feats.pkl', 'wb'))
    print("저장:", S + '/grid_feats.pkl')
