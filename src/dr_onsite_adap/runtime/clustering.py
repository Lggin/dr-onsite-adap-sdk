"""외형 기반 클러스터링 — 검출된 객체를 '같은 클래스 안에서' 물리크기·색·질감으로 서브그룹핑.

분류(검출)가 먼저다. 여기서는 클래스별로 나눈 뒤 각 클래스 **안에서만** 비슷한 것끼리
묶는다 (다른 클래스는 절대 안 섞임). 학습 없음 — 결합 특징 벡터 + 고정 스케일 정규화 +
블록 가중 + union-find 거리 임계.

특징(7차원):
  * 색(3): Lab 중앙값 (밝기 L 은 CLUSTER_L_WEIGHT 로 다운웨이트 → 조명차 완화)
  * 질감(2): **조명 강건** — 로그-하이패스 에너지(=log 영역에서 저주파 조명 제거 후 인쇄 디테일
             std) + Canny 에지 밀도(글씨/획 양). 기존 std_gray 등은 밝기 그라데이션에 오염돼 폐기.
  * 크기(2): **물리 치수(long/short mm)** — 기하에서 받음(시점 불변, 상자 종류의 최강 단서).
             mm 를 못 받으면(2D 검출) 마스크 minAreaRect 픽셀로 폴백(장면 중앙값 대비 상대).
정규화 = 의미 기반 고정 스케일(구분 문턱). 크기 우선 가중이 기본.
출력: 객체별 group_id(전역 고유 int) + group_label('<클래스명>#<서브그룹>', 예 'Carton#0').
"""
import cv2
import numpy as np

from .. import config

# 색·질감 고정 스케일(=구분 문턱). 순서 [L*, a, b, logHP, edge].
_SCALE_COLOR = np.array([18.0, 14.0, 17.0])
_SCALE_TEX = np.array([0.04, 8.0])

# 그룹 시각화 색 팔레트(BGR) — group_id 로 인덱싱. 외형 서브그룹 구분용.
# ⚠️ 클래스 색(Carton=초록, Wrap=빨강, Slip=마젠타, Tote=하늘, Pallet=탄)과 겹치지 않게 고름
# → 그룹 한 개여도 클래스 색과 달라 "그룹화 ON" 이 눈에 보인다.
GROUP_PALETTE = [(0, 255, 255), (255, 255, 0), (0, 140, 255), (255, 0, 128),
                 (128, 255, 0), (255, 255, 255), (0, 215, 255), (255, 128, 0),
                 (203, 192, 255), (128, 128, 255)]


def group_color(group_id):
    """group_id → BGR 그룹 색(-1/None 이면 None)."""
    if group_id is None or group_id < 0:
        return None
    return GROUP_PALETTE[group_id % len(GROUP_PALETTE)]


def _uf_find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


def _frame_maps(color_bgr):
    """프레임당 1회: 그레이·Lab·로그하이패스 디테일·Canny 에지."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    sigma = max(6.0, 0.02 * color_bgr.shape[1])          # 조명(저주파) 추정 블러 폭 ~ 이미지폭 비례
    g = np.log1p(gray.astype(np.float32))                # 곱셈 조명 → 덧셈(homomorphic)
    detail = g - cv2.GaussianBlur(g, (0, 0), sigma)      # 저주파 조명 제거 → 인쇄/반사율 디테일
    edges = cv2.Canny(gray, 40, 120).astype(np.float32)  # 날카로운 경계(글씨)만; 완만한 그늘 무시
    return gray, lab, detail, edges


def _descriptor(mask, lab, detail, edges, size_pair):
    """객체 하나의 결합 특징(색3 + 질감2 + 크기2). size_pair=(long,short) mm 또는 픽셀."""
    er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))  # 가장자리 그림자 배제
    sel = er > 0
    if int(sel.sum()) < 20:
        sel = mask.astype(bool)
    col = np.median(lab[sel], axis=0).astype(float)
    col[0] *= config.CLUSTER_L_WEIGHT
    tex = [float(detail[sel].std()), float(edges[sel].mean())]
    return np.array([col[0], col[1], col[2], tex[0], tex[1],
                     float(size_pair[0]), float(size_pair[1])], dtype=float)


def cluster_by_appearance(masks, class_ids, color_bgr, sizes_mm=None, cfg=config):
    """masks(list[HxW bool|None]) + class_ids → (group_ids: list[int], group_labels: list[str]).

    sizes_mm: 객체별 (long_mm, short_mm) 물리치수 리스트(기하에서). None 또는 원소 None 이면
    그 객체는 2D 마스크 크기로 폴백. 클래스별로 나눠 각 클래스 안에서만 union-find 클러스터링.
    """
    n = len(masks)
    group_ids = [-1] * n
    group_labels = [""] * n
    if n == 0:
        return group_ids, group_labels

    gray, lab, detail, edges = _frame_maps(color_bgr)

    # 크기값(물리 mm 우선, 없으면 minAreaRect 픽셀) 수집
    have_phys = sizes_mm is not None
    size_vals = [None] * n
    feats_raw = [None] * n
    for i, m in enumerate(masks):
        if m is None or int(np.count_nonzero(m)) < cfg.CLUSTER_MIN_PX:
            continue
        sm = sizes_mm[i] if (have_phys and sizes_mm[i] is not None) else None
        if sm is None:
            ys, xs = np.where(m)
            (_c, (w, h), _r) = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            sm = (max(w, h), min(w, h))
            phys = False
        else:
            phys = True
        size_vals[i] = (sm, phys)
        feats_raw[i] = _descriptor(m.astype(bool), lab, detail, edges, sm)

    valid = [i for i in range(n) if feats_raw[i] is not None]
    if not valid:
        return group_ids, group_labels

    # 크기 스케일: 전부 물리면 고정 문턱(mm), 아니면 장면 중앙값 대비 상대(15%)
    all_phys = all(size_vals[i][1] for i in valid)
    if all_phys:
        size_scale = cfg.CLUSTER_SIZE_SCALE_MM
    else:
        med_long = float(np.median([size_vals[i][0][0] for i in valid]))
        size_scale = max(0.15 * med_long, 1.0)
    scale = np.concatenate([_SCALE_COLOR, _SCALE_TEX, [size_scale, size_scale]])
    bw = np.array([cfg.CLUSTER_W_COLOR] * 3 + [cfg.CLUSTER_W_TEX] * 2
                  + [cfg.CLUSTER_W_SIZE] * 2, dtype=float)
    tau = cfg.CLUSTER_TAU

    gid = 0
    for cid in sorted({int(class_ids[i]) for i in valid}):
        idxs = [i for i in valid if int(class_ids[i]) == cid]
        if len(idxs) == 1:
            local = [0]
        else:
            Z = np.stack([feats_raw[i] for i in idxs]) / scale * bw
            par = list(range(len(idxs)))
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if np.linalg.norm(Z[a] - Z[b]) < tau:
                        par[_uf_find(par, a)] = _uf_find(par, b)
            roots = {}
            local = [roots.setdefault(_uf_find(par, a), len(roots)) for a in range(len(idxs))]
        name = config.CLASSES.get(cid, (str(cid),))[0]
        for a, i in enumerate(idxs):
            group_ids[i] = gid + local[a]
            group_labels[i] = f"{name}#{local[a]}"
        gid += max(local) + 1

    return group_ids, group_labels
