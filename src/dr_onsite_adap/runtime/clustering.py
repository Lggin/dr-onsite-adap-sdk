"""외형 기반 클러스터링 — 검출된 객체를 '같은 클래스 안에서' 색·질감·크기로 서브그룹핑.

분류(검출)가 먼저다. 여기서는 클래스별로 나눈 뒤 각 클래스 **안에서만** 비슷한 것끼리
묶는다 (다른 클래스는 절대 안 섞임). 학습 없음 — 결합 특징 벡터 + z-정규화 + 블록 가중 +
union-find 거리 임계. 특징:
  * 색: Lab 중앙값 (밝기 L 을 config.CLUSTER_L_WEIGHT 로 다운웨이트 → 조명차로 같은 종류가
        갈리는 것 완화, 색상/채도 a·b 는 유지)
  * 질감: 마스크 내 그레이 통계 (표준편차·평균 그래디언트·라플라시안 분산·엔트로피, 회전 불변)
  * 크기: 종횡비 + log면적 (마스크 2D 기준, 스케일 견고)
출력: 객체별 group_id(전역 고유 int) + group_label('<클래스명>#<서브그룹>', 예 'Carton#0').
"""
import cv2
import numpy as np

from .. import config

# 차원별 전역 스케일(= final_dataset_v11 객체 1809개 기준 std). per-scene 정규화는 객체 수가
# 적으면(예: 한 클래스에 2~3개) std 가 작아 미세차가 증폭돼 다 쪼개진다 → 전역 고정 스케일로
# 정규화해 객체 수와 무관하게 일관된 거리. 특징 순서: [L*, a, b, std_gray, mean_grad,
# lap_var, entropy, aspect, log_area]
_SCALE = np.array([18.206, 13.741, 16.841, 12.042, 24.471, 1460.753, 0.449, 1.349, 1.232])


def _uf_find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


def _descriptor(mask, gray, lab, gradmag, lap):
    """객체 하나의 결합 특징 벡터(색3 + 질감4 + 크기2)."""
    er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))  # 가장자리 색 오염 제거
    sel = er > 0
    if int(sel.sum()) < 20:
        sel = mask.astype(bool)
    # 색: Lab 중앙값(반사/그림자 강건), 밝기 L 다운웨이트
    col = np.median(lab[sel], axis=0).astype(float)
    col[0] *= config.CLUSTER_L_WEIGHT
    # 질감: 회전 불변 그레이 통계
    g = gray[sel].astype(float)
    hist, _ = np.histogram(g, bins=16, range=(0, 255))
    p = hist / (hist.sum() + 1e-9)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    tex = [float(g.std()), float(gradmag[sel].mean()), float(lap[sel].var()), ent]
    # 크기: 종횡비 + log면적
    ys, xs = np.where(sel)
    (_c, (w, h), _r) = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
    long_s, short_s = max(w, h), min(w, h)
    size = [long_s / max(short_s, 1.0), float(np.log(int(sel.sum()) + 1.0))]
    return np.array(list(col) + tex + size, dtype=float)


def cluster_by_appearance(masks, class_ids, color_bgr, cfg=config):
    """masks(list[HxW bool|None]) + class_ids → (group_ids: list[int], group_labels: list[str]).

    클래스별로 나눠 각 클래스 안에서만 결합특징 유사도로 union-find 클러스터링.
    입력 순서에 정렬된 리스트 반환. 특징 없음/너무 작은 마스크는 group_id=-1.
    """
    n = len(masks)
    group_ids = [-1] * n
    group_labels = [""] * n
    if n == 0:
        return group_ids, group_labels

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradmag = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    bw = np.array([cfg.CLUSTER_W_COLOR] * 3 + [cfg.CLUSTER_W_TEX] * 4
                  + [cfg.CLUSTER_W_SIZE] * 2, dtype=float)
    tau = cfg.CLUSTER_TAU

    feats = [None] * n
    for i, m in enumerate(masks):
        if m is None or int(np.count_nonzero(m)) < cfg.CLUSTER_MIN_PX:
            continue
        feats[i] = _descriptor(m, gray, lab, gradmag, lap)

    gid = 0
    for cid in sorted({int(c) for c in class_ids}):
        idxs = [i for i in range(n) if int(class_ids[i]) == cid and feats[i] is not None]
        if not idxs:
            continue
        if len(idxs) == 1:
            local = [0]
        else:
            Z = np.stack([feats[i] for i in idxs]) / _SCALE * bw   # 전역 고정 스케일 정규화 + 블록 가중
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
