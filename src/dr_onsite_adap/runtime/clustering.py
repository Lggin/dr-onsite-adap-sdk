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
import json
import os

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


def _uf_group(vecs, scale, bw, tau):
    """스케일·가중된 특징들을 거리<tau 로 연결 → 연결요소 로컬 라벨(0,1,2..) 반환."""
    n = len(vecs)
    if n == 1:
        return [0]
    Z = np.stack(vecs) / scale * bw
    par = list(range(n))
    for a in range(n):
        for b in range(a + 1, n):
            if np.linalg.norm(Z[a] - Z[b]) < tau:
                par[_uf_find(par, a)] = _uf_find(par, b)
    roots = {}
    return [roots.setdefault(_uf_find(par, a), len(roots)) for a in range(n)]


def _scale_bw(size_scale, cfg):
    """특징 스케일 벡터(색3+질감2+크기2) + 블록 가중 벡터."""
    scale = np.concatenate([_SCALE_COLOR, _SCALE_TEX, [size_scale, size_scale]])
    bw = np.array([cfg.CLUSTER_W_COLOR] * 3 + [cfg.CLUSTER_W_TEX] * 2
                  + [cfg.CLUSTER_W_SIZE] * 2, dtype=float)
    return scale, bw


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
    scale, bw = _scale_bw(size_scale, cfg)
    tau = cfg.CLUSTER_TAU

    gid = 0
    for cid in sorted({int(class_ids[i]) for i in valid}):
        idxs = [i for i in valid if int(class_ids[i]) == cid]
        local = _uf_group([feats_raw[i] for i in idxs], scale, bw, tau)
        name = config.CLASSES.get(cid, (str(cid),))[0]
        for a, i in enumerate(idxs):
            group_ids[i] = gid + local[a]
            group_labels[i] = f"{name}#{local[a]}"
        gid += max(local) + 1

    return group_ids, group_labels


# ─────────────────────────── 프로토타입(정답) 모드 ───────────────────────────
# 클래스당 최대 N개 레퍼런스 특징을 저장(뱅크)하고, 검출 객체를 자기 클래스의 최근접
# 정답에 배정한다. 그룹 개수가 안정적이고(≤N), 그룹에 의미있는 이름을 줄 수 있다.
# 뱅크 포맷(JSON): { "<cid>": [ {"name":str, "feat":[7], "long_mm":.., "short_mm":..}, ... ] }

def prototype_feature(mask, color_bgr, size_mm, cfg=config):
    """객체 하나(마스크+물리치수)의 프로토타입 특징 7차원 반환(등록용)."""
    _g, lab, detail, edges = _frame_maps(color_bgr)
    return _descriptor(mask.astype(bool), lab, detail, edges, size_mm)


def load_prototypes(path=None):
    """프로토타입 뱅크 JSON 로드(없으면 빈 dict)."""
    p = os.path.expanduser(path or config.PROTOTYPE_PATH)
    if os.path.isfile(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_prototypes(bank, path=None):
    p = os.path.expanduser(path or config.PROTOTYPE_PATH)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    json.dump(bank, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def add_prototype(bank, cid, name, feat, long_mm, short_mm, cfg=config):
    """뱅크에 정답 추가(클래스당 최대 CLUSTER_MAX_PROTOTYPES, 초과 시 가장 오래된 것 제거)."""
    lst = bank.setdefault(str(int(cid)), [])
    lst.append({"name": name, "feat": [float(x) for x in feat],
                "long_mm": float(long_mm), "short_mm": float(short_mm)})
    if len(lst) > cfg.CLUSTER_MAX_PROTOTYPES:
        lst.pop(0)
    return bank


def bank_count(bank):
    return sum(len(v) for v in bank.values())


def assign_to_prototypes(masks, class_ids, color_bgr, sizes_mm, bank, cfg=config):
    """검출 객체를 클래스별 최근접 정답에 배정. 반환 (group_ids, group_labels).

    정답 있는 클래스: 최근접 정답(거리<CLUSTER_MATCH_TAU) 이름/그룹, 멀면 '<클래스>#기타'.
    정답 없는 클래스: 비지도 클러스터링 폴백(<클래스>#0,1,..). group_id = cid*16 + slot(안정).
    """
    n = len(masks)
    group_ids = [-1] * n
    group_labels = [""] * n
    if n == 0:
        return group_ids, group_labels
    _g, lab, detail, edges = _frame_maps(color_bgr)
    scale, bw = _scale_bw(cfg.CLUSTER_SIZE_SCALE_MM, cfg)
    have = sizes_mm is not None

    feat = [None] * n
    for i, m in enumerate(masks):
        if m is None or int(np.count_nonzero(m)) < cfg.CLUSTER_MIN_PX:
            continue
        sm = sizes_mm[i] if (have and sizes_mm[i] is not None) else None
        if sm is None:
            ys, xs = np.where(m)
            (_c, (w, h), _r) = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            sm = (max(w, h), min(w, h))
        feat[i] = _descriptor(m.astype(bool), lab, detail, edges, sm)

    valid = [i for i in range(n) if feat[i] is not None]
    for cid in sorted({int(class_ids[i]) for i in valid}):
        idxs = [i for i in valid if int(class_ids[i]) == cid]
        name_cls = config.CLASSES.get(cid, (str(cid),))[0]
        protos = bank.get(str(cid), [])
        if protos:
            P = np.array([p["feat"] for p in protos], dtype=float)
            for i in idxs:
                dd = np.linalg.norm((feat[i] - P) / scale * bw, axis=1)
                j = int(np.argmin(dd))
                if dd[j] < cfg.CLUSTER_MATCH_TAU:
                    group_ids[i] = cid * 16 + j
                    group_labels[i] = protos[j].get("name") or f"{name_cls}#{j}"
                else:
                    group_ids[i] = cid * 16 + 15
                    group_labels[i] = f"{name_cls}#기타"
        else:   # 정답 없는 클래스 → 비지도 폴백
            local = _uf_group([feat[i] for i in idxs], scale, bw, cfg.CLUSTER_TAU)
            for a, i in enumerate(idxs):
                group_ids[i] = cid * 16 + local[a]
                group_labels[i] = f"{name_cls}#{local[a]}"
    return group_ids, group_labels
