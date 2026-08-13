"""외형 기반 클러스터링 — 검출된 객체를 '같은 클래스 안에서' 물리크기·색·질감으로 서브그룹핑.

분류(검출)가 먼저다. 클래스별로 나눈 뒤 각 클래스 **안에서만** 비슷한 것끼리 묶는다.
학습 없음 — 결합 특징 벡터 + 고정 스케일 정규화 + 블록 가중 + 거리 임계.

특징(7차원): 색(3, Lab 중앙값·밝기 L 다운웨이트) + 질감(2) + 크기(2, 물리 long/short mm).
  * 질감은 **윗면을 정면·고정크기 패치로 워프(rectify)한 뒤** 로그-하이패스 std + Canny
    에지밀도로 측정 → 거리/각도 정규화되어 "같은 박스"가 일관된 값(노이즈 바닥↓). 기하(평면·
    방향·물리치수)가 없으면(2D 검출) 마스크에서 직접(폴백).
  * 색/질감 계산은 CLUSTER_FEAT_SCALE(기본 0.5) 로 **저해상에서** 수행(속도; 색·질감 통계엔 무해).
출력: 객체별 group_id + group_label('<클래스명>#<서브그룹>').
"""
import json
import os

import cv2
import numpy as np

from .. import config

_SCALE_COLOR = np.array([18.0, 14.0, 17.0])
_SCALE_TEX = np.array([0.04, 8.0])   # (로그HP std, Canny 밀도) — 128px 패치 기준
_PATCH = 128

GROUP_PALETTE = [(0, 255, 255), (255, 255, 0), (0, 140, 255), (255, 0, 128),
                 (128, 255, 0), (255, 255, 255), (0, 215, 255), (255, 128, 0),
                 (203, 192, 255), (128, 128, 255)]


def group_color(group_id):
    if group_id is None or group_id < 0:
        return None
    return GROUP_PALETTE[group_id % len(GROUP_PALETTE)]


def _uf_find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


def _uf_group(vecs, scale, bw, tau):
    """단일연결(single-linkage) — 거리<tau 인 쌍을 union-find 로 잇는다.

    ⚠️ 사슬(chaining) 에 취약하다: 두 무리 사이에 **가까운 쌍이 하나만** 생겨도 무리
    전체가 통째로 붙는다. 2026-08-13 실측 — 파란 인쇄박스 308개 무리와 네이비 무지박스
    6개 무리가 최소 쌍거리 1.326 vs tau 1.30 으로 **0.026 차이**로 겨우 갈려 있었고,
    거리 기준을 조금만 바꾸자 바로 병합됐다(두 무리의 중심 거리는 4.42→5.61 로 오히려
    멀어졌는데도). 그래서 기본은 완전연결(_group_link)을 쓴다.
    """
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


def _group_link(vecs, scale, bw, tau, linkage="complete"):
    """응집형 군집. linkage='single'|'complete'|'average'.

    complete — 두 무리를 합치려면 **모든 쌍**이 tau 안이어야 한다. 이상치 한 개가 다리를
               놓아 무리가 통째로 붙는 사슬 현상이 없다.
    average  — 무리 간 평균 거리 기준. complete 보다 관대하다.
    Lance-Williams 갱신으로 O(n²) 메모리, 현장 규모(프레임당 수십 개)에 충분하다.
    """
    n = len(vecs)
    if n == 1:
        return [0]
    if linkage == "single":
        return _uf_group(vecs, scale, bw, tau)
    Z = np.stack(vecs) / scale * bw
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    np.fill_diagonal(D, np.inf)
    members = [[i] for i in range(n)]
    alive = list(range(n))
    while len(alive) > 1:
        sub = D[np.ix_(alive, alive)]
        k = int(np.argmin(sub))
        ia, ib = divmod(k, len(alive))
        if sub[ia, ib] >= tau:
            break
        a, b = alive[ia], alive[ib]
        na, nb = len(members[a]), len(members[b])
        for c in alive:
            if c in (a, b):
                continue
            if linkage == "average":
                d = (na * D[a, c] + nb * D[b, c]) / (na + nb)
            else:                                   # complete
                d = max(D[a, c], D[b, c])
            D[a, c] = D[c, a] = d
        members[a] = members[a] + members[b]
        alive.remove(b)
    out = [0] * n
    for gid, a in enumerate(alive):
        for i in members[a]:
            out[i] = gid
    return out


def _scale_bw(size_scale, cfg):
    scale = np.concatenate([_SCALE_COLOR, _SCALE_TEX, [size_scale, size_scale]])
    bw = np.array([cfg.CLUSTER_W_COLOR] * 3 + [cfg.CLUSTER_W_TEX] * 2
                  + [cfg.CLUSTER_W_SIZE] * 2, dtype=float)
    return scale, bw


def _warp_patch(bgr, geom, intr):
    """윗면을 정면·축정렬 고정크기(_PATCH) 패치로 워프. geom=(center_3d, long_axis, short_axis,
    long_mm, short_mm). intr=(fx,fy,cx,cy) — bgr 해상도 기준. 실패 시 None."""
    fx, fy, cx, cy = intr
    c = np.asarray(geom[0], float); ua = np.asarray(geom[1], float); va = np.asarray(geom[2], float)
    hl = float(geom[3]) / 2000.0; hs = float(geom[4]) / 2000.0
    if hl <= 0 or hs <= 0:
        return None
    pts = [c + a * hl * ua + b * hs * va for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    src = []
    for p in pts:
        if p[2] <= 1e-6:
            return None
        src.append([p[0] / p[2] * fx + cx, p[1] / p[2] * fy + cy])
    dst = np.array([[0, 0], [_PATCH, 0], [_PATCH, _PATCH], [0, _PATCH]], np.float32)
    Hm = cv2.getPerspectiveTransform(np.array(src, np.float32), dst)
    return cv2.warpPerspective(bgr, Hm, (_PATCH, _PATCH))


def _patch_texture(patch):
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    cr = g[16:112, 16:112] if g.shape[0] >= 128 else g          # 중앙 크롭(경계 배제)
    gl = np.log1p(cr.astype(np.float32))
    detail = gl - cv2.GaussianBlur(gl, (0, 0), 4.0)             # 조명(저주파) 제거
    edges = cv2.Canny(cr, 40, 120)
    return float(detail.std()), float(edges.mean())


def _build_features(masks, sizes_mm, geoms, color_bgr, intrinsics, cfg):
    """객체별 7차원 특징 리스트 + 물리크기여부 리스트 반환. 색/질감은 저해상에서."""
    n = len(masks)
    s = cfg.CLUSTER_FEAT_SCALE
    if s < 0.999:
        small = cv2.resize(color_bgr, (max(1, int(color_bgr.shape[1] * s)),
                                       max(1, int(color_bgr.shape[0] * s))))
    else:
        small, s = color_bgr, 1.0
    Hs, Ws = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    geoms = geoms if geoms is not None else [None] * n
    sizes_mm = sizes_mm if sizes_mm is not None else [None] * n

    # ── 프레임 기준 색 정규화 ────────────────────────────────────────────────
    # 조명·노출이 바뀌면 Lab 중앙값이 통째로 이동해, **다른 시점에 등록한 정답 뱅크**와의
    # 거리가 커져 '기타'로 이탈한다(2026-08-12 실측: 대비 ×1.5 에서 유지율 18.5%).
    # 프레임 전체 통계로 각 객체 색을 상대화하면 전역 변화가 상쇄된다.
    #   "median" — 중앙값만 뺀다. 조명 오프셋(밝기 이동)에 대응.
    #   "robust" — 중앙값을 빼고 MAD 로 나눈다. **대비(곱셈) 변화까지** 대응.
    # 프레임 안에서만 묶는 비지도 경로는 객체 간 차이만 쓰므로 원래 오프셋에 불변이고,
    # 이 정규화가 실제로 이득을 주는 곳은 프로토타입(정답) 배정 경로다.
    # ⚠️ 특징 기저가 바뀌므로 **기존 정답 뱅크는 무효** — '정답 비우기' 후 재등록 필요.
    _norm = getattr(cfg, "CLUSTER_COLOR_NORM", "none")
    _c_med = _c_spr = None
    if _norm in ("median", "robust"):
        _flat = lab.reshape(-1, 3).astype(np.float32)
        _c_med = np.median(_flat, axis=0)
        if _norm == "robust":
            # MAD×1.4826 = 정규분포에서 표준편차와 같은 스케일. 0 나눗셈 방지.
            _c_spr = np.maximum(np.median(np.abs(_flat - _c_med), axis=0) * 1.4826, 1e-3)

    raw = {}
    def _raw_maps():
        if not raw:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gl = np.log1p(gray.astype(np.float32))
            raw["d"] = gl - cv2.GaussianBlur(gl, (0, 0), max(6.0, 0.02 * Ws))
            raw["e"] = cv2.Canny(gray, 40, 120).astype(np.float32)
        return raw["d"], raw["e"]

    feats = [None] * n
    phys = [False] * n
    for i, m in enumerate(masks):
        if m is None or int(np.count_nonzero(m)) < cfg.CLUSTER_MIN_PX:
            continue
        ms = (m.astype(bool) if s == 1.0
              else cv2.resize(m.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0)
        if int(ms.sum()) < 8:
            ms = m.astype(bool)
            if int(ms.sum()) < 4:
                continue
        col = np.median(lab[ms], axis=0).astype(float)
        if _c_med is not None:
            col = col - _c_med
            if _c_spr is not None:
                # MAD 단위 → 원래 Lab 과 비슷한 눈금으로 되돌린다. 그래야 CLUSTER_TAU /
                # CLUSTER_MATCH_TAU 의 의미가 정규화 전후로 유지된다.
                col = col / _c_spr * cfg.CLUSTER_COLOR_NORM_SPREAD
        col[0] *= cfg.CLUSTER_L_WEIGHT
        # 질감: 워프 패치 우선(풀해상도 소스 — 인쇄 디테일 보존, 비용은 128px 출력 기준),
        #       실패/기하없음 시 저해상 마스크에서 직접(폴백)
        tex = None
        g = geoms[i]
        if g is not None and intrinsics is not None:
            patch = _warp_patch(color_bgr, g, intrinsics)
            if patch is not None:
                tex = _patch_texture(patch)
        if tex is None:
            d, e = _raw_maps()
            tex = (float(d[ms].std()), float(e[ms].mean()))
        # 크기: 물리 mm 우선, 없으면 원본 마스크 픽셀
        sm = sizes_mm[i]
        if sm is not None:
            phys[i] = True
        else:
            ys, xs = np.where(m)
            (_c, (w, h), _r) = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            sm = (max(w, h), min(w, h))
        feats[i] = np.array([col[0], col[1], col[2], tex[0], tex[1],
                             float(sm[0]), float(sm[1])], dtype=float)
    return feats, phys


def _size_scale(feats, phys, valid, cfg):
    if all(phys[i] for i in valid):
        return cfg.CLUSTER_SIZE_SCALE_MM
    return max(0.15 * float(np.median([feats[i][5] for i in valid])), 1.0)


def cluster_by_appearance(masks, class_ids, color_bgr, sizes_mm=None,
                          geoms=None, intrinsics=None, cfg=config):
    """비지도: 클래스별로 나눠 각 클래스 안에서 거리<TAU 를 union-find 로 묶음."""
    n = len(masks)
    group_ids = [-1] * n
    group_labels = [""] * n
    if n == 0:
        return group_ids, group_labels
    feats, phys = _build_features(masks, sizes_mm, geoms, color_bgr, intrinsics, cfg)
    valid = [i for i in range(n) if feats[i] is not None]
    if not valid:
        return group_ids, group_labels
    scale, bw = _scale_bw(_size_scale(feats, phys, valid, cfg), cfg)
    tau = cfg.CLUSTER_TAU
    gid = 0
    for cid in sorted({int(class_ids[i]) for i in valid}):
        idxs = [i for i in valid if int(class_ids[i]) == cid]
        local = _group_link([feats[i] for i in idxs], scale, bw, tau,
                            getattr(cfg, 'CLUSTER_LINKAGE', 'complete'))
        name = config.CLASSES.get(cid, (str(cid),))[0]
        for a, i in enumerate(idxs):
            group_ids[i] = gid + local[a]
            group_labels[i] = f"{name}#{local[a]}"
        gid += max(local) + 1
    return group_ids, group_labels


# ─────────────────────────── 프로토타입(정답) 모드 ───────────────────────────
def prototype_feature(mask, color_bgr, size_mm, geom=None, intrinsics=None, cfg=config):
    """객체 하나의 7차원 특징(등록용). 워프 텍스처 쓰려면 geom+intrinsics 필요."""
    feats, _ = _build_features([mask], [size_mm], [geom], color_bgr, intrinsics, cfg)
    return feats[0]


def load_prototypes(path=None):
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
    lst = bank.setdefault(str(int(cid)), [])
    lst.append({"name": name, "feat": [float(x) for x in feat],
                "long_mm": float(long_mm), "short_mm": float(short_mm)})
    if len(lst) > cfg.CLUSTER_MAX_PROTOTYPES:
        lst.pop(0)
    return bank


def bank_count(bank):
    return sum(len(v) for v in bank.values())


def assign_to_prototypes(masks, class_ids, color_bgr, sizes_mm, bank,
                         geoms=None, intrinsics=None, cfg=config):
    """검출 객체를 클래스별 최근접 정답에 배정. 정답 없는 클래스는 비지도 폴백.
    같은 이름 정답 여러 개 → 한 그룹. group_id = cid*16 + slot(안정)."""
    n = len(masks)
    group_ids = [-1] * n
    group_labels = [""] * n
    if n == 0:
        return group_ids, group_labels
    feats, phys = _build_features(masks, sizes_mm, geoms, color_bgr, intrinsics, cfg)
    valid = [i for i in range(n) if feats[i] is not None]
    if not valid:
        return group_ids, group_labels
    scale, bw = _scale_bw(_size_scale(feats, phys, valid, cfg), cfg)
    for cid in sorted({int(class_ids[i]) for i in valid}):
        idxs = [i for i in valid if int(class_ids[i]) == cid]
        name_cls = config.CLASSES.get(cid, (str(cid),))[0]
        protos = bank.get(str(cid), [])
        if protos:
            P = np.array([p["feat"] for p in protos], dtype=float)
            pnames = [p.get("name") or f"{name_cls}#{k}" for k, p in enumerate(protos)]
            uniq = []
            for nm in pnames:
                if nm not in uniq:
                    uniq.append(nm)
            for i in idxs:
                dd = np.linalg.norm((feats[i] - P) / scale * bw, axis=1)
                j = int(np.argmin(dd))
                if dd[j] < cfg.CLUSTER_MATCH_TAU:
                    group_ids[i] = cid * 16 + uniq.index(pnames[j])
                    group_labels[i] = pnames[j]
                else:
                    group_ids[i] = cid * 16 + 15
                    group_labels[i] = f"{name_cls}#기타"
        else:
            local = _group_link([feats[i] for i in idxs], scale, bw, cfg.CLUSTER_TAU,
                                getattr(cfg, 'CLUSTER_LINKAGE', 'complete'))
            for a, i in enumerate(idxs):
                group_ids[i] = cid * 16 + local[a]
                group_labels[i] = f"{name_cls}#{local[a]}"
    return group_ids, group_labels
