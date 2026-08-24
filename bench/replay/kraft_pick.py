"""크래프트 근접 리플레이 선정 — 배포 셋과 A/B 가 **같은 규칙**을 쓰게 하는 공용 모듈.

`build_kraft200.py`(실제 배포 셋 생성)와 `build_bench_round.py --replay-pick kraft`
(A/B 벤치)가 이 모듈을 함께 쓴다. 규칙이 갈리면 A/B 가 배포될 것과 다른 것을 재게 된다.

## 크래프트의 정의 (측정 기반)

`~/box_types/` 크래프트 5종 vs 네이비·파랑 3종의 축별 코헨 d:

    b (황청)  +4.82 ★   L (밝기) +4.20 ★   tex_hp −2.30 ★   b_iqr −1.50 ★
    ──── |d| < 1 ────
    size_long +0.85    a +0.61    chroma −0.07  ← 채도는 무용(네이비도 저채도)

크래프트 = **밝고 · 노란쪽 · 색이 균일하고 · 표면이 매끈한**.

## 표준화 척도는 고정 기준을 쓴다

MU/SD 는 `ultimate` 의 Carton 모집단(336,952)에서 한 번 구해 **고정**한다. 소스마다
다시 잡으면 소스가 바뀔 때 거리 단위가 바뀌어 실험 간 비교가 끊긴다. 표준화는 단위
변환일 뿐이므로 하나의 척도를 쓰는 것이 낫다.
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import cv2
from pycocotools import mask as maskutil

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', 'grouping'), os.path.join(_HERE, '..', 'eda')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gridsearch_feat import NAMES, f3_rich       # noqa: E402

AX = NAMES['F3_rich']
KRAFT_AXES = ['L', 'b', 'b_iqr', 'tex_hp']
KI = [AX.index(n) for n in KRAFT_AXES]
KRAFT_TYPES = ['pacific_long', 'pacific_short', 'swanson', 'plain_small', 'red']
RARE_FIRST = ['Slip Sheet', 'Pallet', 'Tote', 'Shrink Wrap']
MIN_AREA = 40
MAX_CARTON_PER_IMG = 40      # 한 장에 228개까지 있어 상한을 둔다(중앙값엔 충분)


def load_reference(S):
    """(기준점 ANCHOR, 척도 MU/SD, 표준화 기준점 ZA)."""
    XB = np.load(os.path.join(S, 'kraft_X.npy'))
    LB = np.load(os.path.join(S, 'kraft_L.npy'), allow_pickle=True)
    anchor = np.median(XB[np.isin(LB, KRAFT_TYPES)][:, KI], axis=0)

    allf = pickle.load(open(os.path.join(S, 'allfeat_train.pkl'), 'rb'))
    cls = np.array([m[0] for m in allf['meta']])
    seen, keep = set(), np.zeros(len(cls), bool)
    for i, m in enumerate(allf['meta']):
        k = (m[2], allf['F3'][i].tobytes())
        if k not in seen:
            seen.add(k)
            keep[i] = True
    pop = allf['F3'][(cls == 'Carton') & keep][:, KI].astype(np.float64)
    mu, sd = pop.mean(0), pop.std(0)
    return anchor, mu, sd, (anchor - mu) / sd, len(pop)


def _decode(a, H, W):
    seg = a.get('segmentation')
    try:
        if isinstance(seg, dict):
            r = (maskutil.frPyObjects(seg, H, W)
                 if isinstance(seg.get('counts'), list) else seg)
            m = maskutil.decode(r)
            return m[..., 0] if m.ndim == 3 else m
        m = np.zeros((H, W), np.uint8)
        for poly in seg or []:
            if len(poly) >= 6:
                cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
        return m
    except Exception:
        return None


def dedup_anns(coco, cat_name_of):
    """(파일명, bbox, 클래스) 중복 주석을 걷어낸다.

    ⚠️ `ultimate`/`ownsource_v2` 는 현장 실촬 40장을 COCO images 에 각각 50번 등록한다.
    걷지 않으면 같은 사진이 리플레이의 상당수를 차지하고, 거리 통계도 왜곡된다.
    """
    id2n = {im['id']: im['file_name'] for im in coco['images']}
    cls_of, ann_of, got, dup = defaultdict(set), defaultdict(list), set(), 0
    for a in coco['annotations']:
        fn = id2n[a['image_id']]
        x, y, w, h = [int(v) for v in a['bbox']]
        if w * h < MIN_AREA:
            continue
        key = (fn, x, y, w, h, a['category_id'])
        if key in got:
            dup += 1
            continue
        got.add(key)
        cls_of[fn].add(cat_name_of(a['category_id']))
        ann_of[fn].append(a)
    return cls_of, ann_of, dup


def score_images(coco, img_dir, MU, SD, ZA, cache=None, cat_name_of=None,
                 verbose=True):
    """파일명 → (크래프트 좌표 z, 거리). Carton 이 없는 이미지는 (None, inf).

    거리 = 그 이미지의 **Carton 들의 4축 중앙값**을 표준화해 기준점까지의 유클리드 거리.
    """
    if cache and os.path.exists(cache):
        d = pickle.load(open(cache, 'rb'))
        if verbose:
            print(f"   크래프트 점수 캐시 사용: {cache}")
        return d['Z'], d['dist'], d['cls'], d['ncar']

    cat_name_of = cat_name_of or (lambda c: {x['id']: x['name']
                                             for x in coco['categories']}[c])
    cls_of, ann_of, dup = dedup_anns(coco, cat_name_of)
    files = sorted(f for f in cls_of if os.path.isfile(os.path.join(img_dir, f)))
    if verbose:
        print(f"   고유 이미지 {len(files):,} (중복 주석 {dup:,} 제거)")

    Z, dist, ncar = {}, {}, Counter()
    import time
    t0 = time.time()
    for n, fn in enumerate(files):
        carts = [a for a in ann_of[fn] if cat_name_of(a['category_id']) == 'Carton']
        ncar[fn] = len(carts)
        if not carts:
            Z[fn], dist[fn] = None, np.inf
            continue
        bgr = cv2.imread(os.path.join(img_dir, fn))
        if bgr is None:
            Z[fn], dist[fn] = None, np.inf
            continue
        H, W = bgr.shape[:2]
        vs = []
        for a in carts[:MAX_CARTON_PER_IMG]:
            x, y, w, h = [int(v) for v in a['bbox']]
            mk = _decode(a, H, W)
            if mk is None:
                continue
            v = f3_rich(bgr[max(0, y):y + h, max(0, x):x + w],
                        mk[max(0, y):y + h, max(0, x):x + w], None)
            if v is not None:
                vs.append([v[i] for i in KI])
        if not vs:
            Z[fn], dist[fn] = None, np.inf
            continue
        z = (np.median(np.array(vs, float), axis=0) - MU) / SD
        Z[fn] = z.tolist()
        dist[fn] = float(np.linalg.norm(z - ZA))
        if verbose and n % 3000 == 0 and n:
            print(f"      {n:,}/{len(files):,}  {time.time()-t0:.0f}초", flush=True)
    cls = {f: sorted(cls_of[f]) for f in files}
    if cache:
        pickle.dump(dict(Z=Z, dist=dist, cls=cls, ncar=dict(ncar)), open(cache, 'wb'))
    return Z, dist, cls, dict(ncar)


def kraft_pick(files, cls, dist, Z, K, floors, rng, ncar=None,
               floor_carton_frac=0.6, pool_pct=25.0, far_frac=0.0,
               seed=0, verbose=True):
    """크래프트 근접 2단계 선정. 반환: (뽑힌 파일명 리스트, 클래스별 확보수, far 집합).

    1단계 클래스 보장 — 희귀 우선. 보장의 `floor_carton_frac` 은 Carton 동반(슬롯이 두
      몫을 한다) + 크래프트 근접, 나머지는 그 클래스 **본래 분포**에서.
      100% 를 Carton 동반으로 하면 Carton 없는 출처(슈링크랩 단품 렌더 등)가 통째로
      빠진다 — 베이스가 그 클래스를 배운 곳이라 위험하다.
    2단계 남은 슬롯 — 크래프트 적격 풀(근접 상위 pool_pct%)에서 **k-medoid 로 고르게**.
      가까운 것부터 자르면 같은 데이터셋의 비슷한 사진만 쌓인다.
    `far_frac` > 0 이면 그 비율만큼 크래프트에서 **먼** 쪽을 일부러 섞는다(valid 용).
    """
    from sklearn.cluster import KMeans
    NC = Counter(ncar or {})
    picked, chosen, have, far = [], set(), Counter(), set()

    def take(f):
        # 이중 담기 방어 — split_near_far 는 후보가 적으면 앞·뒤 조각이 겹친다
        if f in chosen:
            return False
        picked.append(f)
        chosen.add(f)
        for c in cls[f]:
            have[c] += 1
        return True

    def split_near_far(cand, n):
        # 거리를 0.25 폭으로 묶고 같은 묶음 안에서는 Carton 이 많은 것 먼저.
        # 회귀 측정에서 인스턴스 수가 곧 표본 크기다.
        cand = sorted(cand, key=lambda f: (round(dist[f] / 0.25), -NC.get(f, 0)))
        if far_frac <= 0 or len(cand) <= 1:
            return cand[:n], []
        nf = min(int(round(n * far_frac)), max(0, len(cand) // 2))
        nn = min(n - nf, len(cand) - nf)
        return cand[:nn], list(reversed(cand))[:nf]

    for c in RARE_FIRST:
        need = floors.get(c, 0)
        if not need:
            continue
        cand = [f for f in files if c in cls[f] and f not in chosen]
        withc = [f for f in cand if 'Carton' in cls[f]]
        noc = [f for f in cand if 'Carton' not in cls[f]]
        rng.shuffle(noc)                 # Carton 이 없으면 크래프트 거리를 잴 수 없다
        quota_c = int(round(need * floor_carton_frac))
        near, farc = split_near_far(withc, quota_c)
        far.update(farc)
        plan = list(dict.fromkeys(near + farc + noc[:need - quota_c]))
        if len(plan) < need:
            plan += [f for f in withc + noc if f not in plan][:need - len(plan)]
        n0 = len(picked)
        for f in plan:
            if have[c] >= need or len(picked) >= K:
                break
            take(f)
        if verbose:
            nc = sum(1 for f in picked[n0:] if 'Carton' in cls[f])
            print(f"      보장 {c:<12}목표 {need:>3} 확보 {have[c]:>3} "
                  f"(Carton 동반 {nc}/{len(picked)-n0}) 누적 {len(picked):>3}")

    rest = K - len(picked)
    if rest > 0:
        pool = [f for f in files if f not in chosen and np.isfinite(dist[f])]
        if pool:
            thr = float(np.percentile([dist[f] for f in pool], pool_pct))
            elig = [f for f in pool if dist[f] <= thr]
            if verbose:
                print(f"      적격 풀 {len(elig):,}장 (상위 {pool_pct:.0f}%, ≤ {thr:.2f})")
            nf = int(round(min(rest, len(elig)) * far_frac))
            if nf:
                outside = [f for f in pool if dist[f] > thr]
                if outside:
                    Zo = np.array([Z[f] for f in outside])
                    km = KMeans(min(nf, len(outside)), n_init=4, random_state=seed).fit(Zo)
                    for k in range(km.n_clusters):
                        m = np.where(km.labels_ == k)[0]
                        if not len(m):
                            continue
                        j = m[int(np.argmin(np.linalg.norm(
                            Zo[m] - km.cluster_centers_[k], axis=1)))]
                        far.add(outside[j])
                        take(outside[j])
            n = min(K - len(picked), len(elig))
            if n > 0:
                Zs = np.array([Z[f] for f in elig])
                km = KMeans(n, n_init=4, random_state=seed).fit(Zs)
                for k in range(n):
                    m = np.where(km.labels_ == k)[0]
                    if not len(m):
                        continue
                    j = m[int(np.argmin(np.linalg.norm(
                        Zs[m] - km.cluster_centers_[k], axis=1)))]
                    take(elig[j])
            for f in sorted(elig, key=lambda x: dist[x]):     # 빈 군집 보충
                if len(picked) >= K:
                    break
                take(f)
    return picked, have, far
