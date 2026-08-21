"""크래프트 갈색 Carton 망각 방지용 고정 리플레이 셋 (train 200 + valid 100).

`replay_fixed_v11`(train 400) 을 용량 때문에 200 으로 줄인다. 그냥 절반으로 무작위
추출하면 회귀가 −0.0045~−0.0054 로 기준선(−0.005)을 넘나든다고 이미 측정됐다
(`bench/replay/ab_replay_pick.md`). 그래서 **장수를 줄이는 대신 목표를 좁힌다** —
지키려는 대상을 "학습셋 전반"에서 **"크래프트 갈색 Carton"** 으로 한정한다.

## 크래프트를 무엇으로 정의했나 (측정 기반)

직접 촬영한 8종(`~/box_types/`)을 크래프트 계열 5종(pacific_long/pacific_short/
swanson/plain_small/red)과 네이비·파랑 3종(navy_small/navy_long/blue_printed)으로
갈라 축별 코헨 d 를 재면 이렇게 갈린다.

    b (황청)    141.5 vs 113.6   d = +4.82   ★
    L (밝기)    182.0 vs  78.0   d = +4.20   ★
    tex_hp        0.1 vs   0.2   d = -2.30   ★
    b_iqr         2.7 vs  10.7   d = -1.50   ★
    ──── 이하 |d| < 1 ────
    chroma       14.3 vs  14.7   d = -0.07   ← 채도는 무용하다

크래프트 = **밝고 · 노란쪽 · 색이 균일하고 · 표면이 매끈한**. 채도(갈색이라는 직관)는
판별력이 없다 — 네이비도 저채도이기 때문이다. 그래서 위 네 축만 쓴다.

## 선정 규칙

1. 이미지마다 그 이미지의 **Carton 들의 4축 중앙값**을 구한다
2. Carton 모집단(33.7만)으로 표준화하고 **현장 크래프트 기준점**까지 거리를 잰다
3. **모든 단계의 선택을 이 거리 순으로** 한다 — 그래서 200장 전체가 크래프트로 치우친다
4. 단 **클래스 보존 층은 유지**한다. 희귀 클래스부터 최소 보장을 채운다. 이게 없으면
   팔레트가 120/120 → 0/120 으로 전멸했던 2026-08-04 실측이 재현된다. 보장 안에서도
   Carton 동반 + 크래프트 근접을 우선해 한 슬롯이 두 몫을 하게 한다
5. 남은 슬롯은 크래프트 적격 풀(근접 상위 N%) 안에서 **k-medoid 로 고르게** 뽑는다.
   가까운 것부터 자르면 같은 공개 데이터셋의 비슷한 사진만 쌓인다

## 걸러내는 것 (replay_fixed_v11 과 동일)

- **중복 주석 ×50** — 현장 실촬 40장이 COCO images 에 각각 50번 등록돼 있다.
  그대로 뽑으면 같은 사진이 200장의 상당수를 차지한다 → (파일명, bbox) 로 1개만
- **train/valid 교차** — `ingest` 가 현장 라벨을 양쪽에 넣으므로, valid 가 held-out
  이 되도록 train 에 뽑힌 파일명은 valid 에서 제외

⚠️ **이 셋이 400장보다 낫다는 측정은 없다.** 이전에 다른 목적으로 시도한 큐레이션
(합성 원색 제외 + 5분위 층화)은 무작위보다 나빴다(회귀 −0.0066 vs −0.0045).
채택 전에 A/B 가 필요하다.

사용법:
  export S=~/eda_work                     # kraft_X.npy, allfeat_train.pkl 위치
  python3 bench/replay/build_kraft200.py            # 생성
  python3 bench/replay/build_kraft200.py --k-train 300   # 장수 조절
"""
import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import cv2
from pycocotools import mask as maskutil
from sklearn.cluster import KMeans

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'eda'))
sys.path.insert(0, os.path.join(_HERE, '..', 'grouping'))
from gridsearch_feat import NAMES, f3_rich     # noqa: E402
from origins import origin                     # noqa: E402

AX = NAMES['F3_rich']
KRAFT_AXES = ['L', 'b', 'b_iqr', 'tex_hp']     # |코헨 d| >= 1.5 인 축만
KI = [AX.index(n) for n in KRAFT_AXES]
KRAFT_TYPES = ['pacific_long', 'pacific_short', 'swanson', 'plain_small', 'red']
RARE_FIRST = ['Slip Sheet', 'Pallet', 'Tote', 'Shrink Wrap']
MIN_AREA = 40
MAX_CARTON_PER_IMG = 40        # 한 장에 228개까지 있어 상한을 둔다(중앙값엔 충분)

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', default=os.path.expanduser('~/lee3/thirdtask/final_dataset_ultimate'))
ap.add_argument('--out', default=os.path.expanduser('~/depal-al/data/replay_kraft200'))
ap.add_argument('--k-train', type=int, default=200)
ap.add_argument('--k-valid', type=int, default=50)
# 검증된 400장 셋의 클래스 포함 비율(Pallet 34.5% / Slip 22.0% / Tote 41.0% /
# Shrink 26.2%)을 200장으로 내린 값 — 구성은 그대로, 규모만 절반.
ap.add_argument('--floor', default='Slip Sheet:44,Pallet:69,Tote:82,Shrink Wrap:52')
# valid 50 장에 보장을 크게 주면 Carton 이 적은 이미지가 슬롯을 다 먹어 회귀 측정
# 표본(Carton 인스턴스)이 말라버린다 — 20/20/22/15 로 뽑았을 때 212개까지 떨어졌다.
ap.add_argument('--valid-floor', default='Slip Sheet:20,Pallet:14,Tote:14,Shrink Wrap:10')
# valid 의 역할은 **망각 탐지**다. train 과 같은 규칙으로 크래프트만 담으면 우리가
# 의도적으로 버린 영역(궤도렌더 등)의 드리프트를 볼 수 없다 — 경보기가 감시하지 않는
# 구역이 생긴다. 그래서 valid 는 두 층으로 나눈다.
#   kraft 층   : 지키려는 대상 — 여기 점수가 떨어지면 목표 실패
#   control 층 : 버린 영역 포함 — 여기 하락은 예상된 것이나, 얼마나인지는 알아야 한다
# 두 층을 한 mAP 로 뭉치면 서로를 가린다 → valid_slices.json 에 명단을 남겨 따로 채점한다.
ap.add_argument('--valid-far-frac', type=float, default=0.4,
                help='valid 중 크래프트에서 먼 쪽(control 층)으로 채울 비율')
ap.add_argument('--pool-pct', type=float, default=25.0,
                help='크래프트 적격 풀 = 근접 상위 몇 %')
# 클래스 보장 슬롯 중 'Carton 을 동반한 이미지' 로 채울 비율. 1.0 으로 두면 슬롯 효율은
# 최대지만 Carton 없는 출처가 통째로 빠진다 — 실제로 슈링크랩 단품 렌더(render_*)가
# 0% 가 되어 기존 셋의 51% 에서 사라졌다. 그 출처는 베이스 모델이 슈링크랩을 배운 곳이라
# 리플레이에서 빠지면 팔레트 전멸과 같은 종류의 드리프트 위험이 있다.
ap.add_argument('--floor-carton-frac', type=float, default=0.6,
                help='클래스 보장 중 Carton 동반 이미지로 채울 비율(나머지는 그 클래스 '
                     '본래 분포에서)')
ap.add_argument('--seed', type=int, default=20260821)
ap.add_argument('--scan-cache', default=None, help='스캔 결과 재사용(.pkl)')
A = ap.parse_args()
S = os.environ['S']
CACHE = A.scan_cache or os.path.join(S, 'kraft200_scan.pkl')
rng = random.Random(A.seed)


def parse_floor(s):
    out = {}
    for part in s.split(','):
        k, v = part.rsplit(':', 1)
        out[k.strip()] = int(v)
    return out


# ── 1. 크래프트 기준점 (현장 촬영 8종) + 표준화 척도 (Carton 모집단) ────────
XB = np.load(S + '/kraft_X.npy')
LB = np.load(S + '/kraft_L.npy', allow_pickle=True)
kr = np.isin(LB, KRAFT_TYPES)
ANCHOR = np.median(XB[kr][:, KI], axis=0)

allf = pickle.load(open(S + '/allfeat_train.pkl', 'rb'))
acls = np.array([m[0] for m in allf['meta']])
_seen, _keep = set(), np.zeros(len(acls), bool)
for i, m in enumerate(allf['meta']):
    k = (m[2], allf['F3'][i].tobytes())
    if k not in _seen:
        _seen.add(k); _keep[i] = True
POP = allf['F3'][(acls == 'Carton') & _keep][:, KI].astype(np.float64)
MU, SD = POP.mean(0), POP.std(0)
ZA = (ANCHOR - MU) / SD
del allf, _seen
print("크래프트 기준점(현장 %d객체 중앙값): %s" % (
    int(kr.sum()), ' '.join(f"{n}={v:.2f}" for n, v in zip(KRAFT_AXES, ANCHOR))))
print("Carton 모집단 %s 로 표준화 → 기준점 %s" % (
    f"{len(POP):,}", ' '.join(f"{n}={v:+.2f}" for n, v in zip(KRAFT_AXES, ZA))))


# ── 2. 소스 스캔: 파일별 클래스 · 크래프트 좌표 · 거리 ─────────────────────
def decode(a, H, W):
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


def scan(split):
    d = os.path.join(A.src, split)
    coco = json.load(open(os.path.join(d, '_annotations.coco.json')))
    cats = {c['id']: c['name'] for c in coco['categories']}
    id2n = {im['id']: im['file_name'] for im in coco['images']}
    cls_of, ann_of, got, dup = defaultdict(set), defaultdict(list), set(), 0
    for a in coco['annotations']:
        fn = id2n[a['image_id']]
        x, y, w, h = [int(v) for v in a['bbox']]
        if w * h < MIN_AREA:
            continue
        key = (fn, x, y, w, h, a['category_id'])
        if key in got:                     # 현장 40장×50 복제
            dup += 1
            continue
        got.add(key)
        cls_of[fn].add(cats[a['category_id']])
        ann_of[fn].append(a)
    files = sorted(f for f in cls_of if os.path.isfile(os.path.join(d, f)))
    print(f"\n[{split}] 고유 이미지 {len(files):,} (중복 주석 {dup:,} 제거)")

    Zf, dist, org = {}, {}, {}
    t0 = time.time()
    for n, fn in enumerate(files):
        org[fn] = origin(fn)
        carts = [a for a in ann_of[fn] if cats[a['category_id']] == 'Carton']
        if not carts:
            dist[fn] = np.inf; Zf[fn] = None
            continue
        bgr = cv2.imread(os.path.join(d, fn))
        if bgr is None:
            dist[fn] = np.inf; Zf[fn] = None
            continue
        H, W = bgr.shape[:2]
        vs = []
        for a in carts[:MAX_CARTON_PER_IMG]:
            x, y, w, h = [int(v) for v in a['bbox']]
            mk = decode(a, H, W)
            if mk is None:
                continue
            v = f3_rich(bgr[max(0, y):y + h, max(0, x):x + w],
                        mk[max(0, y):y + h, max(0, x):x + w], None)
            if v is not None:
                vs.append([v[i] for i in KI])
        if not vs:
            dist[fn] = np.inf; Zf[fn] = None
            continue
        z = (np.median(np.array(vs, float), axis=0) - MU) / SD
        Zf[fn] = z
        dist[fn] = float(np.linalg.norm(z - ZA))
        if n % 4000 == 0 and n:
            print(f"   {n:,}/{len(files):,}  {time.time()-t0/1:.0f}초", flush=True)
    print(f"   Carton 포함 {sum(1 for f in files if np.isfinite(dist[f])):,}장 · "
          f"{(time.time()-t0)/60:.1f}분")
    return dict(files=files, cls={f: sorted(cls_of[f]) for f in files},
                Z={f: (None if Zf[f] is None else Zf[f].tolist()) for f in files},
                dist=dist, org=org, cats=cats)


def carton_counts(split):
    """파일별 Carton 인스턴스 수. 회귀 측정은 인스턴스 수가 곧 표본 크기다."""
    d = os.path.join(A.src, split)
    coco = json.load(open(os.path.join(d, '_annotations.coco.json')))
    cid = {c['name']: c['id'] for c in coco['categories']}['Carton']
    id2n = {im['id']: im['file_name'] for im in coco['images']}
    got, n = set(), Counter()
    for a in coco['annotations']:
        if a['category_id'] != cid:
            continue
        x, y, w, h = [int(v) for v in a['bbox']]
        if w * h < MIN_AREA:
            continue
        key = (id2n[a['image_id']], x, y, w, h)
        if key in got:
            continue
        got.add(key)
        n[id2n[a['image_id']]] += 1
    return n


if os.path.exists(CACHE):
    SC = pickle.load(open(CACHE, 'rb'))
    print(f"스캔 캐시 사용: {CACHE}")
else:
    SC = {s: scan(s) for s in ('train', 'valid')}
    pickle.dump(SC, open(CACHE, 'wb'))
    print(f"\n스캔 저장: {CACHE}")


# ── 3. 선정 ───────────────────────────────────────────────────────────────
def pick(sc, K, floors, exclude=(), far_frac=0.0, ncar=None):
    """far_frac > 0 이면 그 비율만큼 '크래프트에서 먼' 쪽을 일부러 섞는다(valid 용)."""
    files = [f for f in sc['files'] if f not in exclude]
    cls, dist, Z = sc['cls'], sc['dist'], sc['Z']
    picked, chosen, have = [], set(), Counter()
    far = set()
    NC = ncar or Counter()

    def split_near_far(cand, n):
        """거리 순 앞에서 (1-far_frac), 뒤에서 far_frac 을 뽑아 합친다."""
        # 거리를 0.25 폭으로 묶고 같은 묶음 안에서는 Carton 이 많은 것을 먼저.
        # 회귀 측정에서 인스턴스 수가 곧 표본 크기이기 때문이다.
        cand = sorted(cand, key=lambda f: (round(dist[f] / 0.25), -NC.get(f, 0)))
        if far_frac <= 0 or len(cand) <= 1:
            return cand[:n], []
        nf = min(int(round(n * far_frac)), max(0, len(cand) // 2))
        nn = min(n - nf, len(cand) - nf)
        return cand[:nn], list(reversed(cand))[:nf]

    def take(f):
        # ⚠️ 이중 담기 방어. split_near_far 는 후보가 적으면 앞·뒤 조각이 겹쳐
        # 같은 파일을 두 번 내놓는다(명단 50 중 고유 46 이 되던 원인).
        if f in chosen:
            return False
        picked.append(f); chosen.add(f)
        for c in cls[f]:
            have[c] += 1
        return True

    print("   ── 1단계: 클래스 보장 (희귀 우선) ──")
    frac = A.floor_carton_frac
    for c in RARE_FIRST:
        need = floors.get(c, 0)
        if not need:
            continue
        cand = [f for f in files if c in cls[f] and f not in chosen]
        withc = [f for f in cand if 'Carton' in cls[f]]
        noc = [f for f in cand if 'Carton' not in cls[f]]
        rng.shuffle(noc)              # Carton 이 없으면 크래프트 거리를 잴 수 없다
        # 앞쪽 frac 은 Carton 동반(슬롯 두 몫 + 크래프트), 나머지는 그 클래스 본래 분포
        quota_c = int(round(need * frac))
        near, farc = split_near_far(withc, quota_c)
        far.update(farc)
        plan = list(dict.fromkeys(near + farc + noc[:need - quota_c]))
        # 한쪽이 모자라면 다른 쪽으로 채운다
        if len(plan) < need:
            rest = [f for f in withc + noc if f not in plan]
            plan += rest[:need - len(plan)]
        n0 = len(picked)
        for f in plan:
            if have[c] >= need or len(picked) >= K:
                break
            take(f)
        nc = sum(1 for f in picked[n0:] if 'Carton' in cls[f])
        print(f"      {c:<12}목표 {need:>3}  확보 {have[c]:>3}  "
              f"(Carton 동반 {nc}/{len(picked)-n0})  누적 {len(picked):>3}")

    rest = K - len(picked)
    print(f"   ── 2단계: 남은 {rest}장 = 크래프트 적격 풀에서 k-medoid ──")
    if rest > 0:
        pool = [f for f in files if f not in chosen and np.isfinite(dist[f])]
        if pool:
            thr = float(np.percentile([dist[f] for f in pool], A.pool_pct))
            elig = [f for f in pool if dist[f] <= thr]
            print(f"      적격 풀 {len(elig):,}장 (상위 {A.pool_pct:.0f}%, 거리 ≤ {thr:.2f})")
            n = min(rest, len(elig))
            nf = int(round(n * far_frac))
            # control 층: 적격 풀 밖(크래프트에서 먼 쪽)에서 고르게 — 버린 영역 감시용
            if nf:
                outside = [f for f in pool if dist[f] > thr]
                if outside:
                    Zo = np.array([Z[f] for f in outside])
                    kmo = KMeans(min(nf, len(outside)), n_init=4,
                                 random_state=A.seed).fit(Zo)
                    for k in range(kmo.n_clusters):
                        m = np.where(kmo.labels_ == k)[0]
                        if not len(m):
                            continue
                        j = m[int(np.argmin(np.linalg.norm(
                            Zo[m] - kmo.cluster_centers_[k], axis=1)))]
                        far.add(outside[j]); take(outside[j])
            n = min(K - len(picked), len(elig))
            if n > 0:
                Zs = np.array([Z[f] for f in elig])
                km = KMeans(n, n_init=4, random_state=A.seed).fit(Zs)
                for k in range(n):
                    m = np.where(km.labels_ == k)[0]
                    if not len(m):
                        continue
                    j = m[int(np.argmin(np.linalg.norm(Zs[m] - km.cluster_centers_[k], axis=1)))]
                    take(elig[j])
            for f in sorted(elig, key=lambda x: dist[x]):   # 빈 군집 보충
                if len(picked) >= K:
                    break
                if f not in chosen:
                    take(f)
    return picked, have, far


print(f"\n=== train (K={A.k_train}) ===")
NC_TRAIN = carton_counts('train')
tr, tr_have, _ = pick(SC['train'], A.k_train, parse_floor(A.floor), ncar=NC_TRAIN)
print(f"\n=== valid (K={A.k_valid}) — train 파일명 제외 ===")
NC_VALID = carton_counts('valid')
va, va_have, va_far = pick(SC['valid'], A.k_valid, parse_floor(A.valid_floor),
                           exclude=set(tr), far_frac=A.valid_far_frac, ncar=NC_VALID)


# ── 4. 데이터셋 생성 ──────────────────────────────────────────────────────
def build(split, names):
    src = os.path.join(A.src, split)
    dst = os.path.join(A.out, split)
    os.makedirs(dst, exist_ok=True)
    coco = json.load(open(os.path.join(src, '_annotations.coco.json')))
    cats = coco['categories']
    id2n = {im['id']: im['file_name'] for im in coco['images']}
    want = set(names)
    # 파일명 하나에 images 항목이 여러 개(복제)일 수 있으니 첫 항목만 쓴다
    first, imgs = {}, []
    for im in coco['images']:
        fn = im['file_name']
        if fn in want and fn not in first:
            first[fn] = len(imgs) + 1
            imgs.append({**im, 'id': first[fn]})
    got, anns = set(), []
    for a in coco['annotations']:
        fn = id2n[a['image_id']]
        if fn not in first:
            continue
        x, y, w, h = [int(v) for v in a['bbox']]
        if w * h < MIN_AREA:
            continue
        key = (fn, x, y, w, h, a['category_id'])
        if key in got:
            continue
        got.add(key)
        anns.append({**a, 'id': len(anns) + 1, 'image_id': first[fn]})
    for fn in first:
        shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    json.dump({'images': imgs, 'annotations': anns, 'categories': cats},
              open(os.path.join(dst, '_annotations.coco.json'), 'w'))
    return imgs, anns, cats


os.makedirs(A.out, exist_ok=True)
out = {}
for split, names in (('train', tr), ('valid', va)):
    imgs, anns, cats = build(split, names)
    cname = {c['id']: c['name'] for c in cats}
    per_img = Counter()
    per_inst = Counter(cname[a['category_id']] for a in anns)
    byimg = defaultdict(set)
    for a in anns:
        byimg[a['image_id']].add(cname[a['category_id']])
    for s in byimg.values():
        for c in s:
            per_img[c] += 1
    out[split] = dict(n=len(imgs), labels=len(anns), per_img=dict(per_img),
                      per_inst=dict(per_inst))
    print(f"\n[{split}] {len(imgs)}장 · 라벨 {len(anns):,}")
    for c in ['Pallet', 'Slip Sheet', 'Carton', 'Tote', 'Shrink Wrap']:
        print(f"   {c:<13}이미지 {per_img[c]:>4} ({per_img[c]/max(len(imgs),1)*100:>5.1f}%)"
              f"  인스턴스 {per_inst[c]:>7,}")

# valid 두 층 명단 — 회귀를 한 숫자로 뭉치지 않기 위해
vd = SC['valid']['dist']
slices = dict(
    kraft=sorted(f for f in va if f not in va_far),
    control=sorted(va_far),
    note=('kraft = 지키려는 대상(크래프트 근접). control = 버린 영역 포함(크래프트에서 먼 쪽). '
          '두 층을 한 mAP 로 합치면 서로를 가린다 — 따로 채점할 것. '
          'control 층의 하락은 예상된 것이나 크기는 알아야 한다.'),
    dist={f: (None if not np.isfinite(vd.get(f, np.inf)) else round(vd[f], 3)) for f in va})
json.dump(slices, open(os.path.join(A.out, 'valid_slices.json'), 'w'),
          ensure_ascii=False, indent=1)
kd = [vd[f] for f in slices['kraft'] if np.isfinite(vd.get(f, np.inf))]
cd_ = [vd[f] for f in slices['control'] if np.isfinite(vd.get(f, np.inf))]
print(f"\nvalid 두 층: kraft {len(slices['kraft'])}장"
      f"(거리 중앙 {np.median(kd):.2f})" if kd else "")
print(f"           control {len(slices['control'])}장"
      + (f"(거리 중앙 {np.median(cd_):.2f})" if cd_ else "(Carton 없음)"))

manifest = dict(per_class_min={'train': parse_floor(A.floor),
                               'valid': parse_floor(A.valid_floor)},
                picks={'train': tr, 'valid': va},
                created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                derived_from=A.src, seed=A.seed,
                kraft_axes=KRAFT_AXES,
                kraft_anchor=dict(zip(KRAFT_AXES, [float(v) for v in ANCHOR])),
                scaler=dict(mean=MU.tolist(), std=SD.tolist()),
                pool_pct=A.pool_pct,
                valid_far_frac=A.valid_far_frac,
                valid_slices={'kraft': slices['kraft'], 'control': slices['control']},
                replay_src='(이 데이터셋 자체 — 명단이 함께 배포됨)',
                note=('크래프트 갈색 Carton 망각 방지 목표. 클래스 보장 후 크래프트 '
                      '근접 k-medoid. 파일명 기준 고정 명단. 다시 뽑으려면 '
                      'build_kraft200.py 재실행'))
json.dump(manifest, open(os.path.join(A.out, 'replay_manifest.json'), 'w'),
          ensure_ascii=False, indent=1)
pickle.dump(dict(train=tr, valid=va, out=out,
                 dist={s: {f: SC[s]['dist'][f] for f in (tr if s == 'train' else va)}
                       for s in ('train', 'valid')}),
            open(S + '/kraft200_pick.pkl', 'wb'))
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(A.out) for f in fs) / 1024 / 1024

# ── README 를 스크립트가 직접 쓴다 ────────────────────────────────────────
# 손으로 쓰면 재생성할 때마다 숫자가 낡는다(실제로 그랬다). 방금 계산한 값으로 쓴다.
CL = ['Pallet', 'Slip Sheet', 'Carton', 'Tote', 'Shrink Wrap']
kd = [vd[f] for f in slices['kraft'] if np.isfinite(vd.get(f, np.inf))]
cdd = [vd[f] for f in slices['control'] if np.isfinite(vd.get(f, np.inf))]
trd = [SC['train']['dist'][f] for f in tr
       if np.isfinite(SC['train']['dist'].get(f, np.inf))]


def tbl(split):
    o = out[split]
    return '\n'.join(
        f"| {c} | {o['per_img'].get(c,0)} "
        f"({o['per_img'].get(c,0)/max(o['n'],1)*100:.1f}%) | {o['per_inst'].get(c,0):,} |"
        for c in CL)


open(os.path.join(A.out, 'README.md'), 'w', encoding='utf-8').write(f"""\
# 크래프트 리플레이 셋 `{os.path.basename(A.out)}`

train **{out['train']['n']}**장 + valid **{out['valid']['n']}**장, v11 5클래스.
`replay_fixed_v11`(400+100)의 후속. 용량 **{sz:.0f}MB**.

자동 생성: `bench/replay/build_kraft200.py` (시드 {A.seed},
{time.strftime('%Y-%m-%d %H:%M')}). 손으로 고치지 말 것 — 재생성하면 덮어쓴다.
설계 근거와 한계는 `bench/replay/README.md` 를 볼 것.

## 목표

용량 제약으로 400 → 200. 무작위 200 은 회귀가 기준선(−0.005)을 넘나든다고 이미
측정됐으므로(`bench/replay/ab_replay_pick.md`), 장수를 줄이는 대신 **지키려는 대상을
좁혔다** — 학습셋 전반 대신 **크래프트 갈색 Carton**.

## 크래프트의 정의 (측정)

`~/box_types/` 크래프트 5종 vs 네이비·파랑 3종, 축별 코헨 d 로 골랐다.

| 축 | 코헨 d | 채택 |
|---|---:|---|
| b (황청) | +4.82 | ★ |
| L (밝기) | +4.20 | ★ |
| tex_hp (표면 질감) | −2.30 | ★ |
| b_iqr (색 균일도) | −1.50 | ★ |
| chroma (채도) | −0.07 | ❌ 무용 |

**밝고 · 노란쪽 · 색이 균일하고 · 표면이 매끈한.** "갈색이니 채도가 높다"는 직관은
틀렸다 — 네이비도 저채도다.

기준점 {' '.join(f'{n}={v:.2f}' for n, v in zip(KRAFT_AXES, ANCHOR))}
(Carton 모집단 {len(POP):,} 로 표준화)

## 선정 규칙

1. 이미지마다 그 안의 **Carton 들의 4축 중앙값** → 크래프트 기준점까지 거리
2. **모든 단계의 선택을 이 거리 순으로**
3. 클래스 보존 층 유지(희귀 우선). 없으면 팔레트가 120/120 → 0/120 으로 전멸했던
   2026-08-04 실측이 재현된다
4. 보장의 {A.floor_carton_frac:.0%}는 Carton 동반(슬롯 두 몫), 나머지는 그 클래스 본래
   분포에서 — 100% 를 Carton 동반으로 하면 슈링크랩 단품 렌더가 통째로 빠진다
5. 거리가 비슷하면(0.25 폭 묶음) **Carton 이 많은 이미지 먼저** — 회귀 측정에서
   인스턴스 수가 곧 표본 크기다
6. 남은 슬롯은 크래프트 적격 풀(상위 {A.pool_pct:.0f}%)에서 **k-medoid 로 고르게**

| 항목 | 값 |
|---|---|
| 소스 | `{A.src}` |
| 시드 | {A.seed} |
| 클래스 보장(train) | {', '.join(f'{k} {v}' for k, v in parse_floor(A.floor).items())} |
| 클래스 보장(valid) | {', '.join(f'{k} {v}' for k, v in parse_floor(A.valid_floor).items())} |

## valid 은 두 층이다

train 과 같은 규칙으로 valid 도 크래프트만 담으면 **경보기가 감시하지 않는 구역**이
생긴다 — 의도적으로 버린 궤도렌더 영역의 드리프트를 볼 수 없다. valid 의 역할은
망각 탐지이므로 그건 곤란하다.

| 층 | 장수 | 크래프트 거리 중앙 | 뜻 |
|---|---:|---:|---|
| **kraft** | {len(slices['kraft'])} | {np.median(kd) if kd else float('nan'):.2f} | 지키려는 대상. **여기가 떨어지면 목표 실패** |
| **control** | {len(slices['control'])} | {np.median(cdd) if cdd else float('nan'):.2f} | 버린 영역 포함. 하락은 예상되나 **크기는 알아야 한다** |

명단은 `valid_slices.json`. **두 층을 한 mAP 로 합치면 서로를 가린다** — 따로 채점할 것.

## 구성

train — 크래프트 거리 중앙 **{np.median(trd):.2f}** (기존 400장 셋 1.38 / 무작위 200 1.74)

| 클래스 | 이미지 | 인스턴스 |
|---|---:|---:|
{tbl('train')}

valid

| 클래스 | 이미지 | 인스턴스 |
|---|---:|---:|
{tbl('valid')}

## 걸러낸 것

- **중복 주석 ×50** — 현장 실촬 40장이 COCO `images` 에 각각 50번 등록돼 있다.
  그대로 뽑으면 같은 사진이 200장의 상당수를 차지한다 → `(파일명, bbox)` 로 1개만
- **train/valid 교차** — valid 가 held-out 이 되도록 train 파일명은 valid 에서 제외

검증: `python3 bench/replay/verify_kraft200.py`

## 사용법

```bash
export DEPAL_REPLAY_SRC={A.out}
export DEPAL_REPLAY_MANIFEST={A.out}/replay_manifest.json
dr-onsite-replay
dr-onsite-round
```

⚠️ `config.py` 기본값은 바꾸지 않았다 — A/B 전까지는 명시 지정으로만 쓴다.

## 한계

- ⚠️ **400장보다 낫다는 측정은 없다.** 확인한 것은 "의도한 대로 뽑혔는가" 뿐이다.
  이전에 다른 목적의 큐레이션이 무작위보다 나빴다(회귀 −0.0066 vs −0.0045).
- ⚠️ **valid 를 100 → {out['valid']['n']} 으로 줄였다.** 회귀 측정의 눈금이 바뀌었으므로
  8/18 이전 수치와 직접 비교할 수 없다. Carton 인스턴스도 1,327 →
  {out['valid']['per_inst'].get('Carton', 0):,} 로 줄어 측정이 그만큼 거칠어진다.
- ⚠️ **control 층이 {len(slices['control'])}장뿐**이다. 버린 영역 감시용이지만 이 크기로는
  경보기 역할이 약하다. 궤도렌더 드리프트를 정말 봐야 하면 별도 held-out 이 맞다.
- ⚠️ **valid 의 Slip Sheet 가 3장뿐**이다(기존 셋과 동일). 소스 valid split 자체에
  3장(0.2%)밖에 없다. 슬립시트 회귀는 `dr-onsite-gate` 로 봐야 한다.
- **궤도렌더 `tm_*` 가 거의 0%** 다. 베이스 Carton 데이터의 53%인 최대 출처인데
  크래프트에서 가장 멀어 탈락했다. 의도된 절충이지만 그 영역엔 델타를 붙잡을 것이 없다.
- 이 셋은 여전히 **합성 위주**다. 현장 도메인 성능 판정용이 아니라 **망각 방지용**이다.
""")
print(f"\n생성: {A.out}  ({sz:.0f}MB)  + README.md")
