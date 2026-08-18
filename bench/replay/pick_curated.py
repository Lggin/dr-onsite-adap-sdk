#!/usr/bin/env python3
"""리플레이 '큐레이션' 선정 — 이상치 제외 + 분위 층화 + 클래스 최소 보장.

v12(replay_fixed_v12) 에 쓴 선정 방식을 벤치에서도 쓸 수 있게 분리한 것.
기존 `stratified_pick`(무작위 층화)과 **같은 소스·같은 K** 로 비교하기 위함이다.

기존과의 차이
  · 이미지마다 '이상치 점수'(객체 색·질감이 학습셋 군집 중심에서 얼마나 먼가)를 매긴다
  · 극단(THR 초과)만 버린다 — 세게 자르면 현장에 실제로 있는 박스까지 잃는다
    (현장 실촬 6종의 중심거리가 최대 3.89 였다)
  · 남은 범위를 **분위로 층화**해 뽑는다. "중심에 가까운 순"으로만 뽑으면 전체의 하위
    12% 에 갇혀 베이스가 배운 것 전반을 대표하지 못한다(2026-08-18 실측)
"""
import os
import pickle
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

_CACHE = {}


def _feats(bgr):
    if bgr.size == 0 or min(bgr.shape[:2]) < 8:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = [float(np.median(lab[..., i])) for i in range(3)]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gl = np.log1p(g.astype(np.float32))
    return [L, a, b, float((gl - cv2.GaussianBlur(gl, (0, 0), 3.0)).std()),
            float(cv2.Canny(g, 40, 120).mean())]


def score_images(coco, img_dir, model_pkl, max_obj=6, seed=0, cache_key=None):
    """이미지별 이상치 점수 {image_id: d_med}. 무거우므로 캐시한다."""
    if cache_key and cache_key in _CACHE:
        return _CACHE[cache_key]
    cache_f = f"/tmp/pick_curated_{cache_key}.pkl" if cache_key else None
    if cache_f and os.path.isfile(cache_f):
        d = pickle.load(open(cache_f, 'rb'))
        _CACHE[cache_key] = d
        return d
    m = pickle.load(open(model_pkl, 'rb'))
    sc, KM = m['sc'], m['KM']
    rng = random.Random(seed)
    by = defaultdict(list)
    for a in coco['annotations']:
        by[a['image_id']].append(a)
    out = {}
    for im in coco['images']:
        p = os.path.join(img_dir, im['file_name'])
        anns = by.get(im['id'], [])
        if not anns or not os.path.isfile(p):
            continue
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        ds = []
        for a in (anns if len(anns) <= max_obj else rng.sample(anns, max_obj)):
            x, y, w, h = [int(v) for v in a['bbox']]
            if w * h < 400:
                continue
            f = _feats(bgr[max(0, y):y + h, max(0, x):x + w])
            if f is None:
                continue
            z = sc.transform([f])
            ds.append(float(np.linalg.norm(z - KM.cluster_centers_, axis=1).min()))
        if ds:
            out[im['id']] = float(np.median(ds))
    if cache_f:
        pickle.dump(out, open(cache_f, 'wb'))
    _CACHE[cache_key] = out
    return out


def curated_pick(coco, scores, K, per_class_min, rng, thr=4.0, bins=5):
    """이상치 제외 → 클래스 최소 보장(분위 층화) → 남는 자리도 분위 층화."""
    cats = {c['id']: c['name'] for c in coco['categories']}
    cls_of = defaultdict(set)
    for a in coco['annotations']:
        cls_of[a['image_id']].add(cats[a['category_id']])
    pool = [i for i, d in scores.items() if d <= thr]
    if len(pool) < K:
        pool = sorted(scores, key=lambda i: scores[i])[:max(K * 3, 600)]
    by_cls = defaultdict(list)
    for i in pool:
        for c in cls_of.get(i, ()):
            by_cls[c].append(i)
    sel, seen = [], set()
    for c in sorted(by_cls, key=lambda c: len(by_cls[c])):
        cand = sorted(by_cls[c], key=lambda i: scores[i])
        want = min(per_class_min, len(cand))
        per = max(1, want // bins)
        got = 0
        for ch in np.array_split(np.arange(len(cand)), bins):
            idxs = list(ch)
            rng.shuffle(idxs)
            for j in idxs[:per]:
                if got >= want or len(sel) >= K:
                    break
                i = cand[j]
                if i in seen:
                    continue
                sel.append(i); seen.add(i); got += 1
    rest = sorted([i for i in pool if i not in seen], key=lambda i: scores[i])
    chunks = np.array_split(np.arange(len(rest)), bins)
    while len(sel) < K:
        added = False
        for ch in chunks:
            if len(sel) >= K:
                break
            avail = [i for i in ch if rest[i] not in seen]
            if not avail:
                continue
            j = rng.choice(avail)
            sel.append(rest[j]); seen.add(rest[j]); added = True
        if not added:
            break
    return sel
