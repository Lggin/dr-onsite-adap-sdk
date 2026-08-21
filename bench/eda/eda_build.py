"""EDA용 객체 특징 추출 — ultimate(학습셋) + box_types(정답 6종) 를 같은 공간에.

목적: "같은 Carton 이라도 색·질감·크기가 어떻게 다른가"
  · ultimate 은 품목 라벨이 없다(Carton 하나로 뭉뚱그림) → 군집으로 **숨은 구조**를 찾는다
  · box_types 는 정답 6종이 있다 → 같은 공간에 겹쳐 "학습셋이 현장 박스를 덮는가"를 본다

특징은 그룹화 파이프라인과 같은 규약(Lab 중앙값 + 로그HP/Canny 질감)을 쓴다.
크기는 depth 가 있는 box_types 만 물리 mm, 나머지는 픽셀(비교 시 주의).
"""
import json, os, glob, pickle, random, sys
import numpy as np, cv2

ROOT = os.path.expanduser('~/lee3/thirdtask')
S = os.environ['S']
N_SAMPLE = int(os.environ.get('N_SAMPLE', '6000'))   # ultimate 에서 뽑을 객체 수
MIN_PX = 400
rng = random.Random(0)


def feats_from_crop(bgr):
    """색 3 + 질감 2. 그룹화 파이프라인과 같은 취지(Lab 중앙값, 로그 하이패스, Canny)."""
    if bgr.size == 0 or min(bgr.shape[:2]) < 8:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = [float(np.median(lab[..., i])) for i in range(3)]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gl = np.log1p(g.astype(np.float32))
    hp = gl - cv2.GaussianBlur(gl, (0, 0), 3.0)
    tex1 = float(hp.std())
    tex2 = float(cv2.Canny(g, 40, 120).mean())
    return [L, a, b, tex1, tex2]


rows = []

# ── ultimate: Carton 객체 표본 ────────────────────────────────────────────
d = os.path.join(ROOT, 'final_dataset_ultimate', 'train')
coco = json.load(open(os.path.join(d, '_annotations.coco.json')))
cats = {c['id']: c['name'] for c in coco['categories']}
imgs = {im['id']: im for im in coco['images']}
by_img = {}
for a in coco['annotations']:
    by_img.setdefault(a['image_id'], []).append(a)

# 이미지 단위로 표본을 뽑아 디스크 읽기를 줄인다(한 장에서 여러 객체를 얻는다)
img_ids = [i for i in by_img if any(cats.get(x['category_id']) == 'Carton' for x in by_img[i])]
rng.shuffle(img_ids)
print(f"ultimate: Carton 포함 이미지 {len(img_ids):,}장 중에서 표본 추출", flush=True)

got = 0
for iid in img_ids:
    if got >= N_SAMPLE:
        break
    im = imgs[iid]
    p = os.path.join(d, im['file_name'])
    if not os.path.isfile(p):
        continue
    bgr = cv2.imread(p)
    if bgr is None:
        continue
    H, W = bgr.shape[:2]
    anns = [x for x in by_img[iid] if cats.get(x['category_id']) == 'Carton']
    rng.shuffle(anns)
    for a in anns[:4]:                      # 한 장에서 최대 4개
        x, y, w, h = [int(v) for v in a['bbox']]
        if w * h < MIN_PX:
            continue
        crop = bgr[max(0, y):y + h, max(0, x):x + w]
        f = feats_from_crop(crop)
        if f is None:
            continue
        rows.append(dict(src='ultimate', label='Carton', file=p,
                         bbox=[x, y, w, h], feats=f,
                         px_long=max(w, h), px_short=min(w, h),
                         mm_long=None, mm_short=None,
                         origin=('현장실촬' if im['file_name'].startswith('sam_hard')
                                 else ('합성' if 'synth' in im['file_name'].lower()
                                       or im['file_name'].startswith('tm_')
                                       else '외부/기타'))))
        got += 1
        if got >= N_SAMPLE:
            break
    if got % 1000 < 4 and got:
        print(f"  {got:,}/{N_SAMPLE:,}", flush=True)

# ── box_types: 정답 6종 (물리 mm 포함) ────────────────────────────────────
bt = pickle.load(open(S + '/box_objs.pkl', 'rb'))
for r in bt:
    if r['cid'] != 2:
        continue
    bgr = cv2.imread(r['file'])
    if bgr is None:
        continue
    x1, y1, x2, y2 = [int(v) for v in r['bbox']]
    f = feats_from_crop(bgr[max(0, y1):y2, max(0, x1):x2])
    if f is None:
        continue
    t = r['type'][:-7] if r['type'].endswith('_single') else r['type']
    rows.append(dict(src='box_types', label=t, file=r['file'],
                     bbox=[x1, y1, x2 - x1, y2 - y1], feats=f,
                     px_long=max(x2 - x1, y2 - y1), px_short=min(x2 - x1, y2 - y1),
                     mm_long=(r['size_mm'][0] if r['size_mm'] else None),
                     mm_short=(r['size_mm'][1] if r['size_mm'] else None),
                     origin='현장실촬'))

from collections import Counter
print(f"\n총 객체 {len(rows):,}")
print("  출처:", dict(Counter(r['src'] for r in rows)))
print("  box_types 종류별:", dict(Counter(r['label'] for r in rows if r['src'] == 'box_types')))
print("  ultimate 원본구성:", dict(Counter(r['origin'] for r in rows if r['src'] == 'ultimate')))
pickle.dump(rows, open(S + '/eda_objs.pkl', 'wb'))
print("저장:", S + '/eda_objs.pkl')
