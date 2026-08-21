"""전수 마스크 피처 추출 — ultimate train 379,190 라벨.

정답(품목 라벨)이 없으므로 군집 구조의 **안정성**으로 간접 채점한다.
피처는 grid 실험과 같은 F2(마스크 5차원) / F3(마스크 확장 14차원)을 함께 뽑는다.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np, cv2
from pycocotools import mask as maskutil

sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))   # gridsearch_feat 는 그룹화 벤치에 있다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # origins
from gridsearch_feat import f2_mask, f3_rich

ROOT = os.path.expanduser('~/lee3/thirdtask/final_dataset_ultimate')
SPLIT = os.environ.get('SPLIT', 'train')
OUT = os.path.join(os.environ['S'], f'allfeat_{SPLIT}.pkl')
MIN_AREA = 40


def origin(fn):
    f = fn.lower()
    if f.startswith('sam_hard'):
        return '현장(Thor)'
    if f.startswith('real_'):
        return '실촬(기타)'
    if f.startswith('das_'):
        return '실촬(DAS)'
    if f.startswith('tm_'):
        return '합성(궤도렌더)'
    if f.startswith('fg_'):
        return '합성(현장갭)'
    if f.startswith('slipcp'):
        return '슬립시트증강'
    if 'synth' in f:
        return '합성(기타)'
    if '.rf.' in f or 'roboflow' in f:
        return '외부(roboflow)'
    return '기타'


d = os.path.join(ROOT, SPLIT)
coco = json.load(open(os.path.join(d, '_annotations.coco.json')))
cats = {c['id']: c['name'] for c in coco['categories']}
imgs = {im['id']: im for im in coco['images']}
by = defaultdict(list)
for a in coco['annotations']:
    by[a['image_id']].append(a)

F2, F3, META = [], [], []
t0 = time.time()
done = 0
for k, iid in enumerate(sorted(by)):
    im = imgs[iid]
    p = os.path.join(d, im['file_name'])
    if not os.path.isfile(p):
        continue
    bgr = cv2.imread(p)
    if bgr is None:
        continue
    H, W = bgr.shape[:2]
    for a in by[iid]:
        x, y, w, h = [int(v) for v in a['bbox']]
        if w * h < MIN_AREA:
            continue
        seg = a.get('segmentation')
        try:
            if isinstance(seg, dict):
                r = (maskutil.frPyObjects(seg, H, W)
                     if isinstance(seg.get('counts'), list) else seg)
                m = maskutil.decode(r)
                m = m[..., 0] if m.ndim == 3 else m
            else:
                m = np.zeros((H, W), np.uint8)
                for poly in seg or []:
                    if len(poly) >= 6:
                        cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
        except Exception:
            continue
        crop = bgr[max(0, y):y + h, max(0, x):x + w]
        mk = m[max(0, y):y + h, max(0, x):x + w]
        v2 = f2_mask(crop, mk, None)
        v3 = f3_rich(crop, mk, None)
        if v2 is None or v3 is None:
            continue
        F2.append(v2); F3.append(v3)
        META.append((cats.get(a['category_id'], '?'), origin(im['file_name']),
                     im['file_name'], int(w), int(h)))
        done += 1
    if k % 2000 == 0 and k:
        el = time.time() - t0
        print(f"  {k:,}/{len(by):,}장  객체 {done:,}  {el/60:.1f}분 "
              f"(ETA {el/k*(len(by)-k)/60:.0f}분)", flush=True)

F2 = np.array(F2, np.float32); F3 = np.array(F3, np.float32)
print(f"\n완료: 객체 {len(F2):,}  F2{F2.shape} F3{F3.shape}  {(time.time()-t0)/60:.1f}분")
from collections import Counter
print("  클래스:", dict(Counter(m[0] for m in META)))
print("  출처:", dict(Counter(m[1] for m in META).most_common()))
pickle.dump(dict(F2=F2, F3=F3, meta=META), open(OUT, 'wb'))
print("저장:", OUT, f"({os.path.getsize(OUT)/1024**2:.0f}MB)")
