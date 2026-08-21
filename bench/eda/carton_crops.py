"""Carton 표본의 특징 + 크롭 썸네일 — 산점도에서 점을 클릭하면 볼 객체.

산점도가 점만 보여주면 쓸모가 없다. "이 점이 무슨 박스인가"를 눌러서 봐야 군집이
맞는지 눈으로 판정할 수 있다. 그래서 표본마다 특징과 작은 JPEG 를 같이 만든다.

⚠️ **allfeat_train.pkl 의 행 번호로 크롭을 찾으려 하지 말 것.** `extract_all.py` 는
세그먼테이션 디코딩 실패·f3_rich 실패 객체를 조용히 건너뛰므로(379,190 → 377,914)
COCO 를 다시 훑어 번호를 복원하면 어긋난다. 여기서는 **COCO 주석에서 직접** 표본을
뽑고 특징도 그 자리에서 계산해 자기완결이 되게 한다.

표본 추출은 **출처별 √n 비례 + 최소 하한**으로 한다.
  · 비례로만 뽑으면 현장(Thor) 이 Carton 의 0.02%(59개)라 점이 안 보인다
  · 균등으로 뽑으면 합성이 21만인 실제 편중이 사라진다
  √n 은 그 사이의 타협이며, 각 출처의 실제 개수를 HTML 에 함께 적는다.

출력: $S/carton_sample.pkl — 특징(14) · 메타 · 크롭 JPEG(bytes)
"""
import json, os, pickle, random, sys, time
from collections import Counter, defaultdict
import numpy as np
import cv2
from pycocotools import mask as maskutil

sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))   # gridsearch_feat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridsearch_feat import f3_rich
from origins import origin

S = os.environ['S']
ROOT = os.path.expanduser('~/lee3/thirdtask/final_dataset_ultimate/train')
OUT = S + '/carton_sample.pkl'
N = int(os.environ.get('N_SAMPLE', '5000'))
FLOOR = int(os.environ.get('FLOOR', '200'))
THUMB = int(os.environ.get('THUMB', '92'))      # 긴 변 픽셀
QUAL = int(os.environ.get('QUAL', '74'))
MIN_AREA = 40
rng = random.Random(0)

coco = json.load(open(os.path.join(ROOT, '_annotations.coco.json')))
cid = {c['name']: c['id'] for c in coco['categories']}['Carton']
id2name = {im['id']: im['file_name'] for im in coco['images']}
print(f"클래스 id: Carton={cid}")

# ── Carton 주석 전체 목록 (파일명 → 출처) ────────────────────────────────
# ⚠️ 현장 40장은 COCO images 에 각각 50번 등록돼 있다(2,000 항목 = 40 고유).
#    (파일명, bbox) 로 중복을 걷어내지 않으면 현장 점 하나가 50개로 보인다.
allc, seen = [], set()
dupes = 0
for a in coco['annotations']:
    if a['category_id'] != cid:
        continue
    x, y, w, h = [int(v) for v in a['bbox']]
    if w * h < MIN_AREA:
        continue
    fn = id2name[a['image_id']]
    key = (fn, x, y, w, h)
    if key in seen:
        dupes += 1
        continue
    seen.add(key)
    allc.append((origin(fn), fn, a))
cnt = Counter(o for o, _, _ in allc)
print(f"Carton 주석 {len(allc):,} (중복 등록 {dupes:,}개 제거)")

# ── 출처별 √n 비례 배분 ──────────────────────────────────────────────────
wt = {o: np.sqrt(n) for o, n in cnt.items()}
tot = sum(wt.values())
quota = {o: max(FLOOR, int(round(N * wt[o] / tot))) for o in wt}
over = sum(quota.values()) - N
big = sorted(quota, key=lambda o: -quota[o])
i = 0
while over > 0 and any(quota[o] > FLOOR for o in quota):
    o = big[i % len(big)]
    if quota[o] > FLOOR:
        quota[o] -= 1; over -= 1
    i += 1

byorg = defaultdict(list)
for rec in allc:
    byorg[rec[0]].append(rec)
print(f"\n{'출처':<16}{'전체':>10}{'표본':>8}{'비율':>8}")
pick = []
for o in sorted(cnt, key=lambda x: -cnt[x]):
    k = min(quota[o], len(byorg[o]))
    pick.extend(rng.sample(byorg[o], k))
    print(f"{o:<16}{cnt[o]:>10,}{k:>8,}{k / cnt[o] * 100:>7.1f}%")
print(f"{'합':<16}{len(allc):>10,}{len(pick):>8,}")

# ── 이미지별로 묶어 한 번만 읽는다 ───────────────────────────────────────
byfile = defaultdict(list)
for o, fn, a in pick:
    byfile[fn].append((o, a))
print(f"\n이미지 {len(byfile):,}장 · 크롭 {len(pick):,}개 추출")

F, MET, ORG, CROP = [], [], [], []
t0, skipped = time.time(), Counter()
for k, (fn, lst) in enumerate(sorted(byfile.items())):
    p = os.path.join(ROOT, fn)
    bgr = cv2.imread(p)
    if bgr is None:
        skipped['이미지 없음'] += len(lst); continue
    H, W = bgr.shape[:2]
    for o, a in lst:
        x, y, bw, bh = [int(v) for v in a['bbox']]
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
            skipped['마스크 실패'] += 1; continue
        crop = bgr[max(0, y):y + bh, max(0, x):x + bw]
        mk = m[max(0, y):y + bh, max(0, x):x + bw]
        v = f3_rich(crop, mk, None)        # 전수 추출과 같은 규격(크기=픽셀)
        if v is None:
            skipped['특징 실패'] += 1; continue

        # 썸네일은 맥락이 보이게 약간 넓게 자른다
        pad = max(3, int(0.06 * max(bw, bh)))
        c = bgr[max(0, y - pad):min(H, y + bh + pad),
                max(0, x - pad):min(W, x + bw + pad)]
        if c.size == 0:
            skipped['크롭 빈값'] += 1; continue
        s = THUMB / max(c.shape[:2])
        if s < 1:
            c = cv2.resize(c, (max(1, round(c.shape[1] * s)),
                               max(1, round(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, QUAL])
        if not ok:
            skipped['인코딩 실패'] += 1; continue

        F.append(v); ORG.append(o); CROP.append(buf.tobytes())
        MET.append((fn, int(bw), int(bh), float(a.get('area', bw * bh))))
    if k % 500 == 0 and k:
        print(f"  {len(F):,}/{len(pick):,}  {time.time() - t0:.0f}초", flush=True)

F = np.array(F, np.float32)
sz = sum(len(c) for c in CROP)
print(f"\n완료 {len(F):,}개 · 크롭 합 {sz / 1024 / 1024:.1f}MB "
      f"(평균 {sz / max(len(F), 1) / 1024:.1f}KB) · {(time.time() - t0) / 60:.1f}분")
if skipped:
    print("건너뜀:", dict(skipped))

pickle.dump(dict(F3=F, meta=MET, org=np.array(ORG), crops=CROP,
                 counts=dict(cnt), thumb=THUMB, total=len(allc)),
            open(OUT, 'wb'))
print("저장:", OUT, f"({os.path.getsize(OUT) / 1024 / 1024:.1f}MB)")
