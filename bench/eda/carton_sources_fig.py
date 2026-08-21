"""Carton 출처별 성격 비교 그림 — 숫자로만 말하면 안 믿기는 것.

"도메인 간극이 있다"는 표만 보여주면 추상적이다. 네 출처의 대표 프레임과 크롭을
나란히 놓으면 왜 갈리는지 한눈에 보인다.

  위: 대표 프레임 (Carton 마스크 외곽선)
  아래: 같은 출처의 크롭 6개 + 밝기 L
"""
import json, os, pickle, random, sys
from collections import defaultdict
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pycocotools import mask as maskutil

sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from origins import origin

_TTC = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_TTC):
    font_manager.fontManager.addfont(_TTC)
for cand in ('Noto Sans CJK JP', 'Noto Sans CJK KR', 'NanumGothic'):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = cand
        break
plt.rcParams['axes.unicode_minus'] = False

S = os.environ['S']
ROOT = os.path.expanduser('~/lee3/thirdtask/final_dataset_ultimate/train')
rng = random.Random(1)

coco = json.load(open(os.path.join(ROOT, '_annotations.coco.json')))
CID = {c['name']: c['id'] for c in coco['categories']}['Carton']
id2n = {im['id']: im['file_name'] for im in coco['images']}
ann = defaultdict(list)
byorg = defaultdict(set)
cnt = defaultdict(int)
for a in coco['annotations']:
    fn = id2n[a['image_id']]
    ann[fn].append(a)
    if a['category_id'] == CID:
        byorg[origin(fn)].add(fn)
        cnt[origin(fn)] += 1

# 프레임을 손으로 고르지 않는다 — 출처별 Carton 수가 중앙값인 프레임을 뽑는다
SRC = ['합성(궤도렌더)', '합성(현장갭)', '외부(roboflow)', '현장(Thor)']
NOTE = {'합성(궤도렌더)': '빈 배경 · 무작위 사진 질감 · 카메라 멀다',
        '합성(현장갭)': '밀집 적재 · 크래프트 질감 · 하향 시점',
        '외부(roboflow)': '근접 촬영 · 레터박스 여백 · 워터마크',
        '현장(Thor)': '실제 창고 · 30장뿐 · 장당 98개'}


def pick_typical(o):
    fs = sorted(byorg[o])
    k = [(sum(1 for a in ann[f] if a['category_id'] == CID), f) for f in fs]
    k.sort()
    return k[len(k) // 2][1]


def outline(fn):
    im = cv2.imread(os.path.join(ROOT, fn))
    H, W = im.shape[:2]
    n = 0
    for a in ann[fn]:
        if a['category_id'] != CID:
            continue
        n += 1
        seg = a.get('segmentation')
        try:
            if isinstance(seg, dict):
                r = (maskutil.frPyObjects(seg, H, W)
                     if isinstance(seg.get('counts'), list) else seg)
                m = maskutil.decode(r)
                m = m[..., 0] if m.ndim == 3 else m
            else:
                m = np.zeros((H, W), np.uint8)
                for p in seg or []:
                    if len(p) >= 6:
                        cv2.fillPoly(m, [np.array(p, np.int32).reshape(-1, 2)], 1)
        except Exception:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (40, 230, 255), max(1, int(min(H, W) / 320)))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), n


B = pickle.load(open(S + '/carton_sample.pkl', 'rb'))
ORG, CROP, XS = B['org'], B['crops'], B['F3']

fig = plt.figure(figsize=(16, 8.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.62, 1], hspace=.30, wspace=.10,
                      left=.012, right=.988, top=.855, bottom=.035)

for c, o in enumerate(SRC):
    fn = pick_typical(o)
    img, n = outline(fn)
    ax = fig.add_subplot(gs[0, c])
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#bbb')
    ax.set_title(f"{o}\nCarton {cnt[o]:,}개 · 이 장에 {n}개 · 장당 평균 "
                 f"{cnt[o]/len(byorg[o]):.0f}개\n{NOTE[o]}",
                 fontsize=10.5, loc='left', linespacing=1.6)

    # 아래: 크롭 6개를 한 줄로 이어 붙인다
    idx = [i for i in range(len(ORG)) if ORG[i] == o]
    take = rng.sample(idx, min(6, len(idx)))
    tiles, labs = [], []
    for i in take:
        cr = cv2.imdecode(np.frombuffer(CROP[i], np.uint8), cv2.IMREAD_COLOR)
        cr = cv2.resize(cr, (110, 110), interpolation=cv2.INTER_AREA)
        tiles.append(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))
        labs.append(f"L{XS[i][0]:.0f}")
    strip = np.hstack([np.pad(t, ((0, 0), (2, 2), (0, 0)), constant_values=255)
                       for t in tiles])
    ax2 = fig.add_subplot(gs[1, c])
    ax2.imshow(strip); ax2.set_yticks([])
    ax2.set_xticks([114 * i + 57 for i in range(len(tiles))], labs, fontsize=8.5)
    ax2.tick_params(length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    Ls = XS[np.array(idx)][:, 0]
    ax2.set_xlabel(f"밝기 L 중앙 {np.median(Ls):.0f} "
                   f"(10~90%p {np.percentile(Ls,10):.0f}~{np.percentile(Ls,90):.0f})",
                   fontsize=9.5)

fig.suptitle('Carton 은 어디서 왔나 — 출처별 성격이 다르다\n'
             '위: 대표 프레임(Carton 마스크 외곽선) · 아래: 같은 출처 크롭과 밝기',
             fontsize=13.5, x=.012, ha='left', y=.985, linespacing=1.5)
out = os.path.expanduser('~/carton_출처비교.png')
fig.savefig(out, dpi=118, bbox_inches='tight', facecolor='white')
print("생성:", out, f"({os.path.getsize(out)/1024:.0f}KB)")
for o in SRC:
    print(f"  {o:<16}Carton {cnt[o]:>8,}  이미지 {len(byorg[o]):>6,}  "
          f"장당 {cnt[o]/len(byorg[o]):>5.1f}")
