"""② 대규모(34만) bbox 축소판 → 군집 몽타주 HTML.

⚠️ 34만 객체에는 depth 가 없다. 윗면 워프가 불가하므로 **bbox 크롭을 그대로** CNN 에 넣는
   축소판이다. 정면화·물리크기가 빠지므로 성능은 depth 판보다 낮다. 목적은 정확도 측정이
   아니라 **"학습셋 Carton 이 실제로 품목별 구조를 갖는가"를 눈으로 보는 것**이다.

정답 라벨이 없으므로(학습셋은 'Carton' 하나로만 라벨됨) 순도 같은 지표를 낼 수 없다.
사람이 몽타주를 보고 "한 줄이 같은 박스로 보이는가"를 판단한다.
"""
import os, json, pickle, base64, random, time
from collections import defaultdict, Counter
import numpy as np, cv2, torch
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
ROOT = os.path.expanduser('~/lee3/thirdtask/final_dataset_ultimate/train')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TARGET = int(os.environ.get('N_TARGET', '20000'))
PER_IMG = int(os.environ.get('PER_IMG', '3'))
OUT = os.path.expanduser('~/그룹화_검증_대규모.html')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
rng = random.Random(0)


def prep(patch, size=160):
    """bbox 크롭 → 고정 정사각(배치 처리용). depth 가 없어 종횡비 보존 워프가 불가."""
    im = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


coco = json.load(open(os.path.join(ROOT, '_annotations.coco.json')))
cats = {c['id']: c['name'] for c in coco['categories']}
CID = [k for k, v in cats.items() if v == 'Carton'][0]
imgs = {im['id']: im for im in coco['images']}
by = defaultdict(list)
for a in coco['annotations']:
    if a['category_id'] == CID:
        by[a['image_id']].append(a)
ids = list(by)
rng.shuffle(ids)
print(f"Carton 포함 이미지 {len(ids):,}장에서 표본 {N_TARGET:,} 목표", flush=True)


def origin(fn):
    f = fn.lower()
    if f.startswith('sam_hard'): return '현장(Thor)'
    if f.startswith('real_'):    return '실촬(기타)'
    if f.startswith('das_'):     return '실촬(DAS)'
    if f.startswith('tm_'):      return '합성(궤도렌더)'
    if f.startswith('fg_'):      return '합성(현장갭)'
    if f.startswith('slipcp'):   return '슬립시트증강'
    if 'synth' in f:             return '합성(기타)'
    if '.rf.' in f:              return '외부(roboflow)'
    return '기타'


import torchvision.models as tvm
model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Identity(); model.eval().to(DEV)

CROPS, META, EMB = [], [], []
buf_t, buf_c, buf_m = [], [], []
t0 = time.time()


@torch.no_grad()
def flush():
    global buf_t, buf_c, buf_m
    if not buf_t:
        return
    xs = torch.stack(buf_t).to(DEV)
    v = model(xs).float().cpu().numpy()
    EMB.append(v)
    CROPS.extend(buf_c); META.extend(buf_m)
    buf_t, buf_c, buf_m = [], [], []


done = 0
for k, iid in enumerate(ids):
    if done >= N_TARGET:
        break
    im = imgs[iid]
    p = os.path.join(ROOT, im['file_name'])
    if not os.path.isfile(p):
        continue
    bgr = cv2.imread(p)
    if bgr is None:
        continue
    anns = by[iid]
    rng.shuffle(anns)
    for a in anns[:PER_IMG]:
        x, y, w, h = [int(v) for v in a['bbox']]
        if w * h < 900:                       # 너무 작으면 CNN 입력으로 무의미
            continue
        c = bgr[max(0, y):y + h, max(0, x):x + w]
        if c.size == 0 or min(c.shape[:2]) < 12:
            continue
        buf_t.append(prep(c))
        buf_c.append(cv2.resize(c, (96, 96)))
        buf_m.append((im['file_name'], origin(im['file_name']), w, h))
        done += 1
        if len(buf_t) >= 128:
            flush()
        if done >= N_TARGET:
            break
    if k % 2000 == 0 and k:
        el = time.time() - t0
        print(f"  {k:,}장  객체 {done:,}  {el/60:.1f}분", flush=True)
flush()
E = np.concatenate(EMB)
print(f"\n임베딩 {E.shape}  ({(time.time()-t0)/60:.1f}분)")
print("  출처:", dict(Counter(m[1] for m in META).most_common()))

Ep = PCA(16, random_state=0).fit_transform(E)
Ep = Ep / max(Ep.std(), 1e-9)
# 대규모라 MiniBatchKMeans 로 1차 분할 후, 군집 중심을 다시 병합해 개수를 정리
K1 = int(os.environ.get('K1', '60'))
lab1 = MiniBatchKMeans(K1, n_init=5, random_state=0, batch_size=4096).fit_predict(Ep)
cent = np.stack([Ep[lab1 == g].mean(0) for g in range(K1)])
K2 = int(os.environ.get('K2', '24'))
merge = AgglomerativeClustering(n_clusters=K2, linkage='ward').fit_predict(cent)
LABC = merge[lab1]
print(f"군집: 1차 {K1} → 병합 {K2}  크기 상위 {sorted(Counter(LABC).values(), reverse=True)[:8]}")


def b64(im):
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode() if ok else ''


rows = []
for gi, cnt in Counter(LABC).most_common():
    idx = np.where(LABC == gi)[0]
    pick = idx[:28] if len(idx) <= 28 else np.array(rng.sample(list(idx), 28))
    cells = ''.join(f'<img src="data:image/jpeg;base64,{b64(CROPS[i])}" '
                    f'title="{META[i][1]} {META[i][2]}x{META[i][3]}px">' for i in pick)
    org = ' · '.join(f'{t}:{n}' for t, n in Counter(META[i][1] for i in idx).most_common(3))
    rows.append(f'''<div class="grp"><div class="hd"><b>군집 {gi}</b>
<span class="n">{len(idx):,}개</span><span class="mx">{org}</span></div>
<div class="imgs">{cells}</div></div>''')

CSS = """body{font-family:-apple-system,'Malgun Gothic',sans-serif;margin:16px;background:#fafbfc;color:#172b4d}
h2{margin:6px 0}.meta{color:#5e6c84;font-size:13px;margin-bottom:14px;line-height:1.7}
.grp{background:#fff;border:1px solid #dfe1e6;border-radius:6px;margin-bottom:10px;padding:8px 10px}
.hd{font-size:13px;margin-bottom:6px}.n{color:#5e6c84;margin-left:6px}
.mx{color:#7a869a;margin-left:10px;font-size:12px}
.imgs{display:flex;flex-wrap:wrap;gap:3px}.imgs img{width:96px;height:96px;object-fit:cover;border:1px solid #ebecf0;border-radius:3px}"""
doc = f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>그룹화 검증 — 대규모(학습셋 Carton)</title><style>{CSS}</style></head><body>
<h2>그룹화 검증 ② — 학습셋 Carton 대규모 표본</h2>
<div class="meta">
객체 <b>{len(CROPS):,}</b>개 표본 (전체 339,844 중) · 군집 <b>{K2}</b>개 ·
<b>정답 라벨이 없는 비지도 결과</b>입니다(학습셋은 'Carton' 하나로만 라벨됨).<br>
⚠️ 이 자료에는 <b>depth 가 없어</b> 윗면 워프·물리크기를 쓸 수 없습니다. bbox 크롭을 그대로
ResNet50 에 넣은 <b>축소판</b>이므로 운용 알고리즘보다 성능이 낮습니다.
목적은 정확도 측정이 아니라 <b>학습셋 Carton 이 실제로 품목별 구조를 갖는지 눈으로 확인</b>하는 것입니다.<br>
<b>보는 법</b>: 한 줄이 한 군집입니다. 줄 안의 박스들이 서로 비슷해 보이면 성공,
전혀 다른 것들이 섞여 있으면 실패입니다. 이미지에 커서를 올리면 출처·픽셀크기가 보입니다.
</div>
{''.join(rows)}
</body></html>"""
open(OUT, 'w', encoding='utf-8').write(doc)
print("생성:", OUT, f"({os.path.getsize(OUT)/1024/1024:.1f}MB)")
