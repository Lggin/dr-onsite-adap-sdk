"""① 이상치 크롭 확인  ② 크기·종횡비·밀집도 분포."""
import os, json, pickle, glob
import numpy as np, cv2
from collections import Counter, defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FP)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FP).get_name()
plt.rcParams['axes.unicode_minus'] = False
S = os.environ['S']
OUT = os.path.expanduser('~/eda_이미지')
ROOT = os.path.expanduser('~/lee3/thirdtask')

rows = pickle.load(open(S + '/eda_objs.pkl', 'rb'))
m = pickle.load(open(S + '/eda_model.pkl', 'rb'))
Z = m['sc'].transform(np.array([r['feats'] for r in rows]))
dist = np.linalg.norm(Z[:, None] - m['KM'].cluster_centers_[None], axis=2).min(1)
src = np.array([r['src'] for r in rows])

# ── ① 이상치 몽타주 ───────────────────────────────────────────────────────
u = np.where(src == 'ultimate')[0]
order = u[np.argsort(-dist[u])]
print("=" * 92)
print("① 학습셋 이상치 상위 (중심거리 큰 순)")
print(f"{'순위':>4}{'거리':>7}{'L':>6}{'a':>5}{'b':>5}{'tex1':>7}{'tex2':>7}{'px':>10}  출처")
for k, i in enumerate(order[:12], 1):
    r = rows[i]; L, a, b, t1, t2 = r['feats']
    print(f"{k:>4}{dist[i]:>7.2f}{L:>6.0f}{a:>5.0f}{b:>5.0f}{t1:>7.2f}{t2:>7.1f}"
          f"{f'{r[chr(112)+chr(120)+chr(95)+chr(108)+chr(111)+chr(110)+chr(103)]}x{r[chr(112)+chr(120)+chr(95)+chr(115)+chr(104)+chr(111)+chr(114)+chr(116)]}':>10}  {r['origin']}")

def montage(idxs, path, title, cols=8, th=110):
    cells = []
    for i in idxs:
        r = rows[i]
        bgr = cv2.imread(r['file'])
        if bgr is None:
            continue
        x, y, w, h = r['bbox']
        c = bgr[max(0, y):y + h, max(0, x):x + w]
        if c.size == 0:
            continue
        c = cv2.resize(c, (int(c.shape[1] * th / max(c.shape[0], 1)), th))
        c = cv2.copyMakeBorder(c, 2, 20, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(c, f"{dist[i]:.1f}", (4, th + 16), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (0, 0, 200), 1)
        cells.append(c)
    if not cells:
        return
    W = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, W - c.shape[1], cv2.BORDER_CONSTANT,
                                value=(255, 255, 255)) for c in cells]
    rowsimg = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    Wm = max(r.shape[1] for r in rowsimg)
    rowsimg = [cv2.copyMakeBorder(r, 0, 0, 0, Wm - r.shape[1], cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255)) for r in rowsimg]
    grid = np.vstack(rowsimg)
    head = np.full((34, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 0, 0), 2)
    cv2.imwrite(path, np.vstack([head, grid]))

montage(order[:24], OUT + '/eda3_outlier.png',
        'Outliers: farthest from cluster centers (number = distance)')
montage(u[np.argsort(dist[u])][:24], OUT + '/eda4_typical.png',
        'Typical: closest to cluster centers')

# ── ② 크기·종횡비·밀집도 (전체 ultimate COCO 로 정확히) ──────────────────
d = os.path.join(ROOT, 'final_dataset_ultimate', 'train')
coco = json.load(open(os.path.join(d, '_annotations.coco.json')))
cats = {c['id']: c['name'] for c in coco['categories']}
imsz = {im['id']: (im.get('width') or 0, im.get('height') or 0) for im in coco['images']}
per_cls = defaultdict(list)      # (w, h, area_ratio)
per_img_cnt = defaultdict(Counter)
for a in coco['annotations']:
    n = cats.get(a['category_id'], '?')
    w, h = a['bbox'][2], a['bbox'][3]
    W, H = imsz.get(a['image_id'], (0, 0))
    ar = (w * h) / (W * H) if W and H else np.nan
    per_cls[n].append((w, h, ar))
    per_img_cnt[a['image_id']][n] += 1

print("\n" + "=" * 92)
print("② 클래스별 bbox 크기·종횡비 (ultimate train 전체 379,190 라벨)")
print(f"{'클래스':<13}{'라벨수':>9}{'긴변 px 중앙':>13}{'짧은변':>8}{'종횡비':>8}"
      f"{'면적비%':>9}{'장당 개수(중앙/최대)':>20}")
for n in ['Pallet', 'Slip Sheet', 'Carton', 'Tote', 'Shrink Wrap']:
    v = per_cls.get(n)
    if not v:
        continue
    arr = np.array(v, float)
    lo = np.maximum(arr[:, 0], arr[:, 1]); sh = np.minimum(arr[:, 0], arr[:, 1])
    asp = lo / np.maximum(sh, 1)
    cnt = [c[n] for c in per_img_cnt.values() if n in c]
    print(f"{n:<13}{len(v):>9,}{np.median(lo):>13.0f}{np.median(sh):>8.0f}"
          f"{np.median(asp):>8.2f}{np.nanmedian(arr[:,2])*100:>9.2f}"
          f"{f'{int(np.median(cnt))} / {max(cnt)}':>20}")

# 그림: 클래스별 면적비 분포 + 이미지당 객체수
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
order5 = ['Carton', 'Tote', 'Shrink Wrap', 'Pallet', 'Slip Sheet']
cols = ['#27ae60', '#e67e22', '#c0392b', '#2980b9', '#8e44ad']
for n, c in zip(order5, cols):
    v = per_cls.get(n)
    if not v:
        continue
    ar = np.array(v, float)[:, 2] * 100
    ar = ar[np.isfinite(ar) & (ar > 0)]
    a1.hist(ar, bins=np.logspace(-2, 2, 60), alpha=.5, label=f'{n} ({len(v):,})',
            color=c, density=True)
a1.set_xscale('log'); a1.set_xlabel('객체 면적 / 이미지 면적 (%)  [로그]')
a1.set_ylabel('밀도'); a1.legend(fontsize=9); a1.grid(alpha=.3)
a1.set_title('클래스별 객체 크기 분포', fontsize=12)

names, counts = [], []
for n in order5:
    cnt = [c[n] for c in per_img_cnt.values() if n in c]
    if cnt:
        names.append(n); counts.append(cnt)
a2.boxplot(counts, labels=names, showfliers=False)
a2.set_ylabel('한 이미지당 객체 수'); a2.grid(alpha=.3)
a2.set_title('이미지당 객체 수 (이상치 제외)', fontsize=12)
plt.setp(a2.get_xticklabels(), rotation=15)
fig.suptitle('ultimate train — 클래스별 크기·밀집도', fontsize=13)
fig.tight_layout(); fig.savefig(OUT + '/eda5_size.png', dpi=110, facecolor='w')
print("\n그림:", OUT)
