"""Carton 전용 EDA — 수치와 발표용 그림.

다른 클래스는 다루지 않는다. 관심사는 "학습셋의 Carton 이 어떻게 퍼져 있고, 현장
Carton 과 얼마나 다른가" 하나다.

⚠️ **중복 등록을 먼저 걷어낸다.** `ultimate` 은 현장 40장을 COCO images 에 각각 50번
등록하므로 현장 Carton 2,950 은 실제로 **59개**다. 걷어내지 않으면 도메인 비교에서
같은 객체의 복제본이 학습·시험 양쪽에 들어가 정확도가 부풀려진다(누수).

  A 출처 구성      — Carton 34만이 어디서 왔나 (중복 제거 후)
  B 밝기 분포      — 분류기 없이 간극을 보여준다 (표본이 작아도 무너지지 않는 지표)
  C 축별 간극      — 간극이 어느 축에 있는가 (코헨 d)
  D 현장 8종 위치  — box_types 514객체, 학습 Carton 분포 내 백분위
"""
import os, pickle, random, sys
from collections import Counter
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridsearch_feat import NAMES, f3_rich
from origins import relabel, dedup_mask, SYNTH
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

_TTC = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_TTC):
    font_manager.fontManager.addfont(_TTC)
for cand in ('Noto Sans CJK JP', 'Noto Sans CJK KR', 'NanumGothic'):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = cand
        break
plt.rcParams['axes.unicode_minus'] = False

S = os.environ['S']
AX = NAMES['F3_rich']
rng = random.Random(0)
D = pickle.load(open(S + '/allfeat_train.pkl', 'rb'))
CLS = np.array([m[0] for m in D['meta']])
ORG = relabel(D['meta'])
KEEP = dedup_mask(D['meta'], D['F3'])
raw_carton = int((CLS == 'Carton').sum())
cm = (CLS == 'Carton') & KEEP
X = D['F3'][cm].astype(np.float64)
O = ORG[cm]
FN = np.array([m[2] for m in D['meta']])[cm]
Z = StandardScaler().fit_transform(X)
print(f"Carton {raw_carton:,} → 중복 제거 {len(X):,}\n")

SYN = np.isin(O, SYNTH)
FIELD = O == '현장(Thor)'
EXT = O == '외부(roboflow)'

# ── A 출처 구성 ──────────────────────────────────────────────────────────
print("A. Carton 출처 구성 (중복 제거)")
cnt = Counter(O)
for o, n in cnt.most_common():
    print(f"   {o:<16}{n:>9,}{n/len(O)*100:>7.2f}%")
print(f"   ⚠️ 현장(Thor) 은 원래 2,950 으로 세어지지만 40장×50 복제라 실제 {int(FIELD.sum())}개")

# ── B 밝기 분포 ──────────────────────────────────────────────────────────
print("\nB. 밝기 L 분포 — 분류기 없이 본 간극")
iL = AX.index('L')
Bres = []
for nm, m in (('합성(전체)', SYN), ('외부(roboflow)', EXT), ('현장(Thor)', FIELD)):
    v = X[m, iL]
    Bres.append((nm, v))
    print(f"   {nm:<16}중앙 {np.median(v):>6.1f}  10~90%p "
          f"{np.percentile(v,10):>6.1f}~{np.percentile(v,90):>6.1f}  n={m.sum():,}")

# 분류기도 재본다 — 단 현장은 프레임 단위로 갈라 누수를 막는다
print("\n   참고) 출처 판별 정확도 — 현장은 프레임 단위 분할 (무작위 50%)")
frames = sorted(set(FN[FIELD]))
rng.shuffle(frames)
half = set(frames[:len(frames) // 2])
fa = FIELD & np.isin(FN, list(half))
fb = FIELD & ~np.isin(FN, list(half))
sa = np.array(rng.sample(list(np.where(SYN)[0]), int(fa.sum())))
sb = np.array(rng.sample(list(np.where(SYN)[0]), int(fb.sum())))
tr = np.concatenate([np.where(fa)[0], sa])
te = np.concatenate([np.where(fb)[0], sb])
ytr = np.array(['현장'] * int(fa.sum()) + ['합성'] * len(sa))
yte = np.array(['현장'] * int(fb.sum()) + ['합성'] * len(sb))
k = KNeighborsClassifier(3).fit(Z[tr], ytr)
acc_f = (k.predict(Z[te]) == yte).mean()
print(f"   합성 vs 현장    {acc_f*100:>6.1f}%   (학습 {len(tr)} / 시험 {len(te)}"
      f" — 프레임 {len(half)} vs {len(frames)-len(half)}장, 표본 작음)")
per = 2900
ia = np.array(rng.sample(list(np.where(SYN)[0]), per))
ib = np.array(rng.sample(list(np.where(EXT)[0]), per))
idx = np.concatenate([ia, ib]); lab = np.array(['합성'] * per + ['외부'] * per)
cut = np.array(rng.sample(range(len(idx)), len(idx) // 2))
tm = np.zeros(len(idx), bool); tm[cut] = True
k2 = KNeighborsClassifier(5).fit(Z[idx[tm]], lab[tm])
acc_e = (k2.predict(Z[idx[~tm]]) == lab[~tm]).mean()
print(f"   합성 vs 외부    {acc_e*100:>6.1f}%   (각 {per:,} — 중복 없음)")

# ── C 축별 간극 ──────────────────────────────────────────────────────────
print(f"\nC. 축별 도메인 간극 — 합성 vs 현장 Carton (현장 n={int(FIELD.sum())})")
fi = np.where(FIELD)[0]
si = np.array(rng.sample(list(np.where(SYN)[0]), 5000))
Cres = []
for i, nm in enumerate(AX):
    va, vb = Z[si, i], Z[fi, i]
    sp = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
    d = abs(va.mean() - vb.mean()) / max(sp, 1e-9)
    Cres.append((d, nm, float(np.median(X[si, i])), float(np.median(X[fi, i]))))
Cres.sort(reverse=True)
print(f"   {'축':<12}{'코헨 d':>8}{'합성 중앙':>11}{'현장 중앙':>11}")
for d, nm, ma, mb in Cres:
    print(f"   {nm:<12}{d:>8.2f}{ma:>11.1f}{mb:>11.1f}")

# ── D 현장 8종 (box_types) ───────────────────────────────────────────────
print("\nD. 현장 8종 (box_types) — 학습 Carton 분포 내 백분위 (100 = 가장 바깥)")
objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb')) if r['cid'] == 2]
xb, lb, cache = [], [], {}
for r in objs:
    if r['file'] not in cache:
        cache.clear(); cache[r['file']] = cv2.imread(r['file'])
    bgr = cache[r['file']]
    if bgr is None:
        continue
    x1, y1, x2, y2 = [int(v) for v in r['bbox']]
    v = f3_rich(bgr[max(0, y1):y2, max(0, x1):x2],
                r['mask'][max(0, y1):y2, max(0, x1):x2].astype(np.uint8), None)
    if v is not None:
        xb.append(v); lb.append(r['type'])
XB = np.array(xb, np.float64); LB = np.array(lb)
mu = X.mean(0); Si = np.linalg.pinv(np.cov(X.T) + np.eye(X.shape[1]) * 1e-6)


def maha(A):
    d = A - mu
    return np.sqrt(np.einsum('ij,jk,ik->i', d, Si, d))


dtr = np.sort(maha(X[np.array(rng.sample(range(len(X)), 30000))]))
Dres = []
for t in sorted(set(LB)):
    dm = float(np.median(maha(XB[LB == t])))
    Dres.append((float(np.searchsorted(dtr, dm) / len(dtr) * 100),
                 t.replace('_single', '·낱개'), dm))
Dres.sort()
for pc, t, dm in Dres:
    print(f"   {t:<22}거리 {dm:>5.2f}   백분위 {pc:>5.1f}")
out95 = (maha(XB) > dtr[int(.95 * len(dtr))]).mean() * 100
print(f"   현장 {len(XB)}객체 중 학습 Carton 95% 경계 밖: {out95:.1f}%")

# ── E 내부 구조 ──────────────────────────────────────────────────────────
print("\nE. Carton 내부 구조 — 군집 수별 실루엣")
sub = Z[np.array(rng.sample(range(len(Z)), 6000))]
for k in (2, 3, 4, 6, 8, 10, 15, 20):
    lab = MiniBatchKMeans(k, n_init=5, random_state=0, batch_size=2048).fit_predict(sub)
    print(f"   k={k:<3} {silhouette_score(sub, lab):.3f}")

# ══ 그림 ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15.5, 10.2))
gs = fig.add_gridspec(2, 2, hspace=.44, wspace=.26,
                      left=.15, right=.97, top=.875, bottom=.07)

ax = fig.add_subplot(gs[0, 0])
ords = [o for o, _ in cnt.most_common()][::-1]
vals = [cnt[o] for o in ords]
cols = ['#c62828' if o == '현장(Thor)' else '#8395a7' if o == '외부(roboflow)'
        else '#8e44ad' if o == '슬립시트증강' else '#2e86de' for o in ords]
ax.barh(range(len(ords)), vals, color=cols, height=.62)
ax.set_xscale('log')
ax.set_yticks(range(len(ords)), ords, fontsize=9.5)
for i, (o, v) in enumerate(zip(ords, vals)):
    t = f'{v:,}  ({v/len(O)*100:.2f}%)'
    if o == '현장(Thor)':
        t += '   ← 2,950 으로 세어지지만 40장×50 복제'
    ax.text(v * 1.25, i, t, va='center', fontsize=8.8,
            color='#c62828' if o == '현장(Thor)' else '#333',
            fontweight='bold' if o == '현장(Thor)' else 'normal')
ax.set_xlim(20, 3e6)
ax.set_xlabel('객체 수 (로그, 중복 제거)', fontsize=9)
ax.set_title(f'A. Carton {len(O):,}개는 어디서 왔나\n'
             f'실제 현장 박스는 {int(FIELD.sum())}개 = {FIELD.sum()/len(O)*100:.2f}%',
             fontsize=11.5, loc='left')
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 1])
bp = ax.violinplot([b[1] for b in Bres], vert=False, showmedians=True, widths=.85)
for pc, c in zip(bp['bodies'], ['#2e86de', '#8395a7', '#c62828']):
    pc.set_facecolor(c); pc.set_alpha(.55); pc.set_edgecolor(c)
for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
    if key in bp:
        bp[key].set_color('#333'); bp[key].set_linewidth(1.1)
ax.set_yticks(range(1, len(Bres) + 1),
              [f'{b[0]}\nn={len(b[1]):,}' for b in Bres], fontsize=9.5)
for i, b in enumerate(Bres):
    ax.text(np.median(b[1]), i + 1.34, f'중앙 {np.median(b[1]):.0f}',
            ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('밝기 L (0~255)', fontsize=9)
ax.set_xlim(0, 262)
ax.set_title('B. 밝기 분포 — 분류기 없이 본 간극\n'
             '합성은 L 전체에 퍼지고 현장은 밝은 쪽에 몰린다', fontsize=11.5, loc='left')
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, 0])
rr = Cres[::-1]
cc = ['#1565c0' if n.startswith('size') or n in ('solidity', 'aspect')
      else '#2e7d32' if n.startswith('tex') else '#ef6c00' for _, n, _, _ in rr]
ax.barh(range(len(rr)), [r[0] for r in rr], color=cc, height=.62)
ax.set_yticks(range(len(rr)), [r[1] for r in rr], fontsize=9.5)
for i, r in enumerate(rr):
    ax.text(r[0] + .02, i, f'{r[0]:.2f}', va='center', fontsize=8.8)
ax.set_xlabel('코헨 d — 합성 Carton 과 현장 Carton 의 차이 크기', fontsize=9)
ax.set_xlim(0, max(r[0] for r in rr) * 1.17)
ax.set_title(f'C. 간극은 어느 축에 있는가  (현장 n={int(FIELD.sum())}, 표본 작음)\n'
             '파랑=크기·형태 · 초록=질감 · 주황=색', fontsize=11.5, loc='left')
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, 1])
cc = ['#c62828' if p >= 90 else '#ef6c00' if p >= 70 else '#2e7d32' for p, _, _ in Dres]
ax.barh(range(len(Dres)), [p for p, _, _ in Dres], color=cc, height=.62)
ax.axvline(95, ls='--', c='#c62828', lw=1.2)
ax.text(93.5, -.9, '95 경계', fontsize=8.5, color='#c62828', ha='right')
ax.set_yticks(range(len(Dres)), [t for _, t, _ in Dres], fontsize=9.5)
for i, (p, _, _) in enumerate(Dres):
    ax.text(p + 1.2, i, f'{p:.0f}', va='center', fontsize=8.8)
ax.set_xlim(0, 108)
ax.set_xlabel('학습 Carton 분포 내 백분위 (100 = 가장 바깥)', fontsize=9)
ax.set_title(f'D. 직접 촬영한 8종은 학습셋 안에 있는가 (n={len(XB)})\n'
             '크래프트 계열은 한복판, 파란·네이비는 외곽', fontsize=11.5, loc='left')
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(f'Carton 전용 EDA — 학습셋 {len(O):,}객체 (중복 등록 제거 후) '
             '· final_dataset_ultimate / train',
             fontsize=13.5, x=.013, ha='left', y=.975)
out = os.path.expanduser('~/carton_eda_4패널.png')
fig.savefig(out, dpi=125, bbox_inches='tight', facecolor='white')
print(f"\n생성: {out} ({os.path.getsize(out)/1024:.0f}KB)")
