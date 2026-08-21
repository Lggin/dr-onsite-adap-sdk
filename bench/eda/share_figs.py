"""발표용 정적 그림 — 3D 산점도를 못 띄우는 자리(캡처·인쇄)를 위한 대체.

  A 출처 × 클래스 교차표 히트맵 — 어느 칸이 비어 있는가
  B 도메인 간극 3중 측정
  C 축별 클래스 판별력 (F 비, 로그)
  D 현장 8종의 학습셋 내 위치 (마할라노비스 백분위)
"""
import os, pickle, random, sys, base64
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))   # gridsearch_feat 는 그룹화 벤치에 있다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # origins
from gridsearch_feat import NAMES, f3_rich
from origins import relabel, SYNTH, REAL as REALO

# 이 장비에는 한글 폰트가 .ttc(폰트 모음)로만 있어 matplotlib 기본 목록에 안 잡힌다.
# 직접 등록한다. CJK 모음의 첫 얼굴도 한글 글리프를 전부 포함한다.
_TTC = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_TTC):
    font_manager.fontManager.addfont(_TTC)
for cand in ('Noto Sans CJK JP', 'Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic'):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = cand
        print("한글 폰트:", cand)
        break
else:
    print("⚠️ 한글 폰트 없음 — 축 라벨을 영문으로 낮춤")
plt.rcParams['axes.unicode_minus'] = False
KO = 'CJK' in plt.rcParams['font.family'][0] or 'Nanum' in plt.rcParams['font.family'][0]

S = os.environ['S']
AX = NAMES['F3_rich']
rng = random.Random(0)
D = pickle.load(open(S + '/allfeat_train.pkl', 'rb'))
F3 = D['F3'].astype(np.float64)
CLS = np.array([m[0] for m in D['meta']])
ORG = relabel(D['meta'])

CLSO = ['Carton', 'Tote', 'Shrink Wrap', 'Pallet', 'Slip Sheet']
ORGO = ['합성(궤도렌더)', '합성(현장갭)', '합성(팔레트)', '합성(슈링크랩)',
        '외부(roboflow)', '현장(Thor)', '실촬(기타)', '실촬(DAS)', '슬립시트증강']
EN = {'합성(궤도렌더)': 'synth/orbit', '합성(현장갭)': 'synth/fieldgap',
      '합성(팔레트)': 'synth/pallet', '합성(슈링크랩)': 'synth/shrink',
      '외부(roboflow)': 'external', '현장(Thor)': 'field(Thor)',
      '실촬(기타)': 'real/other', '실촬(DAS)': 'real/DAS', '슬립시트증강': 'slip-aug'}
ylab = ORGO if KO else [EN[o] for o in ORGO]

fig = plt.figure(figsize=(15.5, 11))
gs = fig.add_gridspec(2, 2, hspace=.40, wspace=.28,
                      left=.13, right=.97, top=.885, bottom=.07)

# ── A 출처 × 클래스 ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
M = np.array([[int(((CLS == c) & (ORG == o)).sum()) for c in CLSO] for o in ORGO])
im = ax.imshow(np.log10(M + 1), cmap='YlGnBu', aspect='auto')
ax.set_xticks(range(len(CLSO)), CLSO, rotation=20, ha='right', fontsize=9)
ax.set_yticks(range(len(ORGO)), ylab, fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, '—' if v == 0 else f'{v:,}', ha='center', va='center',
                fontsize=8, color='#b00' if v == 0 else
                ('white' if np.log10(v + 1) > 3.6 else '#123'),
                fontweight='bold' if v == 0 else 'normal')
ax.set_title('A. 출처 × 클래스 (객체 수, 색은 log)\n빈 칸(—)이 학습 공백'
             if KO else 'A. origin x class', fontsize=11, loc='left')

# ── B 도메인 간극 ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
lbl = ['같은 도메인\n(합성→합성)', '합성 → 외부', '합성 → 실촬', '실촬 → 합성']
val = [98.5, 76.4, 44.8, 41.4]
col = ['#4caf50', '#ffb300', '#e53935', '#e53935']
b = ax.bar(range(4), val, color=col, width=.62)
ax.axhline(25, ls='--', c='#888', lw=1)
ax.text(-.42, 27, '무작위 25%', fontsize=8, color='#666', ha='left')
for r, v in zip(b, val):
    ax.text(r.get_x() + r.get_width() / 2, v + 1.6, f'{v:.1f}%',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(4), lbl, fontsize=9)
ax.set_ylim(0, 112); ax.set_ylabel('클래스 분류 정확도 (1-NN)', fontsize=9)
ax.set_title('B. 도메인 간극 — 공통 4클래스·개수 일치\n'
             '같은 도메인 98.5% → 건너가면 44.8%', fontsize=11, loc='left')
ax.spines[['top', 'right']].set_visible(False)

# ── C 축별 판별력 ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
CAP = 4000
idx = []
for c in sorted(set(CLS)):
    w = np.where(CLS == c)[0]
    idx.extend((w if len(w) <= CAP else np.array(rng.sample(list(w), CAP))).tolist())
idx = np.array(sorted(idx))
Xs = F3[idx]; ys = CLS[idx]
Zs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
rows = []
for i, nm in enumerate(AX):
    v = Zs[:, i]; gm = v.mean()
    bt = sum((ys == c).sum() * (v[ys == c].mean() - gm) ** 2
             for c in set(ys)) / (len(set(ys)) - 1)
    wt = sum(((v[ys == c] - v[ys == c].mean()) ** 2).sum()
             for c in set(ys)) / (len(v) - len(set(ys)))
    rows.append((bt / max(wt, 1e-12), nm))
rows.sort()
cols = ['#1565c0' if n.startswith('size') or n in ('solidity', 'aspect')
        else '#2e7d32' if n.startswith('tex') else '#ef6c00' for _, n in rows]
ax.barh(range(len(rows)), [r[0] for r in rows], color=cols, height=.68)
ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=9)
ax.set_xscale('log'); ax.set_xlabel('F 비 (클래스간 분산 / 클래스내 분산, 로그)', fontsize=9)
for i, (f, _) in enumerate(rows):
    ax.text(f * 1.12, i, f'{f:,.0f}', va='center', fontsize=8)
ax.set_title('C. 어느 축이 클래스를 가르는가\n'
             '파랑=크기·형태 · 초록=질감 · 주황=색', fontsize=11, loc='left')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(80, 12000)

# ── D 현장 8종 위치 ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
import cv2
objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb')) if r['cid'] == 2]
Xb, labs, cache = [], [], {}
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
        Xb.append(v); labs.append(r['type'].replace('_single', '·낱개'))
Xb = np.array(Xb, np.float64); labs = np.array(labs)
ct = F3[CLS == 'Carton']
mu = ct.mean(0); Si = np.linalg.pinv(np.cov(ct.T) + np.eye(ct.shape[1]) * 1e-6)


def maha(A):
    d = A - mu
    return np.sqrt(np.einsum('ij,jk,ik->i', d, Si, d))


dtr = np.sort(maha(ct[np.array(rng.sample(range(len(ct)), 20000))]))
pcs = []
for t in sorted(set(labs)):
    dm = np.median(maha(Xb[labs == t]))
    pcs.append((float(np.searchsorted(dtr, dm) / len(dtr) * 100), t))
pcs.sort()
cc = ['#c62828' if p >= 90 else '#ef6c00' if p >= 70 else '#2e7d32' for p, _ in pcs]
ax.barh(range(len(pcs)), [p for p, _ in pcs], color=cc, height=.68)
ax.axvline(95, ls='--', c='#c62828', lw=1.2)
ax.text(93, -.8, '95 경계', fontsize=8, color='#c62828', ha='right')
ax.set_yticks(range(len(pcs)), [t for _, t in pcs], fontsize=9)
ax.set_xlabel('학습 Carton 분포 내 백분위 (100=가장 바깥)', fontsize=9)
for i, (p, _) in enumerate(pcs):
    ax.text(p + 1.2, i, f'{p:.0f}', va='center', fontsize=8)
ax.set_xlim(0, 108)
ax.set_title('D. 현장 8종이 학습셋 안에 있는가\n'
             '크래프트 계열은 한복판, 파란 인쇄는 외곽', fontsize=11, loc='left')
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('학습 데이터셋 EDA — 전수 377,914 객체 (final_dataset_ultimate / train)',
             fontsize=13.5, x=.013, ha='left', y=.985)
out = os.path.expanduser('~/eda_요약_4패널.png')
fig.savefig(out, dpi=125, bbox_inches='tight', facecolor='white')
print("생성:", out, f"({os.path.getsize(out)/1024:.0f}KB)")
b64 = base64.b64encode(open(out, 'rb').read()).decode()
open(S + '/eda_fig_b64.txt', 'w').write(b64)
print("base64:", len(b64) // 1024, "KB")
