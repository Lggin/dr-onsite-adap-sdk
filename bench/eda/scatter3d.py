"""학습 데이터셋 3D 산점도 — 발표용 인터랙티브 HTML.

전수 추출한 14차원 색·질감·형태 특징(377,914 객체)을 3차원으로 투영해 회전하며 본다.
관점을 버튼으로 전환한다.

  ① 클래스별      — 5클래스가 특징 공간에서 갈리는가
  ② 출처별        — 합성/실촬/외부가 같은 영역을 덮는가 (도메인 간극)
  ③ Carton 군집   — Carton 안의 숨은 품목 구조
  ④ 현장 vs 학습셋 — box_types(현장 실촬 8종)를 같은 공간에 겹침
  ⑤ 해석 가능 축  — PCA 대신 밝기·색조·크기 원축 그대로

⚠️ 전수 특징은 bbox 크롭 기반이고 크기 축이 **픽셀**이다(전수에 depth 없음).
   현장 8종도 같은 규격(size_mm=None)으로 다시 계산해 좌표계를 맞춘다.
"""
import os, pickle, random, sys
import numpy as np
import cv2
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

S = os.environ['S']
sys.path.insert(0, S)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))   # gridsearch_feat 는 그룹화 벤치에 있다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # origins
from gridsearch_feat import f3_rich, NAMES
from origins import relabel, COLORS as ORGC

OUT = os.path.expanduser('~/데이터셋_3D산점도.html')
CAP = int(os.environ.get('CAP', '6000'))            # 일반 클래스 상한
CAP_CARTON = int(os.environ.get('CAP_CARTON', '15000'))
AX = NAMES['F3_rich']
rng = random.Random(0)

D = pickle.load(open(S + '/allfeat_train.pkl', 'rb'))
F3, META = D['F3'], D['meta']
CLS = np.array([m[0] for m in META])
ORG = relabel(META)   # render_* 를 합성으로 바로잡은 판정
FN = np.array([m[2] for m in META])
PX = np.array([[m[3], m[4]] for m in META], float)

idx = []
for c in sorted(set(CLS)):
    w = np.where(CLS == c)[0]
    lim = CAP_CARTON if c == 'Carton' else CAP
    idx.extend((w if len(w) <= lim else np.array(rng.sample(list(w), lim))).tolist())
idx = np.array(sorted(idx))
print(f"표본 {len(idx):,} / 전체 {len(CLS):,}")

X = F3[idx].astype(np.float64)
cls, org, fn, px = CLS[idx], ORG[idx], FN[idx], PX[idx]

sc = StandardScaler().fit(X)
p3 = PCA(3, random_state=0).fit(sc.transform(X))
P = p3.transform(sc.transform(X))
ev = p3.explained_variance_ratio_
print(f"PC1~3 설명력 {ev[0]*100:.0f}/{ev[1]*100:.0f}/{ev[2]*100:.0f}%  (합 {ev[:3].sum()*100:.0f}%)")
for k in range(3):
    top = np.argsort(-np.abs(p3.components_[k]))[:5]
    print(f"  PC{k+1}: " + ', '.join(f"{AX[i]}{p3.components_[k][i]:+.2f}" for i in top))

# ── 현장 8종을 같은 규격(픽셀 크기)으로 재계산해 투영 ────────────────────
Pb, labb = None, None
try:
    objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb')) if r['cid'] == 2]
    Xb, labs, cache = [], [], {}
    for r in objs:
        if r['file'] not in cache:
            cache.clear()
            cache[r['file']] = cv2.imread(r['file'])
        bgr = cache[r['file']]
        if bgr is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        crop = bgr[max(0, y1):y2, max(0, x1):x2]
        mk = r['mask'][max(0, y1):y2, max(0, x1):x2].astype(np.uint8)
        v = f3_rich(crop, mk, None)          # size_mm=None → 픽셀 (전수와 동일 규격)
        if v is not None:
            Xb.append(v); labs.append(r['type'])
    Xb = np.array(Xb, np.float64); labb = np.array(labs)
    Pb = np.round(p3.transform(sc.transform(Xb)), 2)
    print(f"현장 8종 {len(Pb)}개 투영 (픽셀 규격 재계산)")
except Exception as e:
    print("현장 데이터 투영 생략:", type(e).__name__, e)

# ── Carton 군집 (숨은 품목 구조) ─────────────────────────────────────────
cm = cls == 'Carton'
NK = 10
kc = MiniBatchKMeans(NK, n_init=5, random_state=0, batch_size=4096).fit_predict(
    sc.transform(X[cm]))

CLSC = {'Carton': '#27ae60', 'Tote': '#e67e22', 'Shrink Wrap': '#c0392b',
        'Pallet': '#2980b9', 'Slip Sheet': '#8e44ad'}
PAL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
PAL8 = ['#e6194b', '#3cb44b', '#bfa000', '#4363d8', '#f58231',
        '#911eb4', '#00b3b3', '#f032e6']

HOV = [f"{cls[i]}<br>출처 {org[i]}<br>{px[i,0]:.0f}×{px[i,1]:.0f}px<br>{fn[i][:30]}"
       for i in range(len(P))]
# 좌표 소수점을 줄여 HTML 용량을 낮춘다(화면상 차이 없음, plotly.js 를 파일에 넣기 위함)
P = np.round(P, 2)

# ⑤ 해석 가능 축: 밝기 L / 색조 chroma / 크기 size_long(로그)
iL, iC, iS = AX.index('L'), AX.index('chroma'), AX.index('size_long')
R = np.round(np.column_stack(
    [X[:, iL], X[:, iC], np.log10(np.maximum(X[:, iS], 1))]), 2)
Rb = (np.round(np.column_stack(
    [Xb[:, iL], Xb[:, iC], np.log10(np.maximum(Xb[:, iS], 1))]), 2)
      if Pb is not None else None)

traces, groups = [], {}


def add(gname, **kw):
    groups.setdefault(gname, []).append(len(traces))
    traces.append(go.Scatter3d(**kw))


def pts(mask_, coords):
    w = np.where(mask_)[0]
    return coords[w, 0], coords[w, 1], coords[w, 2], w


for c, col in CLSC.items():                                   # ① 클래스별
    m = cls == c
    if not m.any():
        continue
    x, y, z, w = pts(m, P)
    add('g1', x=x, y=y, z=z, mode='markers', name=f'{c} ({m.sum():,})',
        marker=dict(size=1.7, color=col, opacity=.5),
        text=[HOV[j] for j in w], hoverinfo='text', visible=True)

for c, col in ORGC.items():                                   # ② 출처별
    m = org == c
    if not m.any():
        continue
    x, y, z, w = pts(m, P)
    add('g2', x=x, y=y, z=z, mode='markers', name=f'{c} ({m.sum():,})',
        marker=dict(size=1.7, color=col, opacity=.5),
        text=[HOV[j] for j in w], hoverinfo='text', visible=False)

ci = np.where(cm)[0]                                          # ③ Carton 군집
for k in range(NK):
    m = kc == k
    if not m.any():
        continue
    sel = ci[m]
    add('g3', x=P[sel, 0], y=P[sel, 1], z=P[sel, 2], mode='markers',
        name=f'C{k} ({m.sum():,})', marker=dict(size=1.7, color=PAL[k], opacity=.55),
        text=[HOV[j] for j in sel], hoverinfo='text', visible=False)

add('g4', x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers',    # ④ 현장 vs 학습셋
    name=f'학습셋 전체 ({len(P):,})',
    marker=dict(size=1.3, color='#cfd8dc', opacity=.3), hoverinfo='skip', visible=False)
if Pb is not None:
    for col, t in zip(PAL8, sorted(set(labb))):
        m = labb == t
        add('g4', x=Pb[m, 0], y=Pb[m, 1], z=Pb[m, 2], mode='markers',
            name=f'현장·{t} ({m.sum()})',
            marker=dict(size=4.5, color=col, line=dict(width=.6, color='#222')),
            text=[t] * int(m.sum()), hoverinfo='text', visible=False)

for c, col in CLSC.items():                                   # ⑤ 원축
    m = cls == c
    if not m.any():
        continue
    x, y, z, w = pts(m, R)
    add('g5', x=x, y=y, z=z, mode='markers', name=f'{c} ({m.sum():,})',
        marker=dict(size=1.7, color=col, opacity=.5),
        hoverinfo='skip', visible=False)
if Rb is not None:
    for col, t in zip(PAL8, sorted(set(labb))):
        m = labb == t
        add('g5', x=Rb[m, 0], y=Rb[m, 1], z=Rb[m, 2], mode='markers',
            name=f'현장·{t}', marker=dict(size=4.5, color=col,
                                        line=dict(width=.6, color='#222')),
            text=[t] * int(m.sum()), hoverinfo='text', visible=False)

N = len(traces)
PC_SCENE = dict(xaxis_title=f'PC1 ({ev[0]*100:.0f}%)',
                yaxis_title=f'PC2 ({ev[1]*100:.0f}%)',
                zaxis_title=f'PC3 ({ev[2]*100:.0f}%)')
R_SCENE = dict(xaxis_title='밝기 L (0~255)', yaxis_title='색조 강도 chroma',
               zaxis_title='크기 log10(긴 변 px)')


def upd(g, scene, title):
    v = [False] * N
    for i in groups[g]:
        v[i] = True
    return [{'visible': v}, {'scene': scene, 'title.text': title}]


T = f'학습 데이터셋 특징 공간 (표본 {len(P):,} / 전체 377,914)'
fig = go.Figure(data=traces)
fig.update_layout(
    title=dict(text=f'{T} — ① 클래스별', x=0.02, y=0.97, font=dict(size=15)),
    scene=PC_SCENE, height=800, margin=dict(l=0, r=0, t=90, b=0),
    legend=dict(itemsizing='constant', font=dict(size=10)),
    updatemenus=[dict(type='buttons', direction='right', x=0.01, y=1.13,
                      xanchor='left', showactive=True, active=0, buttons=[
                          dict(label='① 클래스별', method='update',
                               args=upd('g1', PC_SCENE, f'{T} — ① 클래스별')),
                          dict(label='② 출처별', method='update',
                               args=upd('g2', PC_SCENE, f'{T} — ② 출처별(도메인 간극)')),
                          dict(label='③ Carton 군집', method='update',
                               args=upd('g3', PC_SCENE, f'{T} — ③ Carton 내부 구조')),
                          dict(label='④ 현장 vs 학습셋', method='update',
                               args=upd('g4', PC_SCENE, f'{T} — ④ 현장 8종 위치')),
                          dict(label='⑤ 해석 가능 축', method='update',
                               args=upd('g5', R_SCENE, f'{T} — ⑤ 밝기·색조·크기 원축')),
                      ])])
# plotly.js 를 파일 안에 넣어 사내망/오프라인에서도 열리게 한다
body = fig.to_html(include_plotlyjs=True, full_html=False, div_id='plot3d')

pc_desc = '<br>'.join(
    f"<b>PC{k+1}</b> ({ev[k]*100:.0f}%) = " +
    ' '.join(f"{p3.components_[k][i]:+.2f}·{AX[i]}"
             for i in np.argsort(-np.abs(p3.components_[k]))[:5])
    for k in range(3))
page = f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>학습 데이터셋 3D 산점도</title>
<style>
body{{font-family:-apple-system,'Malgun Gothic','Noto Sans KR',sans-serif;
margin:16px 20px;color:#172b4d;max-width:1400px}}
h2{{margin:2px 0 10px}}
.note{{font-size:13px;color:#5e6c84;line-height:1.8}}
.views{{font-size:13px;line-height:1.9;margin:10px 0}}
.views b{{color:#0052cc}}
.axis{{background:#f4f5f7;border-left:3px solid #4c9aff;padding:9px 13px;
font-size:12.5px;margin:12px 0;line-height:1.8;font-family:ui-monospace,monospace}}
.warn{{background:#fffae6;border-left:3px solid #ffab00;padding:9px 13px;
font-size:12.5px;line-height:1.8;margin:14px 0}}
</style></head><body>
<h2>학습 데이터셋 특징 공간 (3D)</h2>
<div class="note">
<b>드래그로 회전</b> · 스크롤로 확대 · 점에 올리면 클래스·출처·파일명 표시 ·
범례 클릭으로 그룹 켜고 끄기
</div>
<div class="views">
<b>①</b> 클래스별 — 5클래스가 색·질감·크기만으로 갈리는가<br>
<b>②</b> 출처별 — 합성·실촬·외부가 같은 영역을 덮는가 (<b>도메인 간극</b>이 눈에 보이는 곳)<br>
<b>③</b> Carton 군집 — 라벨은 하나인 Carton 안에 <b>품목 구조</b>가 있는가 (서브클래스 그룹화의 근거)<br>
<b>④</b> 현장 vs 학습셋 — 현장 촬영 8종(큰 점)이 학습셋(회색) 안에 들어와 있는가<br>
<b>⑤</b> 해석 가능 축 — PCA 대신 밝기·색조·크기 <b>원축</b>. 축 의미가 바로 읽힌다
</div>
<div class="axis">축 구성 (14개 특징을 표준화한 뒤의 가중합, 상위 5개만)<br>{pc_desc}</div>
{body}
<div class="warn">
<b>읽을 때 주의</b><br>
· 특징은 <b>bbox 크롭 + 마스크</b> 기반이고 크기 축은 <b>픽셀</b>이다 — 전수 데이터에
depth 가 없어 물리 mm 를 쓸 수 없다. 현장 8종도 같은 규격으로 다시 계산해 좌표계를 맞췄다.<br>
· 따라서 이 그림은 운용 그룹화 파이프라인(윗면 3D 워프 + CNN 임베딩, 정확도 98~99%)과
<b>다른 특징 공간</b>이다. "학습셋이 어떻게 분포하는가"를 보는 용도이며 그룹화 성능과
직접 대응하지 않는다.<br>
· 표본은 클래스별 상한(Carton {CAP_CARTON:,} / 그 외 {CAP:,})으로 뽑았다.
실제 분포는 Carton 이 90%다.
</div></body></html>"""
open(OUT, 'w', encoding='utf-8').write(page)
print("생성:", OUT, f"({os.path.getsize(OUT)/1024**2:.1f}MB)")
