"""Carton 분포 3D 산점도 — 점을 클릭하면 그 객체를 본다.

Carton 만 다룬다. 학습셋의 Carton 이 색·질감·크기 공간에서 어떻게 퍼져 있는지 보고,
점을 눌러 실제 박스를 확인해 군집이 맞는지 눈으로 판정하는 것이 목적이다.

관점 4개.
  ① 출처별      — 합성·외부·현장이 같은 영역을 덮는가 (도메인 간극)
  ② 자동 군집   — 라벨 없이 나뉘는 덩어리 = Carton 안의 숨은 품목 구조
  ③ 현장 8종    — 실제로 촬영한 8종(큰 점)이 학습셋 어디에 앉는가
  ④ 해석 가능 축 — PCA 대신 밝기·색조·크기 원축

축(PCA)은 **Carton 전수 33만 개**로 학습한다. 표본으로 축을 잡으면 축 자체가 표본
추출에 흔들린다.

**점은 전수를 다 그린다**(회색 배경층). 표본을 쓰는 것은 **크롭 때문**이다 —
크롭 하나가 base64 로 약 2KB 이므로 33만 개면 670MB 가 되어 한 파일에 넣을 수 없다.
그래서 색이 있고 클릭되는 점만 표본이고, 분포의 모양 자체는 전수로 본다.

⚠️ 특징은 bbox 크롭 + 마스크 기반이고 크기 축이 **픽셀**이다(전수에 depth 없음).
   현장 8종도 같은 규격으로 다시 계산해 좌표계를 맞춘다.
"""
import base64, os, pickle, random, sys
from collections import Counter
import numpy as np
import cv2
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridsearch_feat import NAMES, f3_rich
from origins import relabel, dedup_mask, COLORS

S = os.environ['S']
OUT = os.path.expanduser('~/Carton_분포_3D.html')
NK = int(os.environ.get('NK', '8'))
AX = NAMES['F3_rich']
rng = random.Random(0)

# ── 축 학습: Carton 전수 ─────────────────────────────────────────────────
A = pickle.load(open(S + '/allfeat_train.pkl', 'rb'))
CLS = np.array([m[0] for m in A['meta']])
# 중복 등록(현장 40장×50)을 걷어내고 축을 잡는다 — 안 걷으면 축이 복제본에 끌린다
KEEP = dedup_mask(A['meta'], A['F3'])
POP = A['F3'][(CLS == 'Carton') & KEEP].astype(np.float64)
sc = StandardScaler().fit(POP)
p3 = PCA(3, random_state=0).fit(sc.transform(POP))
ev = p3.explained_variance_ratio_
print(f"축 학습: Carton 전수 {len(POP):,}")
print(f"PC1~3 설명력 {ev[0]*100:.0f}/{ev[1]*100:.0f}/{ev[2]*100:.0f}% (합 {ev[:3].sum()*100:.0f}%)")
for k in range(3):
    top = np.argsort(-np.abs(p3.components_[k]))[:5]
    print(f"  PC{k+1}: " + ', '.join(f"{AX[i]}{p3.components_[k][i]:+.2f}" for i in top))

# ── 전수 좌표 (배경층) — 점은 전부 그린다 ───────────────────────────────
# float32 로 낮춘다 — plotly 는 넘파이를 이진으로 직렬화하므로 dtype 이 곧 파일 크기다
# (f8 → f4 로 절반). 소수 1자리면 화면상 차이가 없다.
PALL = np.round(p3.transform(sc.transform(POP)), 1).astype(np.float32)
iLp, iCp, iSp = AX.index('L'), AX.index('chroma'), AX.index('size_long')
RALL = np.column_stack([np.round(POP[:, iLp]),
                        np.round(POP[:, iCp], 1),
                        np.round(np.log10(np.maximum(POP[:, iSp], 1)), 2)]
                       ).astype(np.float32)
print(f"배경층(전수) {len(PALL):,}점")

# ── 표본 + 크롭 ──────────────────────────────────────────────────────────
B = pickle.load(open(S + '/carton_sample.pkl', 'rb'))
X = B['F3'].astype(np.float64)
ORG = B['org']
MET = B['meta']
CROPS = B['crops']
CNT = B['counts']
TOTAL = B['total']
P = np.round(p3.transform(sc.transform(X)), 2)
print(f"표본 {len(X):,} / Carton 전체 {TOTAL:,}")

# ── 현장 8종을 같은 규격으로 ─────────────────────────────────────────────
FB, FLAB, FCROP = None, None, []
try:
    objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb')) if r['cid'] == 2]
    xb, lb, cache = [], [], {}
    for r in objs:
        if r['file'] not in cache:
            cache.clear(); cache[r['file']] = cv2.imread(r['file'])
        bgr = cache[r['file']]
        if bgr is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        crop = bgr[max(0, y1):y2, max(0, x1):x2]
        mk = r['mask'][max(0, y1):y2, max(0, x1):x2].astype(np.uint8)
        v = f3_rich(crop, mk, None)
        if v is None:
            continue
        s = 92 / max(crop.shape[:2])
        c = cv2.resize(crop, (max(1, round(crop.shape[1] * s)),
                              max(1, round(crop.shape[0] * s))),
                       interpolation=cv2.INTER_AREA) if s < 1 else crop
        ok, buf = cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 74])
        if not ok:
            continue
        xb.append(v); lb.append(r['type']); FCROP.append(buf.tobytes())
    XB = np.array(xb, np.float64); FLAB = np.array(lb)
    FB = np.round(p3.transform(sc.transform(XB)), 2)
    print(f"현장 8종 {len(FB)}개 (픽셀 규격 재계산)")
except Exception as e:
    print("현장 데이터 생략:", type(e).__name__, e)
    XB = None

# ── 자동 군집 ────────────────────────────────────────────────────────────
KC = KMeans(NK, n_init=10, random_state=0).fit_predict(sc.transform(X))
print(f"자동 군집 k={NK}: " + ' '.join(f"C{k}={int((KC==k).sum())}" for k in range(NK)))

# ── 크롭을 하나의 배열로(자바스크립트가 인덱스로 찾는다) ─────────────────
IMG = [base64.b64encode(c).decode() for c in CROPS]
FIMG = [base64.b64encode(c).decode() for c in FCROP]
NS = len(IMG)

# ── 해석 가능 축 ─────────────────────────────────────────────────────────
iL, iC, iS = AX.index('L'), AX.index('chroma'), AX.index('size_long')
R = np.round(np.column_stack([X[:, iL], X[:, iC],
                              np.log10(np.maximum(X[:, iS], 1))]), 2)
RB = (np.round(np.column_stack([XB[:, iL], XB[:, iC],
                                np.log10(np.maximum(XB[:, iS], 1))]), 2)
      if XB is not None else None)

PAL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
       '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']
PAL8 = ['#e6194b', '#3cb44b', '#bfa000', '#4363d8', '#f58231',
        '#911eb4', '#00b3b3', '#f032e6']

traces, groups = [], {}


def add(g, **kw):
    """g 는 관점 이름 하나 또는 여러 개(리스트). 배경층처럼 공용 트레이스는 여러 관점에."""
    i = len(traces)
    for gg in ([g] if isinstance(g, str) else g):
        groups.setdefault(gg, []).append(i)
    traces.append(go.Scatter3d(**kw))


def hov(i):
    fn, w, h, _ = MET[i]
    return f"{ORG[i]}<br>{w}×{h}px<br>C{KC[i]}<br>{fn[:34]}"


HOV = [hov(i) for i in range(NS)]
MK = dict(size=2.6, opacity=.72, line=dict(width=0))
BG = dict(size=1.0, color='#b9bfc6', opacity=.16, line=dict(width=0))

# 배경층: 전수 336,952점. 클릭·hover 없음(그럴 좌표만 있고 크롭이 없다)
add(['g1', 'g2', 'g3'], x=PALL[:, 0], y=PALL[:, 1], z=PALL[:, 2], mode='markers',
    name=f'전수 {len(PALL):,} (배경)', marker=dict(**BG), hoverinfo='skip',
    showlegend=True, visible=True)
add('g4', x=RALL[:, 0], y=RALL[:, 1], z=RALL[:, 2], mode='markers',
    name=f'전수 {len(RALL):,} (배경)', marker=dict(**BG), hoverinfo='skip',
    showlegend=True, visible=False)

# ① 출처별 — 많은 것부터 그려 적은 것이 위로 오게
for o in sorted(set(ORG), key=lambda x: -CNT.get(x, 0)):
    m = ORG == o
    w = np.where(m)[0]
    add('g1', x=P[w, 0], y=P[w, 1], z=P[w, 2], mode='markers',
        name=f'{o} {m.sum():,} / {CNT.get(o,0):,}',
        marker=dict(color=COLORS.get(o, '#888'), **MK),
        customdata=w.tolist(), text=[HOV[j] for j in w], hoverinfo='text', visible=True)

# ② 자동 군집
for k in range(NK):
    w = np.where(KC == k)[0]
    if not len(w):
        continue
    add('g2', x=P[w, 0], y=P[w, 1], z=P[w, 2], mode='markers',
        name=f'C{k} ({len(w):,})', marker=dict(color=PAL[k], **MK),
        customdata=w.tolist(), text=[HOV[j] for j in w], hoverinfo='text', visible=False)

# ③ 현장 8종 겹치기 (배경은 위의 전수층이 담당)
add('g3', x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers',
    name=f'표본 {NS:,} (클릭 가능)',
    marker=dict(size=2.0, color='#7f8b96', opacity=.5, line=dict(width=0)),
    customdata=list(range(NS)), text=HOV, hoverinfo='text', visible=False)
if FB is not None:
    off = 0
    for col, t in zip(PAL8, sorted(set(FLAB))):
        m = FLAB == t
        w = np.where(m)[0]
        add('g3', x=FB[w, 0], y=FB[w, 1], z=FB[w, 2], mode='markers',
            name=f'현장·{t} ({m.sum()})',
            marker=dict(size=5, color=col, line=dict(width=.7, color='#222')),
            customdata=(NS + w).tolist(),          # NS 이상은 현장 크롭
            text=[t] * len(w), hoverinfo='text', visible=False)

# ④ 해석 가능 축
for o in sorted(set(ORG), key=lambda x: -CNT.get(x, 0)):
    w = np.where(ORG == o)[0]
    add('g4', x=R[w, 0], y=R[w, 1], z=R[w, 2], mode='markers',
        name=f'{o} ({len(w):,})', marker=dict(color=COLORS.get(o, '#888'), **MK),
        customdata=w.tolist(), text=[HOV[j] for j in w], hoverinfo='text', visible=False)
if RB is not None:
    for col, t in zip(PAL8, sorted(set(FLAB))):
        w = np.where(FLAB == t)[0]
        add('g4', x=RB[w, 0], y=RB[w, 1], z=RB[w, 2], mode='markers',
            name=f'현장·{t}', marker=dict(size=5, color=col,
                                        line=dict(width=.7, color='#222')),
            customdata=(NS + w).tolist(), text=[t] * len(w), hoverinfo='text', visible=False)

N = len(traces)
PCS = dict(xaxis_title=f'PC1 ({ev[0]*100:.0f}%)', yaxis_title=f'PC2 ({ev[1]*100:.0f}%)',
           zaxis_title=f'PC3 ({ev[2]*100:.0f}%)')
RS = dict(xaxis_title='밝기 L (0~255)', yaxis_title='색조 강도 chroma',
          zaxis_title='크기 log10(긴 변 px)')


def upd(g, scene):
    v = [False] * N
    for i in groups[g]:
        v[i] = True
    return [{'visible': v}, {'scene': scene}]


fig = go.Figure(data=traces)
fig.update_layout(
    scene=PCS, height=760, margin=dict(l=0, r=0, t=54, b=0),
    legend=dict(itemsizing='constant', font=dict(size=10), x=0, y=1,
                bgcolor='rgba(255,255,255,.72)'),
    paper_bgcolor='white',
    updatemenus=[dict(type='buttons', direction='right', x=0, y=1.085,
                      xanchor='left', showactive=True, active=0, buttons=[
                          dict(label='① 출처별', method='update', args=upd('g1', PCS)),
                          dict(label='② 자동 군집', method='update', args=upd('g2', PCS)),
                          dict(label='③ 현장 8종', method='update', args=upd('g3', PCS)),
                          dict(label='④ 해석 가능 축', method='update', args=upd('g4', RS)),
                      ])])
body = fig.to_html(include_plotlyjs=True, full_html=False, div_id='plot')

# ── 클릭 → 객체 보기 ─────────────────────────────────────────────────────
META_JS = [[MET[i][0], int(MET[i][1]), int(MET[i][2]), str(ORG[i]), int(KC[i])]
           for i in range(NS)]
FMETA_JS = [[str(FLAB[i]), '현장 촬영'] for i in range(len(FIMG))]

# ── 축 해설: 하중이 아니라 "해석 가능한 양과의 상관" 으로 이름을 붙인다 ──
# 상위 하중 5개만 보고 이름을 붙이면 틀린다(PC1 의 최대 하중은 tex_lbp 지만
# 실제로는 크기·채도와 더 강하게 붙어 있다).
PPOP = p3.transform(sc.transform(POP))
QUANT = {'밝기 L': POP[:, AX.index('L')],
         '채도 chroma': POP[:, AX.index('chroma')],
         '긴 변 px': POP[:, AX.index('size_long')],
         '짧은 변 px': POP[:, AX.index('size_short')],
         '무늬 tex_lbp': POP[:, AX.index('tex_lbp')],
         '고주파 질감 tex_hp': POP[:, AX.index('tex_hp')],
         '에지 tex_edge': POP[:, AX.index('tex_edge')],
         '색 얼룩 L_iqr': POP[:, AX.index('L_iqr')]}
CORR = {k: [float(np.corrcoef(PPOP[:, k], v)[0, 1]) for v in QUANT.values()]
        for k in range(3)}
QN = list(QUANT)
PCNAME = ['크고 · 진하고 · 무늬 많은 ↔ 작고 · 흐리고 · 매끈한',
          '고주파 질감 많고 색 옅은 ↔ 질감 매끈하고 색 진한  (크기와 무관)',
          '밝고 · 크고 · 색 균일한 ↔ 어둡고 · 작고 · 색 얼룩진']
PCNOTE = ['근접 촬영(외부) 과 원거리 렌더(궤도) 를 가르는 축',
          '표면 무늬의 거칠기. 크기와 상관 0.02 로 독립이다',
          '<b>현장이 가장 잘 갈리는 축</b> (합성 vs 현장 코헨 d = 2.20)']


def corr_row(k):
    order = sorted(range(len(QN)), key=lambda i: -abs(CORR[k][i]))[:5]
    return ' · '.join(f"{QN[i]} <b>{CORR[k][i]:+.2f}</b>" for i in order)


pc_desc = ''.join(
    f"<div class=pcrow><span class=pcname>PC{k+1} <em>{ev[k]*100:.0f}%</em></span>"
    f"<span class=pcbody><b>{PCNAME[k]}</b><br>"
    f"<span class=pccorr>상관: {corr_row(k)}</span><br>"
    f"<span class=pcnote>{PCNOTE[k]}</span></span></div>"
    for k in range(3))
src_rows = '\n'.join(
    f"<tr><td>{o}</td><td class=n>{CNT[o]:,}</td>"
    f"<td class=n>{int((ORG==o).sum()):,}</td>"
    f"<td class=n>{int((ORG==o).sum())/CNT[o]*100:.1f}%</td></tr>"
    for o in sorted(CNT, key=lambda x: -CNT[x]))

import json
page = f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Carton 분포 3D — 클릭하면 객체가 보인다</title>
<style>
:root{{--ink:#20232a;--mut:#767a83;--line:#d8d4cc;--raise:#f2f0ec;--dep:#1d6fa5;
--kraft:#9c5f2b;--kraft-soft:#f6efe7}}
*{{box-sizing:border-box}}
body{{margin:0;background:#faf9f7;color:var(--ink);
font-family:"Pretendard","Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",
system-ui,sans-serif;font-size:15px;line-height:1.7}}
.wrap{{max-width:1560px;margin:0 auto;padding:20px 22px 60px}}
h1{{font-size:23px;margin:0 0 8px;letter-spacing:-.02em;font-weight:800}}
.sub{{font-size:13.5px;color:var(--mut);margin:0 0 16px;max-width:900px}}
.mono{{font-family:ui-monospace,"D2Coding",Consolas,monospace}}
.grid{{display:grid;grid-template-columns:1fr 320px;gap:18px;align-items:start}}
@media (max-width:1080px){{.grid{{grid-template-columns:1fr}}}}
#plot{{border:1px solid var(--line);background:#fff}}
.side{{position:sticky;top:16px;display:flex;flex-direction:column;gap:14px}}
.card{{border:1px solid var(--line);background:#fff;padding:14px 15px}}
.card h2{{font-size:11.5px;font-family:ui-monospace,monospace;letter-spacing:.13em;
text-transform:uppercase;color:var(--mut);margin:0 0 10px;font-weight:600}}
#big{{width:100%;aspect-ratio:1;background:var(--raise);border:1px solid var(--line);
display:flex;align-items:center;justify-content:center;overflow:hidden}}
#big img{{max-width:100%;max-height:100%;image-rendering:pixelated}}
#big .ph{{font-size:13px;color:var(--mut);text-align:center;padding:0 18px;line-height:1.6}}
#info{{margin-top:11px;font-size:12.5px;line-height:1.85}}
#info .r{{display:flex;justify-content:space-between;gap:10px;
border-bottom:1px dotted var(--line);padding:1px 0}}
#info .r span:first-child{{color:var(--mut)}}
#info .r span:last-child{{font-family:ui-monospace,monospace;text-align:right;
word-break:break-all}}
#strip{{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}}
#strip img{{width:100%;aspect-ratio:1;object-fit:contain;background:var(--raise);
border:1px solid var(--line);cursor:pointer;image-rendering:pixelated}}
#strip img:hover{{border-color:var(--dep)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{padding:4px 7px;border-bottom:1px solid var(--line);text-align:left}}
th{{font-size:10.5px;color:var(--mut);letter-spacing:.06em;text-transform:uppercase}}
td.n{{text-align:right;font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.axis{{background:var(--raise);border-left:3px solid var(--dep);padding:12px 15px;
font-size:13px;margin:14px 0;line-height:1.7}}
.axis > .t{{font-size:11px;font-family:ui-monospace,monospace;letter-spacing:.12em;
text-transform:uppercase;color:var(--mut);display:block;margin-bottom:9px}}
.pcrow{{display:flex;gap:14px;padding:7px 0;border-top:1px solid var(--line)}}
.pcrow:first-of-type{{border-top:0}}
.pcname{{flex:0 0 88px;font-family:ui-monospace,monospace;font-weight:700;
color:var(--dep);font-size:13.5px}}
.pcname em{{display:block;font-style:normal;font-weight:400;color:var(--mut);
font-size:11.5px}}
.pcbody{{flex:1;min-width:0}}
.pccorr{{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--mut)}}
.pcnote{{font-size:12px;color:var(--kraft)}}
.warn{{background:var(--kraft-soft);border-left:3px solid var(--kraft);padding:11px 15px;
font-size:13px;line-height:1.7;margin:14px 0}}
.warn b{{color:var(--kraft)}}
.hint{{font-size:12.5px;color:var(--mut);margin:9px 0 0}}
</style></head><body><div class=wrap>

<h1>Carton 분포 3D — 점을 클릭하면 그 객체가 보인다</h1>
<p class=sub>
학습셋의 <b>Carton {TOTAL:,}개</b>(중복 등록 제거 후)를 색·질감·크기 공간에
<b>전부</b> 그렸다 — 옅은 회색이 전수다. 그 위에 색이 있는 <b>{NS:,}개</b>는
<b>클릭하면 오른쪽에 그 박스가 뜬다</b>(크롭을 파일에 담을 수 있는 만큼).
드래그로 회전, 스크롤로 확대, 범례 클릭으로 그룹 켜고 끄기.
</p>

<div class=grid>
  <div>
    {body}
    <div class=axis><span class=t>x · y · z 축이 뜻하는 것 (관점 ①②③)</span>
      {pc_desc}
      <div class=pcrow><span class=pcname>관점 ④</span><span class=pcbody>
      PCA 를 쓰지 않고 <b>밝기 L · 채도 chroma · 크기 log10(긴 변 px)</b> 를 그대로
      축으로 쓴다. 해석은 쉽지만 세 축이 서로 겹친다.</span></div>
    </div>
    <p class=hint>
      ① 출처별 — 합성·외부·현장이 같은 영역을 덮는가 &nbsp;·&nbsp;
      ② 자동 군집 — 라벨 없이 나뉘는 덩어리 (k={NK}) &nbsp;·&nbsp;
      ③ 현장 8종 — 실제 촬영한 박스(큰 점)의 위치 &nbsp;·&nbsp;
      ④ 해석 가능 축 — 밝기·색조·크기 원축
    </p>
    <div class=warn>
      <b>읽을 때 주의</b> — 특징은 bbox 크롭 + 마스크 기반이고 <b>크기 축이 픽셀</b>이다
      (전수 데이터에 depth 가 없다). 같은 박스라도 카메라가 멀면 작게 나온다.
      운용 그룹화(윗면 3D 워프 + CNN 임베딩)와는 <b>다른 특징 공간</b>이므로 여기서
      보이는 군집을 그룹화 성능으로 읽으면 안 된다.
      PC1~3 설명력은 합 {ev[:3].sum()*100:.0f}% 다.
    </div>
  </div>

  <div class=side>
    <div class=card>
      <h2>클릭한 객체</h2>
      <div id=big><div class=ph>점을 클릭하세요</div></div>
      <div id=info></div>
    </div>
    <div class=card>
      <h2>최근 클릭 (누르면 다시 크게)</h2>
      <div id=strip></div>
    </div>
    <div class=card>
      <h2>표본 추출 — 출처별 √n 비례</h2>
      <table><thead><tr><th>출처</th><th class=n>전체</th><th class=n>표본</th>
        <th class=n>비율</th></tr></thead><tbody>
        {src_rows}
      </tbody></table>
      <p class=hint><b>점은 전수를 다 그린다.</b> 표본이 필요한 이유는 <b>크롭</b>이다 —
        크롭 하나가 base64 로 약 2KB 이므로 33만 개면 670MB 가 되어 한 파일에 못 담는다.
        좌표만이면 4.6MB 라서 전량이 가능하다.<br>
        클릭용 표본은 출처별 √n 비례로 뽑았다. 비례로만 뽑으면 현장이 0.02%라 안 보이고,
        균등이면 합성 18만의 실제 편중이 사라진다.
        <b>현장 59개는 전부 넣었다</b> — 그게 학습셋의 실제 현장 Carton 전부다.</p>
    </div>
  </div>
</div>

<script>
const IMG={json.dumps(IMG)};
const FIMG={json.dumps(FIMG)};
const MET={json.dumps(META_JS, ensure_ascii=False)};
const FMET={json.dumps(FMETA_JS, ensure_ascii=False)};
const NS={NS};
const big=document.getElementById('big'), info=document.getElementById('info'),
      strip=document.getElementById('strip'), hist=[];

function row(k,v){{return '<div class=r><span>'+k+'</span><span>'+v+'</span></div>';}}
function src(i){{return 'data:image/jpeg;base64,'+(i<NS?IMG[i]:FIMG[i-NS]);}}

function show(i){{
  big.innerHTML='<img src="'+src(i)+'" alt="객체 크롭">';
  if(i<NS){{
    const m=MET[i];
    info.innerHTML=row('출처',m[3])+row('크기',m[1]+' × '+m[2]+' px')
      +row('자동 군집','C'+m[4])+row('원본',m[0]);
  }} else {{
    const m=FMET[i-NS];
    info.innerHTML=row('종류',m[0])+row('출처',m[1]);
  }}
  if(hist[0]!==i){{
    hist.unshift(i); if(hist.length>12) hist.pop();
    strip.innerHTML=hist.map(j=>'<img src="'+src(j)+'" data-i="'+j+'" alt="크롭">').join('');
  }}
}}
strip.addEventListener('click',e=>{{
  const i=e.target.dataset&&e.target.dataset.i; if(i!==undefined) show(+i);
}});
(function bind(){{
  const gd=document.getElementById('plot');
  if(!gd||typeof gd.on!=='function'){{setTimeout(bind,60);return;}}
  gd.on('plotly_click',d=>{{
    const p=d.points&&d.points[0]; if(!p) return;
    let i=p.customdata;
    if(Array.isArray(i)) i=i[0];
    i=Number(i);
    if(!Number.isFinite(i)) return;
    show(i);
  }});
}})();
</script>
</div></body></html>"""
open(OUT, 'w', encoding='utf-8').write(page)
print("생성:", OUT, f"({os.path.getsize(OUT)/1024/1024:.1f}MB)")
