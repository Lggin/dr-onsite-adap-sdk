"""① depth 자료에 운용 알고리즘 그대로 적용 → 군집 몽타주 HTML.

파이프라인(실제 운용과 동일):
  RFDETR 마스크 → 윗면 3D 사각형(estimate_top_face_v2) → 종횡비 보존 워프
  → ResNet50(fc 제거) 4방향 회전평균 → PCA16 → 손수 7차원과 결합 → 군집

⚠️ 정답 라벨을 **쓰지 않는다**(비지도). 사람이 눈으로 "한 줄에 다른 박스가 섞였나"를
   보고 평가할 수 있게, 군집별로 크롭을 나열한다.
"""
import os, pickle, base64, html
from collections import defaultdict, Counter
import numpy as np, cv2, torch
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime import clustering as CL
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
OUT = os.path.expanduser(os.environ.get('OUTHTML','~/그룹화_검증_depth.html'))


def warp_w2(bgr, geom, intr):
    fx, fy, cx, cy = intr
    c = np.asarray(geom[0], float)
    ua = np.asarray(geom[1], float); va = np.asarray(geom[2], float)
    lmm, smm = float(geom[3]), float(geom[4])
    if lmm <= 0 or smm <= 0:
        return None
    hl, hs = lmm / 2000.0, smm / 2000.0
    pts = [c + a * hl * ua + b * hs * va for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    src = []
    for p in pts:
        if p[2] <= 1e-6:
            return None
        src.append([p[0] / p[2] * fx + cx, p[1] / p[2] * fy + cy])
    H = max(8, int(round(PATCH * max(smm / lmm, 0.05))))
    dst = np.array([[0, 0], [PATCH, 0], [PATCH, H], [0, H]], np.float32)
    M = cv2.getPerspectiveTransform(np.array(src, np.float32), dst)
    return cv2.warpPerspective(bgr, M, (PATCH, H))


def prep_native(patch):
    h, w = patch.shape[:2]
    s = 224 / max(h, w)
    nw, nh = max(16, int(round(w * s))), max(16, int(round(h * s)))
    im = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


@torch.no_grad()
def embed(model, patches):
    out = []
    for p in patches:
        acc = None
        for k in (0, 1, 2, 3):
            x = prep_native(np.rot90(p, k).copy()).unsqueeze(0)
            v = model(x.to(DEV)).float().cpu().numpy()
            acc = v if acc is None else acc + v
        out.append(acc / 4)
    return np.concatenate(out)


class Cfg:
    def __init__(self):
        for k in dir(config):
            if k.startswith('CLUSTER_'):
                setattr(self, k, getattr(config, k))
cfg = Cfg()

objs = [r for r in pickle.load(open(S + '/box_objs.pkl', 'rb'))
        if r['cid'] == 2 and r['size_mm'] and r['geom']]
byfile = defaultdict(list)
for r in objs:
    byfile[r['file']].append(r)

BASE, PATCHES, CROPS, TRUTH, MM = [], [], [], [], []
for f, rs in byfile.items():
    img = cv2.imread(f)
    geoms = [(r['geom']['center_3d'], r['geom']['long_axis_3d'], r['geom']['short_axis_3d'],
              r['geom']['long_side_mm'], r['geom']['short_side_mm']) for r in rs]
    fs, _ = CL._build_features([r['mask'] for r in rs], [r['size_mm'] for r in rs],
                               geoms, img, rs[0]['intr'], cfg)
    for r, ft, gm in zip(rs, fs, geoms):
        if ft is None:
            continue
        p = warp_w2(img, gm, r['intr'])
        if p is None or min(p.shape[:2]) < 8:
            continue
        x1, y1, x2, y2 = [int(v) for v in r['bbox']]
        CROPS.append(img[max(0, y1):y2, max(0, x1):x2].copy())
        BASE.append(ft); PATCHES.append(p); MM.append(r['size_mm'])
        t = r['type'][:-7] if r['type'].endswith('_single') else r['type']
        TRUTH.append(t)
BASE = np.stack(BASE); MM = np.stack(MM).astype(float); TRUTH = np.array(TRUTH)
print(f"객체 {len(BASE)}개 / 촬영 종류 {len(set(TRUTH))}종")

import torchvision.models as tvm
model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Identity(); model.eval().to(DEV)
E = embed(model, PATCHES)
Ep = PCA(16, random_state=0).fit_transform(E); Ep /= max(Ep.std(), 1e-9)
scale, bw = CL._scale_bw(config.CLUSTER_SIZE_SCALE_MM, cfg)
Z = np.hstack([BASE / scale * bw, Ep * 0.35])
print(f"특징 {Z.shape}")

# 군집 — 품목 수를 모른다는 전제로 거리 임계 기반(완전연결).
# ⚠️ 임계를 "군집 수 = 촬영 종류 수"에 맞추면 안 된다(정답을 미리 아는 셈).
#    실무에서는 과분할이 오분류보다 안전하다 — 같은 박스가 나뉘면 사람이 합칠 수 있지만
#    다른 박스가 섞이면 잘못 집는다. 그래서 순도를 우선해 임계를 고른다.
THR = float(os.environ.get('THR', '2.5'))
LABC = AgglomerativeClustering(n_clusters=None, distance_threshold=THR,
                               linkage='complete').fit_predict(Z)
thr, k = THR, len(set(LABC))
print(f"→ 임계 {thr} (군집 {k}개, 최대 {Counter(LABC).most_common(1)[0][1]}개)")


def b64(im, th=104):
    h, w = im.shape[:2]
    im = cv2.resize(im, (max(1, int(w * th / max(h, 1))), th))
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 76])
    return base64.b64encode(buf).decode() if ok else ''


rows = []
for gi, cnt in Counter(LABC).most_common():
    idx = np.where(LABC == gi)[0]
    tc = Counter(TRUTH[idx])
    pure = tc.most_common(1)[0][1] / len(idx)
    cells = ''.join(
        f'<img src="data:image/jpeg;base64,{b64(CROPS[i])}" title="{html.escape(TRUTH[i])} '
        f'{MM[i,0]:.0f}x{MM[i,1]:.0f}mm">' for i in idx[:24])
    mix = ' · '.join(f'{html.escape(t)}:{n}' for t, n in tc.most_common())
    badge = 'ok' if pure >= 0.95 else ('warn' if pure >= 0.8 else 'bad')
    rows.append(f'''<div class="grp">
<div class="hd"><b>군집 {gi}</b> <span class="n">{len(idx)}개</span>
<span class="pu {badge}">순도 {pure*100:.0f}%</span>
<span class="mx">{mix}</span></div>
<div class="imgs">{cells}{'<span class="more">+%d</span>' % (len(idx)-24) if len(idx) > 24 else ''}</div></div>''')

# 정답 대비 지표(참고용)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score([sorted(set(TRUTH)).index(t) for t in TRUTH], LABC)
purity = sum(Counter(TRUTH[LABC == g]).most_common(1)[0][1] for g in set(LABC)) / len(TRUTH)

CSS = """body{font-family:-apple-system,'Malgun Gothic',sans-serif;margin:16px;background:#fafbfc;color:#172b4d}
h2{margin:6px 0}.meta{color:#5e6c84;font-size:13px;margin-bottom:14px;line-height:1.7}
.grp{background:#fff;border:1px solid #dfe1e6;border-radius:6px;margin-bottom:10px;padding:8px 10px}
.hd{font-size:13px;margin-bottom:6px}.n{color:#5e6c84;margin-left:6px}
.pu{padding:1px 7px;border-radius:9px;margin-left:8px;font-weight:700;font-size:12px}
.pu.ok{background:#e3fcef;color:#006644}.pu.warn{background:#fffae6;color:#974f0c}
.pu.bad{background:#ffebe6;color:#bf2600}
.mx{color:#7a869a;margin-left:10px;font-size:12px}
.imgs{display:flex;flex-wrap:wrap;gap:3px;align-items:flex-start}
.imgs img{height:104px;border:1px solid #ebecf0;border-radius:3px}
.more{color:#7a869a;font-size:12px;align-self:center;padding:0 6px}"""
doc = f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>그룹화 검증 — depth 자료(운용 알고리즘)</title><style>{CSS}</style></head><body>
<h2>그룹화 검증 ① — depth 자료에 운용 알고리즘 그대로</h2>
<div class="meta">
객체 <b>{len(Z)}</b>개 · 군집 <b>{k}</b>개 (거리임계 {thr}, 완전연결) ·
<b>정답 라벨을 쓰지 않은 비지도 결과</b>입니다.<br>
파이프라인: RFDETR 마스크 → 윗면 3D 사각형 → 종횡비 보존 워프 →
ResNet50(fc 제거) 4방향 회전평균 → PCA16 → 손수 7차원(색·질감·물리크기) 결합 → 완전연결 군집<br>
<b>보는 법</b>: 한 줄이 한 군집입니다. <b>줄 안에 다른 박스가 섞여 있으면 실패</b>,
같은 박스가 여러 줄로 나뉘어 있으면 과분할입니다. 이미지에 커서를 올리면 촬영 종류·실측 크기가 보입니다.<br>
참고 지표(촬영 종류를 정답으로 봤을 때): 순도 <b>{purity*100:.1f}%</b> · ARI <b>{ari:.3f}</b>
</div>
{''.join(rows)}
</body></html>"""
open(OUT, 'w', encoding='utf-8').write(doc)
print(f"\n순도 {purity*100:.1f}%  ARI {ari:.3f}")
print("생성:", OUT, f"({os.path.getsize(OUT)/1024/1024:.1f}MB)")
