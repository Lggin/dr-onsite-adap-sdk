"""box_types 프레임에 검출기를 돌려 마스크 + 물리 크기(mm)를 뽑는다.

폴더명 = 그룹 정답. 객체별 라벨링이 필요 없다.
depth 가 있으므로 그룹화 특징의 크기 2차원을 **실제 mm** 로 채울 수 있다(지금까지는 픽셀).
"""
import glob, json, os, pickle
import numpy as np, cv2
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config
from dr_onsite_adap.runtime.inference import SegModel
from dr_onsite_adap.runtime import geometry as GEO

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
W = os.path.join(S, 'continual_master_operational.pth')
ROOT = os.path.expanduser('~/box_types')
CONF = 0.3
NAMES = {c: n for c, (n, _) in config.CLASSES.items()}

def _scene_up(depth, intr):
    """장면의 지배적 평면(팔레트/바닥) 법선 = up. 반복 재가중으로 이상치 축소."""
    if depth is None or intr is None:
        return None
    fx, fy, cx, cy = intr
    H, Wd = depth.shape
    v, u = np.mgrid[0:H, 0:Wd]
    ok = np.isfinite(depth) & (depth > config.Z_MIN) & (depth < config.Z_MAX)
    if ok.sum() < 500:
        return None
    P = np.stack([(u[ok] - cx) / fx * depth[ok],
                  (v[ok] - cy) / fy * depth[ok], depth[ok]], 1)
    n = None
    for _ in range(3):
        c = P.mean(0); A = P - c
        n = np.linalg.svd(A, full_matrices=False)[2][-1]
        d = np.abs(A @ n)
        P = P[d < max(np.percentile(d, 80), 0.01)]
    n = n / np.linalg.norm(n)
    return -n if n[2] > 0 else n          # 카메라 위쪽을 향하도록


model = SegModel(W, config.NUM_CLASSES, 'cuda')
out = []
from collections import Counter, defaultdict
percls = defaultdict(Counter)

for d in sorted(x for x in glob.glob(os.path.join(ROOT, '*')) if os.path.isdir(x)):
    tname = os.path.basename(d)
    for f in sorted(glob.glob(os.path.join(d, '*.jpg'))):
        img = cv2.imread(f)
        npz = f[:-4] + '.npz'
        depth = intr = up = None
        if os.path.exists(npz):
            z = np.load(npz)
            depth = z['depth'] if 'depth' in z else None
            intr = tuple(z['intrinsics'].tolist()) if 'intrinsics' in z else None
            up = z['up_dir'] if 'up_dir' in z else None
            # ⚠️ 촬영 시 IMU 를 껐고(capture 가 up.json 을 읽지 않는다) npz 의 up_dir 이
            #    빈 배열로 저장됐다. 대신 **장면에서 직접 추정**한다 — 팔레트/바닥이
            #    지배적인 수평면이라 그 법선이 곧 up 이다. 48장 전부 서로 1도 이내로
            #    일치했고 demo_gui 의 UP_DIR 과 9.8도 차이(같은 마운트), 반면 ~/up.json
            #    과는 53도 차이라 up.json 쪽이 낡은 값이다.
            if up is None or np.asarray(up).size != 3:
                up = _scene_up(depth, intr)
        p = model.predict(img, conf_threshold=CONF)
        pb, pc, ps = np.asarray(p.xyxy), np.asarray(p.class_id), np.asarray(p.confidence)
        masks = getattr(p, 'mask', None)
        for k in range(len(pb)):
            cid = int(pc[k])
            percls[tname][NAMES.get(cid, cid)] += 1
            m = None
            if masks is not None and len(masks) > k:
                m = np.asarray(masks[k]).astype(bool)
            if m is None or m.sum() < config.CLUSTER_MIN_PX:
                x1, y1, x2, y2 = [int(v) for v in pb[k]]
                m = np.zeros(img.shape[:2], bool); m[max(0, y1):y2, max(0, x1):x2] = True
            rec = dict(type=tname, file=f, cid=cid, conf=float(ps[k]),
                       bbox=[float(v) for v in pb[k]], mask=m, img=f,
                       size_mm=None, geom=None, intr=intr)
            # 물리 크기 — depth 가 있으면 윗면 기하로 long/short mm 추정
            if depth is not None and intr is not None:
                try:
                    g, why = GEO.estimate_top_face_v2(m, depth, intr, config,
                                                      up_dir=up, with_pointcloud=False)
                    if g:
                        rec['geom'] = {k: v for k, v in g.items() if k != 'points'}
                        rec['size_mm'] = (float(g['long_side_mm']), float(g['short_side_mm']))
                    else:
                        rec['fail'] = why
                except Exception as e:
                    rec['fail'] = f'{type(e).__name__}: {e}'
            out.append(rec)

print(f"{'폴더':<22}{'검출 클래스 분포'}")
for t in sorted(percls):
    print(f"{t:<22}{dict(percls[t])}")
n_mm = sum(1 for r in out if r['size_mm'])
print(f"\n총 객체 {len(out)}개 | 물리크기(mm) 확보 {n_mm}개 ({n_mm/max(len(out),1)*100:.0f}%)")
with open(S + '/box_objs.pkl', 'wb') as fp:
    pickle.dump(out, fp)
print("저장:", S + '/box_objs.pkl')
