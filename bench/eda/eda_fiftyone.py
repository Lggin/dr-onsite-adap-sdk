"""FiftyOne 데이터셋 구축 — 객체 단위 크롭으로 색·질감 공간을 브라우저에서 탐색.

객체를 이미지로 보고 싶은 것이므로 **크롭을 개별 샘플**로 만든다(원본 프레임을 샘플로
쓰면 한 장에 수십 개가 섞여 클릭해도 어떤 객체인지 알 수 없다).
"""
import os, pickle, shutil
import numpy as np, cv2
import fiftyone as fo
import fiftyone.brain as fob

S = os.environ['S']
CROP = os.path.expanduser('~/eda_crops')
NAME = 'carton_eda'
rows = pickle.load(open(S + '/eda_objs.pkl', 'rb'))
model = pickle.load(open(S + '/eda_model.pkl', 'rb'))
sc, KM, pca = model['sc'], model['KM'], model['pca']

shutil.rmtree(CROP, ignore_errors=True)
os.makedirs(CROP, exist_ok=True)
if NAME in fo.list_datasets():
    fo.delete_dataset(NAME)

X = np.array([r['feats'] for r in rows])
Z = sc.transform(X)
clus = KM.predict(Z)
P = pca.transform(Z)
dist = np.linalg.norm(Z[:, None] - KM.cluster_centers_[None], axis=2).min(1)

samples = []
for i, r in enumerate(rows):
    bgr = cv2.imread(r['file'])
    if bgr is None:
        continue
    x, y, w, h = r['bbox']
    crop = bgr[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        continue
    p = os.path.join(CROP, f"{i:06d}_{r['src']}_{r['label']}.jpg")
    cv2.imwrite(p, crop)
    L, a, b, t1, t2 = r['feats']
    s = fo.Sample(filepath=p)
    s['src'] = r['src']                       # ultimate | box_types
    s['label'] = fo.Classification(label=r['label'])
    s['origin'] = r['origin']                 # 현장실촬 | 합성 | 외부/기타
    s['cluster'] = f'C{int(clus[i])}'
    s['L'] = L; s['a'] = a; s['b'] = b
    s['tex_hp'] = t1; s['tex_edge'] = t2
    s['px_long'] = r['px_long']; s['px_short'] = r['px_short']
    s['aspect'] = round(r['px_long'] / max(r['px_short'], 1), 3)
    s['mm_long'] = r['mm_long']; s['mm_short'] = r['mm_short']
    s['dist_to_center'] = float(dist[i])       # 학습셋 중심에서 얼마나 먼가 = 이상치 점수
    s['pc1'] = float(P[i, 0]); s['pc2'] = float(P[i, 1])
    s['source_file'] = os.path.basename(r['file'])
    samples.append(s)

ds = fo.Dataset(NAME, persistent=True)
ds.add_samples(samples)
print(f"샘플 {len(ds):,}개 등록")

# 색·질감 5차원을 그대로 임베딩으로 써서 2D 시각화(브라우저에서 lasso 선택 가능)
fob.compute_visualization(ds, embeddings=Z, brain_key='colortex', num_dims=2, verbose=False)
print("시각화 계산 완료 (brain_key='colortex')")
print("\n=== 요약 ===")
print(ds)
