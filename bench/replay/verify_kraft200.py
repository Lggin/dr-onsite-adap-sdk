"""생성된 크래프트 200장 셋 검증 — 무결성 + 목표 달성 여부.

세 가지를 본다.

1. **무결성** — 파일 누락, train/valid 파일명 교차, 중복, 라벨 0인 클래스
2. **목표 달성** — 크래프트 근접도가 기존 400장 셋과 무작위 200장 대비 실제로 좋아졌나
3. **부작용** — 클래스 보존이 얼마나 얇아졌나, 출처가 한쪽으로 쏠렸나

⚠️ 여기서 확인하는 것은 "의도한 대로 뽑혔는가" 뿐이다. **학습 성능은 A/B 로만 알 수
있다** — 이전에 다른 목적의 큐레이션이 무작위보다 나빴던 전례가 있다.
"""
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'eda'))
from origins import origin           # noqa: E402

S = os.environ['S']
NEW = os.path.expanduser(os.environ.get('KRAFT_OUT', '~/depal-al/data/replay_kraft200'))
OLD = os.path.expanduser('~/depal-al/data/replay_fixed_v11')
CLASSES = ['Pallet', 'Slip Sheet', 'Carton', 'Tote', 'Shrink Wrap']
rng = random.Random(0)
SC = pickle.load(open(os.path.join(S, 'kraft200_scan.pkl'), 'rb'))

print("=" * 74)
print("1. 무결성")
print("=" * 74)
names = {}
ok = True
for split in ('train', 'valid'):
    d = os.path.join(NEW, split)
    coco = json.load(open(os.path.join(d, '_annotations.coco.json')))
    cname = {c['id']: c['name'] for c in coco['categories']}
    fns = [im['file_name'] for im in coco['images']]
    names[split] = set(fns)
    missing = [f for f in fns if not os.path.isfile(os.path.join(d, f))]
    ondisk = {f for f in os.listdir(d) if not f.endswith('.json')}
    orphan = ondisk - set(fns)
    dup = [f for f, n in Counter(fns).items() if n > 1]
    ids = [im['id'] for im in coco['images']]
    aimg = {a['image_id'] for a in coco['annotations']}
    inst = Counter(cname[a['category_id']] for a in coco['annotations'])
    empty = [c for c in CLASSES if inst[c] == 0]
    print(f"\n[{split}] 이미지 {len(fns)} · 라벨 {len(coco['annotations']):,}")
    print(f"   파일 누락 {len(missing)} · 고아 파일 {len(orphan)} · 중복 파일명 {len(dup)}")
    print(f"   image id 중복 {len(ids)-len(set(ids))} · 주석이 가리키는 없는 id "
          f"{len(aimg - set(ids))}")
    print(f"   라벨 0인 클래스: {empty if empty else '없음'}")
    if missing or orphan or dup or (len(ids) != len(set(ids))) or (aimg - set(ids)):
        ok = False
    if empty and split == 'train':
        ok = False
cross = names['train'] & names['valid']
print(f"\ntrain/valid 파일명 교차: {len(cross)}장 {'✅' if not cross else '❌ ' + str(list(cross)[:3])}")
if cross:
    ok = False

print("\n" + "=" * 74)
print("2. 목표 달성 — 크래프트 근접도")
print("=" * 74)
dist = SC['train']['dist']
newtr = [f for f in names['train'] if f in dist and np.isfinite(dist[f])]
oldtr = [f for f in os.listdir(os.path.join(OLD, 'train'))
         if not f.endswith('.json')]
oldtr_ok = [f for f in oldtr if f in dist and np.isfinite(dist[f])]
pool = [f for f in SC['train']['files'] if np.isfinite(dist[f])]
rand200 = rng.sample(pool, 200)


def stat(label, fs):
    d = np.array([dist[f] for f in fs])
    return (f"{label:<26}n={len(fs):>4}  중앙 {np.median(d):>5.2f}  "
            f"평균 {d.mean():>5.2f}  25%p {np.percentile(d,25):>5.2f}  "
            f"75%p {np.percentile(d,75):>5.2f}  "
            f"적격(≤{THR:.2f}) {int((d<=THR).sum())/len(d)*100:>5.1f}%")


THR = float(np.percentile([dist[f] for f in pool], 25))
print(f"(적격 = 소스 Carton 이미지 중 크래프트 근접 상위 25%, 거리 ≤ {THR:.2f})\n")
print(stat('신규 크래프트 200', newtr))
print(stat('기존 고정셋 400', oldtr_ok))
print(stat('무작위 200 (대조)', rand200))
print(stat('소스 전체', pool))
dn = np.median([dist[f] for f in newtr])
do = np.median([dist[f] for f in oldtr_ok])
print(f"\n→ 기존 400 대비 크래프트 거리 중앙값 {do:.2f} → {dn:.2f} "
      f"({(dn-do)/do*100:+.0f}%)")

print("\n" + "=" * 74)
print("3. 부작용 — 클래스 보존과 출처 쏠림")
print("=" * 74)


def compose(root, split):
    d = os.path.join(root, split)
    coco = json.load(open(os.path.join(d, '_annotations.coco.json')))
    cname = {c['id']: c['name'] for c in coco['categories']}
    id2n = {im['id']: im['file_name'] for im in coco['images']}
    byimg = defaultdict(set)
    inst = Counter()
    for a in coco['annotations']:
        byimg[id2n[a['image_id']]].add(cname[a['category_id']])
        inst[cname[a['category_id']]] += 1
    per = Counter()
    for s in byimg.values():
        for c in s:
            per[c] += 1
    return len(coco['images']), per, inst


for split in ('train', 'valid'):
    nN, pN, iN = compose(NEW, split)
    nO, pO, iO = compose(OLD, split)
    print(f"\n[{split}]  신규 {nN}장 vs 기존 {nO}장")
    print(f"   {'클래스':<13}{'신규 이미지':>12}{'기존 이미지':>12}   "
          f"{'신규 인스턴스':>14}{'기존 인스턴스':>14}")
    for c in CLASSES:
        fN = pN[c] / max(nN, 1) * 100
        fO = pO[c] / max(nO, 1) * 100
        flag = ''
        if pN[c] == 0:
            flag = '  ❌ 없음'
        elif fN < fO * 0.6:
            flag = '  ⚠️ 비율 하락'
        print(f"   {c:<13}{pN[c]:>6} ({fN:>4.1f}%){pO[c]:>6} ({fO:>4.1f}%)"
              f"{iN[c]:>14,}{iO[c]:>14,}{flag}")

print("\n출처 구성 (train)")
for label, fs in (('신규 크래프트 200', names['train']),
                  ('기존 고정셋 400', set(oldtr))):
    c = Counter(origin(f) for f in fs)
    tot = sum(c.values())
    print(f"   {label}: " + ' · '.join(f"{k} {v}({v/tot*100:.0f}%)"
                                       for k, v in c.most_common()))

sz = {}
for label, root in (('신규', NEW), ('기존', OLD)):
    sz[label] = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(root) for f in fs) / 1024 / 1024
print(f"\n용량: 기존 {sz['기존']:.0f}MB → 신규 {sz['신규']:.0f}MB "
      f"({(sz['신규']-sz['기존'])/sz['기존']*100:+.0f}%)")

print("\n" + "=" * 74)
print(f"무결성 판정: {'✅ 통과' if ok else '❌ 실패'}")
print("⚠️ 학습 성능은 여기서 알 수 없다 — A/B 필요")
print("=" * 74)
sys.exit(0 if ok else 1)
