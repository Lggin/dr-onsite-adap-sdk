"""final_dataset_* 12개 계보 EDA — 구성·중복·클래스 변천·정리 여지."""
import json, os, glob, hashlib, pickle
from collections import Counter, defaultdict

ROOT = os.path.expanduser('~/lee3/thirdtask')
S = os.environ['S']
OUT = {}

def load(p):
    with open(p) as f:
        return json.load(f)

# 이미지 파일명은 출처를 담고 있다 — 접두어로 소스 계열을 분류한다.
def origin(fn):
    f = fn.lower()
    if f.startswith('sam_hard'):        return '현장 실촬'
    if f.startswith('lee5synth') or 'synth' in f:  return '합성(Isaac)'
    if f.startswith('rp_') :            return '리플레이링크'
    if 'roboflow' in f or '.rf.' in f:  return '외부(roboflow)'
    if f.startswith('scd') or 'scd_' in f:  return 'SCD'
    if f.startswith('render') or f.startswith('blend'): return '합성(렌더)'
    return '기타'

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, 'final_dataset_*'))):
    name = os.path.basename(d)
    rec = {'name': name, 'splits': {}}
    for sp in ('train', 'valid'):
        p = os.path.join(d, sp, '_annotations.coco.json')
        if not os.path.exists(p):
            continue
        c = load(p)
        names = [im['file_name'] for im in c['images']]
        cats = {x['id']: x['name'] for x in c['categories']}
        ann_by_img = defaultdict(list)
        for a in c['annotations']:
            ann_by_img[a['image_id']].append(a)
        inst = Counter(cats.get(a['category_id'], a['category_id']) for a in c['annotations'])
        img_with = Counter()
        for im in c['images']:
            for cid in {a['category_id'] for a in ann_by_img[im['id']]}:
                img_with[cats.get(cid, cid)] += 1
        rec['splits'][sp] = dict(
            n_img=len(c['images']), n_ann=len(c['annotations']),
            uniq=len(set(names)), cats=[cats[k] for k in sorted(cats)],
            inst=dict(inst), img_with=dict(img_with),
            origin=dict(Counter(origin(n) for n in names)),
            names=set(names),
            per_img=[len(ann_by_img[im['id']]) for im in c['images']],
        )
    if rec['splits']:
        rows.append(rec)

# ── 1. 기본 구성 ──────────────────────────────────────────────────────────
print("=" * 108)
print("① 데이터셋 구성")
print(f"{'데이터셋':<30}{'train':>8}{'고유':>8}{'중복':>7}{'라벨':>9}"
      f"{'장당':>6}{'valid':>7}{'클래스':>4}  용량")
for r in rows:
    t = r['splits'].get('train'); v = r['splits'].get('valid')
    if not t:
        continue
    dup = t['n_img'] - t['uniq']
    sz = os.popen(f"du -sh {ROOT}/{r['name']} 2>/dev/null | cut -f1").read().strip()
    avg = sum(t['per_img']) / max(len(t['per_img']), 1)
    print(f"{r['name']:<30}{t['n_img']:>8,}{t['uniq']:>8,}{dup:>7,}{t['n_ann']:>9,}"
          f"{avg:>6.1f}{(v['n_img'] if v else 0):>7,}{len(t['cats']):>4}  {sz}")

# ── 2. 클래스 구성 변천 ───────────────────────────────────────────────────
print("\n" + "=" * 108)
print("② 클래스 구성 (순서 = category_id)")
seen = {}
for r in rows:
    t = r['splits'].get('train')
    if not t:
        continue
    key = tuple(t['cats'])
    seen.setdefault(key, []).append(r['name'])
for k, v in seen.items():
    print(f"  {' / '.join(k)}")
    print(f"      → {', '.join(v)}")

# ── 3. 출처 구성 ──────────────────────────────────────────────────────────
print("\n" + "=" * 108)
print("③ train 이미지 출처 구성 (%)")
keys = ['현장 실촬', '합성(Isaac)', '합성(렌더)', '외부(roboflow)', 'SCD', '기타']
print(f"{'데이터셋':<30}" + "".join(f"{k[:9]:>12}" for k in keys))
for r in rows:
    t = r['splits'].get('train')
    if not t:
        continue
    n = t['n_img']
    print(f"{r['name']:<30}" + "".join(
        f"{t['origin'].get(k, 0)/n*100:>11.1f}%" for k in keys))

# ── 4. 클래스 불균형 ──────────────────────────────────────────────────────
print("\n" + "=" * 108)
print("④ train 클래스별 인스턴스 (괄호=포함 이미지 비율)")
allc = []
for r in rows:
    t = r['splits'].get('train')
    if t:
        allc += t['cats']
for r in rows:
    t = r['splits'].get('train')
    if not t:
        continue
    parts = []
    for c in t['cats']:
        i = t['inst'].get(c, 0); w = t['img_with'].get(c, 0)
        parts.append(f"{c[:9]} {i:,}({w/t['n_img']*100:.0f}%)")
    print(f"  {r['name']:<30}{' | '.join(parts)}")

# ── 5. 계보(파일명 겹침) ──────────────────────────────────────────────────
print("\n" + "=" * 108)
print("⑤ 계보 — train 파일명 겹침 (행 기준 %: 행의 몇 %가 열에도 있나)")
ns = [(r['name'], r['splits']['train']['names']) for r in rows if 'train' in r['splits']]
short = [n.replace('final_dataset_', '')[:11] for n, _ in ns]
print(f"{'':<13}" + "".join(f"{s:>12}" for s in short))
for (na, sa), la in zip(ns, short):
    line = f"{la:<13}"
    for nb, sb in ns:
        line += f"{len(sa & sb)/max(len(sa),1)*100:>11.0f}%"
    print(line)

pickle.dump(rows, open(S + '/eda_rows.pkl', 'wb'))
print("\n저장:", S + '/eda_rows.pkl')
