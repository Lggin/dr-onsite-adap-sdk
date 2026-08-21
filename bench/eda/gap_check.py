"""도메인 간극 수치 검증 — 7.6% 가 실제 간극인가 표본 구성 왜곡인가.

무작위(20%)보다 낮은 정확도는 보통 두 가지 중 하나다.
  (가) 학습·시험의 클래스 구성이 달라 다수 클래스로 쏠린다
  (나) 특징 분포가 실제로 어긋나 체계적으로 틀린 클래스를 고른다

클래스별 재현율과 지지수를 보면 갈린다. 두 도메인에 공통으로 존재하는 클래스만
남기고, 클래스별 개수를 맞춘 뒤 다시 잰다.
"""
import os, pickle, random, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.environ['S'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'grouping'))   # gridsearch_feat 는 그룹화 벤치에 있다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # origins
from origins import relabel, SYNTH, REAL as REALO, EXTERNAL
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

S = os.environ['S']
rng = random.Random(0)
D = pickle.load(open(S + '/allfeat_train.pkl', 'rb'))
F3 = D['F3'].astype(np.float64)
CLS = np.array([m[0] for m in D['meta']])
ORG = relabel(D['meta'])
Z = StandardScaler().fit_transform(F3)          # 전수로 표준화(표본 왜곡 제거)

SYN = np.isin(ORG, SYNTH)
REAL = np.isin(ORG, REALO)
EXT = ORG == '외부(roboflow)'
print("도메인 크기:", f"합성 {SYN.sum():,} / 실촬 {REAL.sum():,} / 외부 {EXT.sum():,}\n")
for nm, m in (('합성', SYN), ('실촬', REAL), ('외부', EXT)):
    c = Counter(CLS[m])
    tot = sum(c.values())
    print(f"  {nm:<4}" + '  '.join(f"{k} {v:,}({v/tot*100:.0f}%)" for k, v in c.most_common()))


def balanced(mask, per, seed=0):
    r = random.Random(seed)
    out = []
    for c in sorted(set(CLS[mask])):
        w = np.where(mask & (CLS == c))[0]
        if len(w) < per:
            continue
        out.extend(r.sample(list(w), per))
    return np.array(sorted(out))


print("\n① 클래스 구성을 맞추지 않은 채로 (앞선 측정 재현)")
for nm, tr, te in (('합성 → 실촬', SYN, REAL), ('합성 → 외부', SYN, EXT)):
    a = balanced(tr, 3000, 1)
    k = KNeighborsClassifier(1).fit(Z[a], CLS[a])
    pr = k.predict(Z[te])
    acc = (pr == CLS[te]).mean()
    print(f"\n  {nm}  전체 {acc*100:.1f}%")
    print(f"  {'실제':<13}{'개수':>7}{'재현율':>8}   가장 많이 틀린 예측")
    for c in sorted(set(CLS[te])):
        m = CLS[te] == c
        if m.sum() < 20:
            continue
        wrong = Counter(pr[m][pr[m] != c]).most_common(1)
        w = f"{wrong[0][0]} {wrong[0][1]/m.sum()*100:.0f}%" if wrong else '-'
        print(f"  {c:<13}{m.sum():>7,}{(pr[m]==c).mean()*100:>7.1f}%   {w}")

print("\n② 공통 클래스 + 클래스별 개수 일치 (구성 왜곡 제거)")
for nm, tr, te in (('합성 → 실촬', SYN, REAL), ('실촬 → 합성', REAL, SYN),
                   ('합성 → 외부', SYN, EXT), ('외부 → 합성', EXT, SYN)):
    common = sorted(set(CLS[tr]) & set(CLS[te]))
    per_te = min(min((CLS[te] == c).sum() for c in common), 800)
    per_tr = min(min((CLS[tr] == c).sum() for c in common), 2000)
    if per_te < 30:
        print(f"  {nm:<12} 공통 클래스 표본 부족 (min {per_te})")
        continue
    a = balanced(tr & np.isin(CLS, common), per_tr, 1)
    b = balanced(te & np.isin(CLS, common), per_te, 2)
    k = KNeighborsClassifier(1).fit(Z[a], CLS[a])
    acc = (k.predict(Z[b]) == CLS[b]).mean()
    # 같은 도메인 내부 기준선
    b2 = balanced(tr & np.isin(CLS, common), per_te, 7)
    b2 = np.array([i for i in b2 if i not in set(a)])
    base = (k.predict(Z[b2]) == CLS[b2]).mean() if len(b2) > 30 else float('nan')
    print(f"  {nm:<12}{acc*100:>6.1f}%   (같은 도메인 기준선 {base*100:.1f}%, "
          f"무작위 {100/len(common):.0f}%, 클래스 {len(common)}개×{per_te})")

print("\n③ 클래스를 고정하고 도메인만 맞히기 — 간극의 직접 측정")
print("   (Carton 만. 정확도가 높을수록 합성과 실촬이 특징으로 구분된다 = 간극이 크다)")
for nm, m in (('합성 vs 실촬', SYN | REAL), ('합성 vs 외부', SYN | EXT),
              ('실촬 vs 외부', REAL | EXT)):
    sel = np.where(m & (CLS == 'Carton'))[0]
    dom = np.where(SYN[sel], '합성', np.where(REAL[sel], '실촬', '외부'))
    per = min(Counter(dom).values())
    per = min(per, 3000)
    take = []
    for d in sorted(set(dom)):
        w = sel[dom == d]
        take.extend(rng.sample(list(w), per))
    take = np.array(take)
    dd = np.where(SYN[take], '합성', np.where(REAL[take], '실촬', '외부'))
    cut = rng.sample(range(len(take)), len(take) // 2)
    tr = np.zeros(len(take), bool); tr[cut] = True
    k = KNeighborsClassifier(5).fit(Z[take[tr]], dd[tr])
    print(f"   {nm:<12}{(k.predict(Z[take[~tr]])==dd[~tr]).mean()*100:>6.1f}%  "
          f"(무작위 50%, 각 {per:,})")
