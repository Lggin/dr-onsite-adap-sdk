"""실제 조명 변화로 거리(7×7) 학습 — 처음 보는 종류에서 평가.

어제 M 은 HSV 합성 변형으로 배워 실제 조명에서 오히려 손해였다(83.8% → 78.5%).
이번엔 **같은 종류 안의 실제 변동**(조명 on/off + 개체차)을 잡음으로, **종류 간 차이**를
신호로 삼는다(Fisher 판별 관점).

    S_W = 종류 내 산포(잡음: 조명·개체차)      S_B = 종류 간 산포(신호)
    M   = (S_W + λI)^(−1/2)                    → 잡음 방향을 눌러 정규화

⚠️ 방법론: **학습에 쓴 종류로 평가하면 무의미**하다. 종류를 반으로 갈라 3종으로 배우고
   나머지 3종(처음 보는 박스)에서 잰다. 운영도 "거리는 한 번 배우고 새 박스는 등록만"
   이므로 이 시험이 실제 상황과 같다.
"""
import os, pickle
import numpy as np
os.environ.setdefault("DEPAL_NUM_CLASSES", "5")
from dr_onsite_adap import config

S = os.environ.get('GROUPING_WORK', os.path.expanduser('~/grouping_work'))
Z = np.load(S + '/box_Z.npy')
is_reg = np.load(S + '/box_isreg.npy')
meta = pickle.load(open(S + '/box_meta.pkl', 'rb'))
LAB, types = meta['LAB'], meta['types']
MATCH_TAU = config.CLUSTER_MATCH_TAU
AX = ['L', 'a', 'b', 'tex1', 'tex2', 'size_long', 'size_short']


def learn(train_types, lam=0.10):
    """train_types 안의 종류 내 산포로 화이트닝 행렬 M 을 만든다."""
    devs = []
    for t in train_types:
        X = Z[LAB == t]
        devs.append(X - X.mean(0))
    D = np.concatenate(devs)
    S_W = D.T @ D / max(len(D) - len(train_types), 1)
    S_W = S_W + lam * np.trace(S_W) / S_W.shape[0] * np.eye(S_W.shape[0])
    w, V = np.linalg.eigh(S_W)
    M = V @ np.diag(w ** -0.5) @ V.T
    # 거리 눈금 보정 — MATCH_TAU 의미 유지
    rng = np.random.RandomState(0)
    idx = rng.choice(len(Z), size=min(400, len(Z)), replace=False)
    db = np.linalg.norm(Z[idx][:, None] - Z[idx][None], axis=2)
    Za = Z[idx] @ M.T
    da = np.linalg.norm(Za[:, None] - Za[None], axis=2)
    return M * (np.median(db[db > 0]) / np.median(da[da > 0]))


def evaluate(Zx, subset=None, quiet=True):
    """밝은 프레임 medoid 등록 → 어두운 프레임 배정. subset 지정 시 그 종류만 채점."""
    P, PL = [], []
    for t in types:
        m = np.where(is_reg & (LAB == t))[0]
        c = Zx[m].mean(0)
        P.append(Zx[m][int(np.argmin(np.linalg.norm(Zx[m] - c, axis=1)))]); PL.append(t)
    P = np.stack(P); PL = np.array(PL)
    te = np.where(~is_reg)[0]
    if subset is not None:
        te = te[np.isin(LAB[te], subset)]
    d = np.linalg.norm(Zx[te][:, None] - P[None], axis=2)
    j = d.argmin(1)
    pred = np.where(d.min(1) < MATCH_TAU, PL[j], '기타')
    acc = float((pred == LAB[te]).mean())
    if not quiet:
        for t in (subset or types):
            sel = LAB[te] == t
            if sel.any():
                p = pred[sel]
                wrong = [(x, int((p == x).sum())) for x in set(p) if x != t]
                wrong.sort(key=lambda z: -z[1])
                print(f"    {t:<16}{sel.sum():>4}{(p==t).mean()*100:>7.0f}%   "
                      f"{', '.join(f'{a}:{b}' for a, b in wrong[:2])}")
    return acc


HALF_A = ['blue_printed', 'navy_long', 'pacific_short']
HALF_B = ['navy_small', 'pacific_long', 'swanson']

print(f"기준선(고정 가중) 전체 {evaluate(Z)*100:.1f}%")
print(f"  A조 {evaluate(Z, HALF_A)*100:.1f}%   B조 {evaluate(Z, HALF_B)*100:.1f}%\n")

print("── 교차 검증: 3종으로 배우고 나머지 3종(처음 보는 박스)에서 평가 ──")
unseen = []
for train, test, nm in ((HALF_A, HALF_B, 'A→B'), (HALF_B, HALF_A, 'B→A')):
    M = learn(train)
    a_seen = evaluate(Z @ M.T, train)
    a_unseen = evaluate(Z @ M.T, test)
    base = evaluate(Z, test)
    unseen.append((a_unseen, base))
    print(f"\n[{nm}] 학습={train}")
    print(f"   학습에 쓴 종류  {a_seen*100:5.1f}%")
    print(f"   **처음 보는 종류 {a_unseen*100:5.1f}%**  (고정 가중 {base*100:.1f}%, "
          f"{(a_unseen-base)*100:+.1f}%p)")
    evaluate(Z @ M.T, test, quiet=False)

mu = np.mean([u for u, _ in unseen]); mb = np.mean([b for _, b in unseen])
print(f"\n처음 보는 종류 평균: 학습된 거리 {mu*100:.1f}%  vs  고정 가중 {mb*100:.1f}%"
      f"  ({(mu-mb)*100:+.1f}%p)")

M_all = learn(types)
print(f"\n(참고) 전 종류로 학습 후 전체 평가 {evaluate(Z @ M_all.T)*100:.1f}% — 낙관적 상한")
print("\n전 종류 학습 시 축별 실효 가중:")
for nm, v in zip(AX, np.linalg.norm(M_all, axis=1)):
    print(f"   {nm:<12}{v:6.3f}")
np.save(S + '/metric_real.npy', M_all)
