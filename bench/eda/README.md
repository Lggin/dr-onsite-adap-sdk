# bench/eda — 학습 데이터셋 EDA

"우리가 학습에 쓰는 데이터가 실제로 어떻게 생겼나"를 재는 스크립트 모음.
2026-08-18 표본(Carton 6,000) 으로 시작해 08-21 **전수 377,914 객체**로 확장했다.

**팀 공유 문서**: `~/공유_데이터셋EDA_및_그룹화성능_20260821.{md,html}`
**인터랙티브 3D**: `~/데이터셋_3D산점도.html`

---

## 핵심 결과

| | 결과 |
|---|---|
| 대상 | `final_dataset_ultimate/train` — 라벨 379,190 중 **377,914 객체**에서 특징 추출 |
| 클래스 불균형 | **223:1** (Carton 339,844 / Slip Sheet 1,533). 원인은 이미지 수가 아니라 **객체 밀도** |
| 클래스 분리 가능성 | 14차원 색·질감·크기 + 1-NN 5겹 CV **92.2%** (무작위 20%) |
| 가장 강한 축 | `size_short` F=5,547 → `size_long` 4,118 → `solidity` 2,067. **크기가 압도적** |
| 도메인 간극 | 같은 도메인 98.5% → **합성→실촬 44.8%**. Carton 만 놓고 도메인을 맞히면 **99.6%** |
| 학습 공백 | Slip Sheet 절차생성 합성 **0**. Shrink Wrap 합성 8,266은 **전부 단품 렌더**(회색 배경, 팔레트 없음) |
| Carton 내부 구조 | 뚜렷한 최적 k 없음(실루엣 0.16~0.20) = 저수준 통계로는 품목이 **연속적** |

---

## 함정 두 개 (다시 밟지 않기)

### 1. `render_*` 출처 오판정

`extract_all.py` 의 초기 `origin()` 이 `render_*.jpg` 8,266개를 '기타'로 떨어뜨렸다.
실제로 열어 보니 **슈링크랩 단품 합성 렌더**였다. 지난 문서의 "기타 17.8%"가 이것이다.
→ 판정은 `origins.py` 를 **단일 소스**로 쓴다. meta 에 파일명이 있으므로 전수 재추출
(19분) 없이 사후 재판정하면 된다.

### 2. 무작위보다 낮은 정확도는 대개 구성 왜곡

"합성으로 학습 → 실촬 채점"이 처음 **7.6%**(무작위 20% 미만)로 나왔다. 원인은 간극이
아니라 **클래스 구성 차이**였다(합성은 Carton 92%, 실촬은 Shrink Wrap 42%. 게다가 당시엔
합성 쪽에 Shrink Wrap 이 아예 없는 것으로 잘못 집계됨).

→ `gap_check.py` 처럼 **공통 클래스만 남기고 클래스별 개수를 맞춘 뒤** 재고, 반드시
**같은 도메인 기준선**을 함께 찍는다. 그러면 44.8% vs 98.5% 라는 해석 가능한 수가 나온다.

---

## 실행

```bash
export S=~/eda_work            # 중간 산출물(.pkl) 저장 위치
mkdir -p $S
cp <운영가중치>.pth $S/continual_master_operational.pth   # box_detect 계열이 쓴다

python3 bench/eda/eda_datasets.py    # ① 데이터셋 12개 계보·중복·심볼릭링크 실사
python3 bench/eda/extract_all.py     # ② 전수 특징 추출 → allfeat_train.pkl (약 19분)
python3 bench/eda/eda_more.py        # ③ 클래스 밀도·크기·valid split 점검
python3 bench/eda/scatter_stats.py   # ④ 분리 가능성·축별 F·도메인 간극·현장 위치
python3 bench/eda/gap_check.py       # ⑤ 도메인 간극 검증(구성 왜곡 제거)
python3 bench/eda/scatter3d.py       # ⑥ 인터랙티브 3D HTML
python3 bench/eda/share_figs.py      # ⑦ 발표용 정적 4패널 PNG
python3 bench/eda/md2html.py <md> <png...>   # ⑧ 컨플루언스용 단일 HTML
```

`extract_all.py` 는 `bench/grouping/gridsearch_feat.py` 의 `f2_mask`/`f3_rich` 를 그대로
쓴다 — 그룹화 실험과 **같은 특징 정의**를 유지하려는 것이다. 경로는 각 스크립트가
자동으로 잡는다.

`eda_build.py` / `eda_analyze.py` / `eda_fiftyone.py` 는 표본(Carton 6,000) 시절 경로다.
FiftyOne 뷰어는 객체 크롭을 **개별 샘플**로 만든다(원본 프레임을 샘플로 쓰면 한 장에
수십 개가 섞여 클릭해도 어떤 객체인지 알 수 없다).

---

## 산출물

| 파일 | 내용 |
|---|---|
| `$S/allfeat_train.pkl` | 전수 특징 37MB — F2(5차원) / F3(14차원) / meta(클래스·출처·파일명·px) |
| `~/데이터셋_3D산점도.html` | 인터랙티브 3D, 관점 5개, 표본 32,517 |
| `~/eda_요약_4패널.png` | 출처×클래스 / 도메인 간극 / 축별 F / 현장 8종 위치 |

---

## 한계

- **크기 축이 픽셀이다.** 전수 데이터에 depth 가 없어 물리 mm 를 쓸 수 없다.
  현장 데이터를 겹칠 때는 반드시 `size_mm=None` 으로 다시 계산해 좌표계를 맞춰야 한다
- **운용 그룹화와 다른 특징 공간이다.** 운용은 윗면 3D 워프 + CNN 임베딩이다.
  여기서 보이는 구조를 그룹화 성능으로 읽으면 안 된다
- **PCA 3축이 55%밖에 못 잡는다.** 14축이 상당히 독립적이라는 뜻이므로 3D 그림은
  구조의 요약일 뿐이다
- **도메인 간극 99.6% 는 저수준 통계 기준**이다. RFDETR 이 보는 표현이 아니므로
  "검출 성능 간극"으로 옮겨 읽으면 안 된다
