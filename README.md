# dr-onsite-adap-sdk

디팔레타이징(상자 내리기) 로봇을 위한 **현장 적응형 비전 SDK**입니다.
팔레트 위를 내려다보는 RGB-D 카메라로 각 물체의 **윗면 중심과 장/단축 방향(3D)**
— 즉 흡착 그리퍼의 픽 좌표/자세 — 를 계산하고, 팔레트가 비면 알려주며,
현장에서 어려웠던 케이스를 모아 **검출 모델을 스스로 개선(액티브러닝)** 합니다.

크게 두 부분으로 나뉩니다.

- **`dr_onsite_adap.runtime`** — 라이브 RGB-D 검출(RF-DETR 세그멘테이션) +
  윗면 픽 좌표 추정 + 가려짐 필터 + SAM 라벨링 + 화면 표시(2D) + 3D 스냅샷.
- **`dr_onsite_adap.pipeline`** — 액티브러닝 학습 루프: 라벨 수집 → 학습셋 구성 →
  LoRA 미세조정 → 합치기 → 평가.

---

## 원리 한눈에

**윗면 추정 (`runtime/geometry.py`).** 물체마다: 마스크 외곽선을 따고, 픽셀별
"월드 수직 방향 높이"의 절삭평균(trimmed median)으로 윗면 층을 고른 뒤(카메라가
기울어도 불변), 수평면을 만들어 외곽선을 그 면에 투영하고, **평면 좌표에서**
`minAreaRect` 를 돌립니다(원근 왜곡이 자동 보정됨). 마지막에 사각형을 3D로 되올려
중심·장축·단축·변 길이(mm)를 냅니다. 합성 기울임 테스트에서 sub-mm 정확도.

**액티브러닝 (`pipeline/`).** 현장에서 어려운 케이스를 라이브 앱('t')의 SAM으로
라벨 → COCO 데이터셋에 추가 → **팔레트가 든 합성 데이터(리플레이)** 와 섞어 LoRA를
한 라운드 돌립니다. 이 리플레이 덕분에 적은 현장 데이터로도 새 외형을 배우면서
기존 능력(예: 팔레트)을 잊지 않습니다.

---

## 설치

```bash
pip install -e ".[label,train,orbbec]"     # RealSense 카메라면 [orbbec] 대신 [realsense]
```

카메라 SDK는 선택 설치입니다(쓰는 기종만). `pyorbbecsdk` 는 PyPI 휠이 없을 수 있는데,
그럴 땐 <https://github.com/orbbec/pyorbbecsdk> 에서 빌드하세요.

### Jetson(Thor 등 엣지) 설치 — 주의
JetPack 환경은 **그 환경의 torch(예: 2.12+cuXX)를 그대로 써야** 합니다. 그래서 SDK는
의존성을 건드리지 않게 `--no-deps` 로 설치하고, 학습용 패키지는 따로 설치하세요:

```bash
pip install -e . --no-deps                       # SDK 본체 (torch 안 건드림)
pip install peft pytorch-lightning pycocotools faster-coco-eval   # 학습용
```

`pycocotools`/`faster-coco-eval` 는 rfdetr 의 COCO 데이터 로더와 mAP 평가에 필요합니다
(추론만 할 거면 불필요). 검증 결과 Thor(aarch64, CUDA 13)에서 LoRA 학습 + merge 가
정상 동작합니다.

---

## 설정 (환경변수)

코드에 절대경로를 박지 않고 **환경변수로** 지정합니다. 기본값은 `DEPAL_HOME`(기본=
현재 폴더) 기준 상대경로입니다. 자주 쓰는 것:

| 환경변수 | 의미 |
|---|---|
| `DEPAL_HOME` | 기본 경로들의 기준 폴더 |
| `DEPAL_WEIGHTS` | 검출 가중치 `.pth` (실행용) |
| `DEPAL_NUM_CLASSES` | 가중치의 클래스 수 (기본 4) |
| `DEPAL_DEVICE` | `cpu` 또는 `cuda` |
| `DEPAL_SAM_WEIGHTS` | SAM 체크포인트 (라벨링용) |
| `DEPAL_DATA_DIR` | 액티브러닝 COCO 데이터셋 폴더 |
| `DEPAL_CAMERA_BACKEND` | `auto` / `realsense` / `orbbec` |
| `DEPAL_CAMERA_SERIAL` | (Orbbec 여러 대 연결 시) 쓸 카메라 시리얼. 비우면 첫 장치(device[0]) |
| `DEPAL_UP_DIR` | 카메라 기준 월드 수직 방향, 예: `-0.005,-0.428,-0.904` (IMU 없는 고정 마운트용) |
| `DEPAL_Z_MAX` | 유효 depth 최대값(m). 높이 단 마운트면 키우세요 |

학습용 경로: `DEPAL_BASE_WEIGHTS`, `DEPAL_TRAIN_DATA`, `DEPAL_LORA_SAFE`,
`DEPAL_LORA_RUN`, `DEPAL_FINAL`, `DEPAL_PALLET_DATA`, `DEPAL_REPLAY_OUT`.
전체 목록과 기본값은 `src/dr_onsite_adap/config.py` 참고.

---

## 실행 — 라이브 앱

```bash
DEPAL_WEIGHTS=/경로/model.pth DEPAL_DEVICE=cuda dr-onsite-run
```

키: `d` 검출(다시) · `v` 3D 보기 · `c` 지우기 · `r` 관심영역(ROI) 지정/해제 ·
`t` SAM 라벨 · `q` 종료. 팔레트는 보이는데 픽 대상이 0개면 화면에
**PALLET EMPTY — DONE** 배너가 뜹니다.

### 데스크톱 GUI (권장, 처음 보는 사람용)
키보드 대신 **버튼·카메라 드롭다운·검출 목록 표**가 있는 창입니다. PySide6 필요:

```bash
pip install "dr-onsite-adap[gui]"          # PySide6 설치 (Jetson은 휠 있음)
DEPAL_WEIGHTS=/경로/model.pth DEPAL_DEVICE=cuda dr-onsite-gui
```

사이드바: 검출 · 3D 보기 · ROI 지정/해제(영상 위 드래그) · 라벨링(SAM) · 중력 재측정 · 지우기.

> **중력 재측정(🧭)**: 카메라 마운트 각도가 바뀌면 윗면 추정에 쓰는 `up_dir`(월드 수직)
> 이 틀어집니다. 이 버튼은 현재 depth 의 **지배 수평면(바닥/팔레트) 법선**으로 up_dir 을
> 다시 측정해 `up.json`(`DEPAL_UP_PATH`)에 저장합니다 — IMU 없이, 다음 실행에도 자동 적용.
> 우선순위: IMU > 측정값(up.json) > `DEPAL_UP_DIR`.
카메라가 여러 대면 드롭다운에서 시리얼로 선택. 모델 로딩·카메라·검출은 백그라운드라
창이 멈추지 않습니다. (가벼운 키보드 버전은 위 `dr-onsite-run`.)

탭 3개:
- **실시간 인식** — 영상 + 검출 목록 표
- **라벨 데이터** — 't'로 저장한 라벨 이미지 썸네일(라벨 후 자동 갱신)
- **학습** — 파이프라인(취합/리플레이/학습/병합/전체 라운드)을 버튼으로 실행하고
  **로그를 실시간으로** 봅니다. 학습은 서브프로세스라 창은 안 멈추지만, **같은 GPU를
  공유**하므로 학습 중엔 실시간 검출이 느려집니다. 경로는 환경변수(`DEPAL_BASE_WEIGHTS`,
  `DEPAL_TRAIN_DATA`, `DEPAL_FINAL` 등)로 지정합니다.

---

## 학습 — 액티브러닝 한 라운드

```bash
# 0. (처음 한 번) 팔레트 라벨 합성 데이터셋 만들고 + 카테고리 선언
dr-onsite-dataset build-pallet --src-base <실측COCO> --synth <합성1> <합성2> --out $DEPAL_PALLET_DATA
dr-onsite-dataset declare-pallet --data-dir $DEPAL_DATA_DIR

# 1. 라이브 앱 't' 로 모은 현장 라벨을 데이터셋에 추가
dr-onsite-ingest

# 2. 학습셋 구성(실측 + 팔레트 든 합성 리플레이)
dr-onsite-replay

# 3. 라운드 실행(LoRA 학습 → 합치기) 후 평가
DEPAL_TRAIN_DATA=$DEPAL_REPLAY_OUT dr-onsite-round --replay
dr-onsite-eval --models 이전=/경로/old.pth 새것=$DEPAL_FINAL --data $DEPAL_PALLET_DATA/valid

# 4. 배포: 현장 PC에서 DEPAL_WEIGHTS 를 $DEPAL_FINAL 로 지정.
#    다음 라운드: DEPAL_BASE_WEIGHTS 를 이 결과로 올리세요.
```

`dr-onsite-round` 는 학습셋에 팔레트 라벨이 0개면 멈춥니다(팔레트 없는 배치로
학습하면 팔레트 인식이 약해질 수 있음). `--replay` 로 팔레트 든 합성을 섞거나
`--force` 로 강행하세요.

각 명령은 `--help` 로 한국어 설명을 볼 수 있습니다 (예: `dr-onsite-replay --help`).

---

## 폴더 구조

```
src/dr_onsite_adap/
├── config.py            환경변수 기반 설정(경로·클래스·기하·하이퍼파라미터)
├── runtime/
│   ├── app.py           라이브 루프 + 키 입력
│   ├── detect.py        검출 실행 / 화면 표시
│   ├── geometry.py      윗면 추정(평면좌표 minAreaRect)
│   ├── occlusion.py     자기 외곽선 기반 가려짐 필터
│   ├── inference.py     RF-DETR 세그멘테이션 래퍼
│   ├── viz.py viz3d.py  2D 표시 + 3D 스냅샷
│   ├── labeler.py       내장 SAM 라벨링
│   └── cameras/         realsense + orbbec 백엔드, 자동 감지
└── pipeline/
    ├── ingest.py        SAM 라벨 → COCO (덮어쓰지 않고 추가)
    ├── replay.py        리플레이 혼합 학습셋 생성
    ├── train_lora.py    LoRA 미세조정
    ├── merge.py         어댑터 합쳐 단일 .pth
    ├── eval.py          클래스별 AP@0.5 비교
    ├── dataset.py       build-pallet / declare-pallet 도구
    └── round.py         라운드 자동 실행(리플레이 → 학습 → 합치기)
```

> 참고: import 패키지명은 `dr_onsite_adap`, CLI 명령은 `dr-onsite-*` 입니다.
> 모델 가중치(`.pth`)와 데이터셋은 git 에 올라가지 않습니다(.gitignore).
