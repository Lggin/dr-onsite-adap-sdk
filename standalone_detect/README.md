# depal_detect — 디팔레타이징 검출 모듈 (독립 파일)

박스 / 토트 / 슈링크랩 / 팔레트를 검출하는 **단일 파일 facade**입니다.
모델 로드·전처리·임계·클래스 설정을 한 곳에 묶어서, **누가 어느 코드에서 쓰든 동일한 검출 결과**가 나오도록 했습니다.
("같은 가중치인데 코드마다 검출율이 다른" 문제 방지)

> 이 `depal_detect.py` 는 우리 SDK(`dr_onsite_adap`)를 **import 하지 않습니다.** 공개 라이브러리(torch/rfdetr/opencv/numpy)만 쓰므로 **SDK 설치 없이** 이 파일 하나만 복사해 쓰면 됩니다.

---

## 1. 준비물

| 구성 | 받는 곳 |
|---|---|
| `depal_detect.py` | 이 레포(GitHub) — `standalone_detect/` |
| `README.md` (이 문서) | 이 레포(GitHub) |
| `continual_master_v9_full.pth` (가중치, ~141MB) | **담당자에게 전달받기** (대용량이라 git 미포함) |

## 2. 설치

```bash
pip install torch rfdetr opencv-python numpy
```
- GPU(CUDA) 권장. 없으면 자동으로 CPU 로 동작(느림).

## 3. 빠른 시작

### 파이썬
```python
from depal_detect import Detector

det = Detector("continual_master_v9_full.pth")   # device 자동(cuda 우선)

results = det.detect("frame.jpg")                 # 경로 / ndarray 모두 가능
for r in results:
    print(r.class_name, round(r.confidence, 2), r.bbox)

# 시각화 저장
import cv2
cv2.imwrite("out.jpg", det.draw("frame.jpg", results))
```

### CLI
```bash
python depal_detect.py frame.jpg --weights continual_master_v9_full.pth --threshold 0.5 --save out.jpg
```

## 4. 입력 형식 (★ 중요)

`detect(image, threshold=0.5, is_bgr=True)`

| 입력 | 사용법 |
|---|---|
| 파일 경로 | `det.detect("frame.jpg")` |
| `cv2.imread()` 결과 (**BGR**) | `det.detect(img)` — 기본 `is_bgr=True` |
| PIL / matplotlib 등 (**RGB**) | `det.detect(img, is_bgr=False)` |

> cv2 로 읽은 이미지는 BGR 입니다. **그대로 넣으면 됩니다(내부에서 RGB 변환)**.
> RGB 배열인데 `is_bgr=True`(기본) 로 두면 색이 뒤집혀 검출율이 떨어지니, 입력 채널 순서만 맞춰 주세요.

## 5. 출력 형식

`detect()` 는 `Detection` 리스트를 반환합니다:

```python
@dataclass
class Detection:
    class_id:   int                       # 0~4
    class_name: str                       # "Pallet" / "Slip Sheet" / "Carton" / "Tote" / "Shrink Wrap"
    confidence: float                     # 0~1
    bbox:       tuple[int,int,int,int]     # (x1, y1, x2, y2) 픽셀
    mask:       np.ndarray | None          # HxW bool (세그멘테이션 마스크)
```

### 클래스 ID
| id | 이름 |
|----|------|
| 0 | Pallet (팔레트) |
| 1 | Slip Sheet (슬립시트) |
| 2 | Carton (종이박스) |
| 3 | Tote (토트박스) |
| 4 | Shrink Wrap (슈링크랩) |

## 6. 검출율이 낮게 나올 때 (체크리스트)

가중치는 동일하므로, 결과가 다르면 **호출 방식**이 원인입니다. 이 모듈을 그대로 쓰면 아래가 전부 처리되지만, 직접 rfdetr 을 호출하던 코드라면 확인하세요:

1. **`num_classes=5`** 로 로드했는가 — 안 맞으면 분류 헤드가 어긋나 검출이 망가집니다. (이 모듈 기본값 5)
2. **이미지를 RGB 로** 넣었는가 — cv2 의 BGR 을 그대로 모델에 넣으면 안 됩니다. (이 모듈이 자동 변환)
3. **임계 인자명이 `threshold=`** 인가 — rfdetr 은 `conf_threshold=` 를 무시하고 기본 0.5 를 씁니다. (이 모듈이 `threshold=` 로 전달)
4. **입력 해상도** — 학습 해상도와 크게 다른 크기/비율로 넣으면 객체 스케일이 달라져 손해. 가능하면 원본 비율 유지.

## 7. 파라미터

```python
Detector(weights, num_classes=5, device=None, resolution=None)
#   device     : "cuda" / "cpu" / None(자동)
#   resolution : 입력 해상도 override. None=모델 기본(권장). 학습과 다르게 주지 말 것.

det.detect(image, threshold=0.5, is_bgr=True)
det.draw(image, detections, is_bgr=True)   # 검출 그린 BGR 이미지 반환
```
