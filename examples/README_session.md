# OnsiteSession — 고수준 SDK (UI 개발팀용)

UI 를 올리는 팀을 위한 **high-level API**. SDK 가 **카메라 + 검출/픽자세 루프를 소유**하고
백그라운드 스레드에서 돌리며, 결과를 **콜백(push)** 과 **폴링(pull)** 양쪽으로 준다.
파이프라인(카메라·전처리·검출·기하·스무딩)을 다시 배선할 필요 없이 **이벤트와 제어
메서드에 바인딩만** 하면 된다. UI 툴킷 비종속(Qt/웹/ROS 무관, 내부에 cv2 창·PySide 없음).

> 프레임만 직접 넣고 결과만 받고 싶으면 저수준 `OnsiteDetector.process()` 도 그대로 있음.
> 이 문서는 카메라+루프까지 SDK 가 맡는 **고수준** `OnsiteSession` 설명이다.

---

## 1. 설치 / 준비
```bash
pip install -e .            # 또는 사내 배포 방식
# 가중치(continual_master_v11.pth)는 담당자에게 받아 경로 지정
```

## 2. 빠른 시작
```python
from dr_onsite_adap import OnsiteSession

sess = OnsiteSession(weights="continual_master_v11.pth", geom_algo="v2")

@sess.on_result                       # PUSH: 프레임마다 호출
def _(scene):
    for p in scene.pick_targets:
        print(p.class_name, p.center_3d, p.long_side_mm, p.short_side_mm)
    if scene.done:
        print("빈 팔레트 — 작업 완료")

sess.start()                          # 카메라+추론 스레드 시작
...
scene = sess.latest()                 # PULL: 최신 결과(스레드세이프, None 가능)
img   = sess.draw_latest()            # 시각화 BGR ndarray (UI 에 표시)
sess.stop()                           # with OnsiteSession(...) as sess: 도 가능
```
바로 돌려보는 예제: [`session_quickstart.py`](session_quickstart.py)

## 3. ★ 카메라 없이 실행 (소스 주입)
`OnsiteSession` 은 카메라를 직접 다루지 않고 `read()` 를 가진 **소스 객체**에서 프레임을 받는다.
- `source=None`(기본) → SDK 가 실제 카메라를 연다.
- `source=<객체>` → 그 객체의 `read()` 를 대신 호출 → **카메라 없이 동일 동작**.

저장된 RGB-D 캡처(`rgbd_*.npz`)를 재생하는 `ReplaySource` 제공:
```python
from dr_onsite_adap.runtime.session import ReplaySource
import glob
src = ReplaySource(sorted(glob.glob("/path/rgbd_*.npz")), loop=True)
sess = OnsiteSession(weights="...", source=src)   # 카메라 불필요
```
**커스텀 소스(ROS 등)** 도 같은 계약만 지키면 끼울 수 있다:
```
read() -> (color_bgr[HxWx3 uint8], depth_m[HxW float, 미터],
           intrinsics(fx,fy,cx,cy), up_dir(x,y,z)|None)   # 프레임 없으면 None
stop() -> None
```

## 4. 이벤트 (push) / 폴링 (pull)
| API | 설명 |
|---|---|
| `on_result(fn)` | 프레임마다 `fn(SceneResult)` (데코레이터 가능) |
| `on_status(fn)` | `fn({"fps": float})` |
| `on_error(fn)` | `fn(Exception)` — 한 프레임 실패해도 루프는 계속 |
| `latest()` | 최신 `SceneResult`(없으면 None), 스레드세이프 |
| `fps` | 현재 추정 FPS |
| `draw_latest()` | 최신 결과를 최신 프레임에 그린 BGR ndarray |

> ⚠️ 콜백은 **백그라운드 스레드**에서 호출된다. Qt/Tk 등은 UI 갱신을 메인 스레드로
> 넘겨야 하므로, 콜백에선 데이터만 큐에 넣고 UI 스레드에서 그리는 패턴을 권장.

## 5. 제어 (UI 버튼에 바인딩)
| 메서드 | 동작 |
|---|---|
| `start()` / `stop()` | 생명주기 (with 컨텍스트 지원) |
| `set_live(bool)` | 실시간 모드. off 면 `trigger()` 때만 1회 검출 |
| `trigger()` | 비실시간에서 단발 검출 |
| `set_threshold(t)` | 신뢰도 임계 |
| `set_geometry("v1"/"v2")` | 윗면 기하 알고리즘 |
| `set_roi(x,y,w,h)` / `clear_roi()` | 관심영역 크롭 |
| `measure_gravity()` | 중력(up) 재측정·저장 → up 튜플/None |
| `load_weights(path)` | 재시작 없이 가중치 핫스왑 |

## 6. 결과 형식
```python
@dataclass
class SceneResult:
    pick_targets: list[PickPose]   # 가림 통과 + 윗면 추정된 픽 대상
    pallets:      list[Pallet]
    item_count:   int              # 픽 대상 검출 수
    done:         bool             # 팔레트만 남고 픽 대상 0 = 완료
    counts:       dict[int,int]
    inference_time: float
    intrinsics:   tuple|None

@dataclass
class PickPose:
    class_id:int; class_name:str; confidence:float
    center_3d:(x,y,z) m  # 카메라 프레임 픽 지점
    center_2d:(u,v) px
    long_axis_3d / short_axis_3d:(x,y,z)  # 단위 벡터
    long_side_mm / short_side_mm:float
    top_normal_3d:(x,y,z)|None  # 윗면 법선(그리퍼 접근축)
    mask: HxW bool|None
```

### 클래스 ID (★ 새 순서, v10~)
| id | 이름 |
|----|------|
| 0 | Pallet |
| 1 | Carton |
| 2 | Tote |
| 3 | Shrink Wrap |

---
문의: (담당자)
