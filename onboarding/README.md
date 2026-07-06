# 온보딩 (UI 팀 · x86 개발기)

이 파이프라인(검출 + 윗면 픽자세)을 **처음 받은 개발자**가 **명령 몇 개로** 개발을
시작하도록 자동화한 폴더입니다. 카메라가 없는 x86 개발기 기준(카메라 없이 샘플 재생으로 검증).

## 3단계 시작

```bash
# 0) 레포 클론
git clone <레포> && cd dr-onsite-adap-sdk   # (사내 접근권한 필요, 브랜치 lab1)

# 1) 설치 — venv + SDK + 의존성 (한 방)
bash onboarding/install.sh

# 2) 가중치 배치 — 담당자에게 continual_master_v10.pth 받아서:
mkdir -p weights && cp <받은경로>/continual_master_v10.pth weights/
#   (다른 위치면 onboarding/.env 의 DEPAL_WEIGHTS 수정. GPU 없으면 .env 에서 DEPAL_DEVICE=cpu)

# 3) 검증 실행 — 카메라 없이 샘플 프레임으로 OnsiteSession 동작 확인
bash onboarding/run.sh
```
`[hello] picks=... ` 가 뜨고 `✅ ... 정상 동작` 이 나오면 준비 끝.

## 내 앱 짜기
```python
# my_app.py
from dr_onsite_adap import OnsiteSession
sess = OnsiteSession()          # weights/geom/device 는 .env 에서
@sess.on_result
def on_scene(scene):
    for p in scene.pick_targets:
        ...                      # p.class_name / p.center_3d / p.long_side_mm ...
sess.start(); ...; sess.stop()
```
```bash
bash onboarding/run.sh my_app.py   # .env 를 읽어 venv 로 실행
```
- 카메라 없이 개발: `OnsiteSession(source=ReplaySource([...npz]))` (샘플/캡처 재생)
- 실제 카메라: `.env` 에 `DEPAL_CAMERA_BACKEND/SERIAL` 채우고 `OnsiteSession()` (source 생략)
- 고수준 API 전체 레퍼런스: [`../examples/README_session.md`](../examples/README_session.md)

## 폴더 구성
| 파일 | 역할 |
|---|---|
| `install.sh` | venv + SDK(+의존성) 설치 + `.env` 생성 |
| `.env.example` → `.env` | 설정(가중치·기하·장치·카메라). `run.sh` 가 읽음 |
| `run.sh` | `.env` 적용 + venv 파이썬으로 스크립트 실행 |
| `hello_session.py` | 설치 검증용 최소 예제(카메라 없이) |
| `sample/frame.npz` | 검증용 샘플 RGB-D 1장 |

## 주의
- `install.sh` 는 `pip install -e .` 로 `torch·rfdetr·opencv` 등을 깝니다(용량 큼, 수 분).
- **가중치(.pth)는 git 에 없음** — 담당자 배포.
- **클래스 순서(★)**: 0=Pallet, 1=Carton, 2=Tote, 3=Shrink Wrap.
- torch(CUDA)·카메라 SDK 는 환경 따라 설치가 까다로울 수 있음 — 문제 시 담당자 문의.
