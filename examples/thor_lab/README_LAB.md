# 핸즈온 랩 — SDK / facade / 저수준·고수준 직접 체험 (Thor)

> **가정**: 당신은 이 파이프라인을 GitHub 에서 처음 받은 개발자다. 개념을 글로 읽는 대신
> **Thor 에서 직접 실행하며** 몸으로 배운다. 3단계면 감이 잡힌다.

## 0. 개념 30초 요약 (커피머신 비유)
- **SDK** = 설치해서 `import` 해 쓰는 도구 모음. (= 카페에 들인 **에스프레소 머신**)
- **facade(파사드)** = 복잡한 내부(카메라·모델·기하)를 가린 **간단한 앞문**. (= 머신 **버튼 패널**)
- **저수준(low-level)** `OnsiteDetector` = 한 프레임씩 당신이 넣고 처리. 통제 크고 배선도 당신 몫. (= 핸드드립)
- **고수준(high-level)** `OnsiteSession` = 카메라·루프까지 SDK 가 다 함. 당신은 콜백만. (= 원버튼 머신)

둘 다 **같은 내부**를 쓰는 facade. **추상화 높이만** 다르다.

## 1. 준비 (Thor, NoMachine 터미널)
```bash
cd ~/dr-onsite-adap-sdk/examples/thor_lab
```
모든 스텝은 `run.sh` 로 실행한다(운영과 동일한 env=v10 가중치·v2 기하·카메라 자동 세팅):
```bash
bash run.sh <스크립트>
```

---

## STEP 1 — "SDK 란?" (import 한 줄로 도구 얻기)
```bash
bash run.sh step1_hello.py
```
**볼 것**: 버전·클래스 목록이 뜨고 "준비 완료". → *설치한 도구를 import 하니 검출기 객체가 생겼다.* 이게 SDK.

---

## STEP 2 — 저수준 facade `OnsiteDetector`
```bash
bash run.sh step2_lowlevel.py
```
**하는 일**: 저장된 캡처 1장을 **당신이** 읽어서 `det.process(프레임)` 에 넣는다.
**볼 것**: 검출 결과(클래스·중심·치수) + `/tmp/step2_out.jpg`.
**깨달을 점**: SDK 가 해준 건 `process()` **한 줄(한 프레임)**. 프레임을 어디서 구하고 언제 부를지는 당신 몫.
실시간으로 하려면 여기에 **카메라 열기 + 루프 + 스레드** 를 당신이 더 짜야 한다. → STEP 3 와 비교.

`step2_lowlevel.py` 를 **열어서** 읽어보라: 프레임 준비 코드가 당신 것이고, SDK 는 한 줄이다.

---

## STEP 3 — 고수준 facade `OnsiteSession`
### (a) 실제 카메라
```bash
bash run.sh step3_highlevel.py
```
**볼 것**: `[실제 카메라] SDK 가 Thor 카메라를 엽니다...` 후 프레임마다 `[live] picks=...` 가 흐른다.
**깨달을 점**: 당신은 **카메라 코드를 한 줄도 안 썼다.** `OnsiteSession()` + `on_result` 콜백 + `start()` 가 전부.
카메라 열기·프레임 추출·루프·스레드는 **SDK 가** 했다.

### (b) 카메라 없이 (소스 주입)
```bash
bash run.sh step3_highlevel.py --replay '/home/dr/geom_captures/rgbd_*.npz'
```
같은 코드인데 카메라 대신 저장 프레임을 재생. → *고수준도 "프레임 소스" 만 바꿔 끼우면 카메라 없이 개발 가능.*

`step3_highlevel.py` 를 **열어서** STEP 2 와 나란히 비교하라: STEP 2 의 '프레임 준비·루프' 가 통째로 사라졌다.

---

## STEP 4 — 정리 (직접 비교)
두 파일을 나란히 열고 "내가 짠 줄" 을 세어 보라:

| | 저수준 `OnsiteDetector` (STEP 2) | 고수준 `OnsiteSession` (STEP 3) |
|---|---|---|
| SDK 가 해주는 것 | 한 프레임 처리(`process`) | 카메라+루프+스레드+제어 전부 |
| **당신이 짜는 것** | 프레임 준비·루프·(실시간이면) 스레드 | **콜백 하나 + start/stop** |
| 적합 | 자체 프레임 소스(ROS)·프레임 단위 통제 | UI 팀이 빨리 화면 올릴 때 |

**한 줄 결론**: facade 는 "복잡한 내부를 가린 앞문"이고, **저수준은 앞문이 낮아(프레임 단위) 당신이 더 조립**, **고수준은 앞문이 높아(카메라째) SDK 가 다 조립**한다.

## 더 해보기 (고수준 제어 — UI 버튼에 붙는 것들)
`OnsiteSession` 객체에서:
```python
sess.set_roi(x,y,w,h); sess.clear_roi()   # 관심영역
sess.measure_gravity()                      # 중력(up) 재측정
sess.load_weights("다른.pth")              # 재시작 없이 가중치 교체
sess.set_threshold(0.5); sess.set_geometry("v2"); sess.set_live(True)
scene = sess.latest()                        # 콜백 대신 폴링으로 최신 결과
```
자세한 레퍼런스: [`../README_session.md`](../README_session.md)
