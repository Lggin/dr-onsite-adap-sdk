# deploy/ — 현장(Thor) 실행 스크립트

Thor 현장 PC의 운영 런처 스크립트 사본(버전관리/백업용). **Thor 고유 경로·시리얼·
중력벡터가 박혀 있으므로 다른 PC에선 값을 수정**해서 쓰세요.

| 파일 | 용도 |
|---|---|
| `demo_gui.sh` | 데모 GUI(`dr-onsite-gui`) — 실시간 검출 + 현장 학습 탭 |
| `demo_capture.sh` | 키보드 앱(`dr-onsite-run`) — RGB-D 캡처('g') / 라벨링('t') |

## 현재 운영 설정 (2026-06-26)

- `DEPAL_WEIGHTS` = **continual_master_v9_full.pth** (운영 검출 가중치)
- `DEPAL_BASE_WEIGHTS` = **continual_master_v9_full.pth** (현장 LoRA 라운드 학습 베이스)
- `DEPAL_GEOM_ALGO` = **v2** (윗면 기하: 평면 fit + 옆면 거부 + 최소면적)
- `DEPAL_DEPTH_MODE` = 320x288, `DEPAL_USE_IMU` = (gui:0 / capture:1), `DEPAL_UP_PATH` = ~/up.json
- 가중치 파일은 git 미포함 — `~/depal_weights_backup/` 에 별도 배포

> Thor 에서 스크립트를 고치면 이 사본도 갱신(여기서 scp)해 두면 설정 이력이 남습니다.
