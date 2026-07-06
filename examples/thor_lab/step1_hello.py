"""STEP 1 — 'SDK' 가 뭔가?

SDK = 설치해서 `import` 하면 쓰는 도구 모음. 내부(카메라·모델·기하)를 당신이 짜지 않고
함수/클래스를 불러다 쓴다. 여기선 import 한 줄로 검출기(detector)가 생기는 걸 본다.

실행:  bash run.sh step1_hello.py
"""
import dr_onsite_adap                       # ← 이게 'SDK 를 가져오는' 것
from dr_onsite_adap import OnsiteDetector, config

print("SDK 버전:", dr_onsite_adap.__version__)
print("이 SDK 가 아는 클래스:", {k: v[0] for k, v in config.CLASSES.items()})
print("가중치 로드 중... (수 초)")

det = OnsiteDetector()   # env 의 DEPAL_WEIGHTS/NUM_CLASSES/GEOM_ALGO 를 그대로 사용

print("\n✅ 준비 완료. 방금 한 일 = 'pip 로 깐 도구를 import 해서 검출기 객체를 얻음'.")
print("   → 이게 SDK 다. 다음 스텝에서 이 검출기를 실제로 써 본다.")
