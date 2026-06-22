import os

import cv2
import torch
from rfdetr import RFDETRSegMedium

from . import lean_postproc

# lean 후처리 사전선별 floor. 운영 검출 임계 이하여야 결과가 baseline 과 동일.
_PRESELECT_FLOOR = float(os.environ.get("DEPAL_MASK_PRESELECT", "0.05"))


class SegModel:
    def __init__(self, weights_path, num_classes, device):
        print(f"Loading RFDETRSegMedium ({device}) ← {weights_path}")
        self.model = RFDETRSegMedium(
            num_classes=num_classes,
            pretrain_weights=weights_path,
            device=device,
        )
        self.model.optimize_for_inference()
        self.device = device
        # 마스크 후처리를 lean 버전으로 교체 (살아남는 마스크만 업샘플 → 후처리 ~20ms 절감)
        lean_postproc.enable()
        print("Model ready.")

    def predict(self, bgr_frame, conf_threshold):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # GPU 텐서(C,H,W,[0,1])로 직접 입력 → resize/normalize 가 GPU 에서 수행(전처리 가속).
        # include_source_image=False 로 원본 이미지 D2H 복사 생략(SDK 는 source_image 미사용).
        img = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255.0)
        # 사전선별 임계는 검출 임계 이하로 (결과 동일성 보장)
        lean_postproc.set_preselect(min(_PRESELECT_FLOOR, conf_threshold))
        with torch.no_grad():
            return self.model.predict(
                img,
                conf_threshold=conf_threshold,
                device=self.device,
                include_source_image=False,
            )
