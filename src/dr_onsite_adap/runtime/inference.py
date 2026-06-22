import os

import cv2
import torch
from PIL import Image
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
        pil_img = Image.fromarray(rgb)
        # 사전선별 임계는 검출 임계 이하로 (결과 동일성 보장)
        lean_postproc.set_preselect(min(_PRESELECT_FLOOR, conf_threshold))
        with torch.no_grad():
            return self.model.predict(
                pil_img,
                conf_threshold=conf_threshold,
                device=self.device,
            )
