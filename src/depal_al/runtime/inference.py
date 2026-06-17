import cv2
import torch
from PIL import Image
from rfdetr import RFDETRSegMedium


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
        print("Model ready.")

    def predict(self, bgr_frame, conf_threshold):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        with torch.no_grad():
            return self.model.predict(
                pil_img,
                conf_threshold=conf_threshold,
                device=self.device,
            )
