"""depal_detect — 디팔레타이징 검출 facade (Pallet / Slip Sheet / Carton / Tote / Shrink Wrap).

가중치는 같은데 코드마다 검출율이 달라지는 문제를 막기 위해, 모델 로드·전처리·
임계·클래스 설정을 **한 곳에 캡슐화**한다. 이 파일 + 가중치 파일만 있으면 누가
어디서 import 하든 동일한 검출 결과가 나온다.

필요 패키지:  torch, rfdetr, opencv-python, numpy

사용 (파이썬)
-----------
    from depal_detect import Detector

    det = Detector("continual_master_v11.pth")     # device 자동(cuda 우선)
    results = det.detect("frame.jpg")                   # 경로 / BGR ndarray / RGB ndarray
    for r in results:
        print(r.class_name, round(r.confidence, 2), r.bbox)   # r.mask(HxW bool)도 있음

    vis = det.draw("frame.jpg", results)                # 시각화(BGR) — cv2.imwrite 로 저장

CLI
---
    python depal_detect.py frame.jpg --weights continual_master_v11.pth --threshold 0.5 --save out.jpg

주의 (남이 쓸 때 검출율 떨어지는 흔한 원인 — 이 모듈이 전부 처리함)
  * num_classes 는 가중치와 같아야 함(여기 기본 5). 안 맞으면 헤드 어긋나 검출 망가짐.
  * cv2 로 읽은 이미지는 BGR → 모델엔 RGB 로 넣어야 함(여기서 변환).
  * rfdetr 의 임계 인자명은 `threshold=` (conf_threshold 아님).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from rfdetr import RFDETRSegMedium

# 학습 시 클래스 순서와 동일해야 한다(가중치에 박혀 있음). v11~ 5클래스 순서.
CLASS_NAMES = {0: "Pallet", 1: "Slip Sheet", 2: "Carton", 3: "Tote", 4: "Shrink Wrap"}
_CLASS_BGR = {0: (50, 100, 160), 1: (255, 0, 255), 2: (0, 255, 0), 3: (255, 165, 0), 4: (0, 0, 255)}


@dataclass
class Detection:
    """검출 1건. bbox = (x1, y1, x2, y2) 픽셀 정수, mask = HxW bool(없으면 None)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None = None


class Detector:
    """RF-DETR Seg 박스/토트/랩/팔레트 검출기 facade.

    Parameters
    ----------
    weights : str
        학습된 가중치 경로(예: continual_master_v11.pth).
    num_classes : int
        가중치의 클래스 수. 기본 5(Pallet/Slip Sheet/Carton/Tote/Shrink Wrap). **가중치와 반드시 일치.**
    device : str | None
        "cuda"/"cpu". None 이면 가능하면 cuda 자동.
    resolution : int | None
        입력 해상도 override. None 이면 모델 기본(RF-DETR Seg Medium = 432). 학습과
        다르게 주지 말 것(검출 스케일이 달라짐).
    """

    def __init__(self, weights: str, num_classes: int = 5,
                 device: str | None = None, resolution: int | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"num_classes": num_classes, "pretrain_weights": weights,
              "device": self.device}
        if resolution is not None:
            kw["resolution"] = resolution
        self.model = RFDETRSegMedium(**kw)
        self.model.optimize_for_inference()
        self.num_classes = num_classes

    # ---- 내부: 입력을 RGB uint8 HWC 로 정규화 ----
    @staticmethod
    def _to_rgb(image, is_bgr: bool) -> np.ndarray:
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise FileNotFoundError(f"이미지를 열 수 없음: {image}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"HxWx3 이미지가 필요(받은 shape={arr.shape})")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if is_bgr else arr

    def detect(self, image, threshold: float = 0.5,
               is_bgr: bool = True) -> list[Detection]:
        """검출 실행.

        image : 파일 경로(str) / ndarray(HxWx3). ndarray 면 is_bgr 로 채널 순서 지정
                (cv2.imread 결과는 BGR=True, matplotlib/PIL 은 RGB=False).
        threshold : 신뢰도 임계(기본 0.5).
        반환 : Detection 리스트.
        """
        rgb = self._to_rgb(image, is_bgr)
        with torch.no_grad():
            preds = self.model.predict(rgb, threshold=threshold)

        xyxy = np.asarray(preds.xyxy)
        cls = np.asarray(preds.class_id)
        conf = np.asarray(preds.confidence)
        masks = getattr(preds, "mask", None)

        out: list[Detection] = []
        for i in range(len(xyxy)):
            cid = int(cls[i])
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
            out.append(Detection(
                class_id=cid,
                class_name=CLASS_NAMES.get(cid, str(cid)),
                confidence=float(conf[i]),
                bbox=(x1, y1, x2, y2),
                mask=(masks[i].astype(bool) if masks is not None else None),
            ))
        return out

    def draw(self, image, detections: list[Detection],
             is_bgr: bool = True) -> np.ndarray:
        """검출 결과를 그린 BGR 이미지를 반환(cv2.imwrite 로 저장)."""
        if isinstance(image, str):
            vis = cv2.imread(image)
        else:
            arr = np.asarray(image)
            vis = arr.copy() if is_bgr else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        for d in detections:
            color = _CLASS_BGR.get(d.class_id, (255, 255, 255))
            if d.mask is not None:
                ov = vis.copy()
                ov[d.mask] = color
                vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{d.class_name} {d.confidence:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return vis


def main():
    ap = argparse.ArgumentParser(description="depal_detect facade — 박스/토트/랩/팔레트 검출")
    ap.add_argument("image", help="이미지 파일 경로")
    ap.add_argument("--weights", required=True, help="가중치 경로(.pth)")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None, help="시각화 저장 경로(.jpg)")
    args = ap.parse_args()

    det = Detector(args.weights, num_classes=args.num_classes, device=args.device)
    results = det.detect(args.image, threshold=args.threshold)
    print(f"검출 {len(results)}건 (threshold={args.threshold})")
    for r in results:
        print(f"  {r.class_name:7s} conf={r.confidence:.3f} bbox={r.bbox}")
    if args.save:
        cv2.imwrite(args.save, det.draw(args.image, results))
        print(f"시각화 저장: {args.save}")


if __name__ == "__main__":
    main()
