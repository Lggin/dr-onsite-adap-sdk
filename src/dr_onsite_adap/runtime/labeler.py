"""라이브 앱에 내장된 SAM 라벨링.

앱에서 't' 를 누르면 현재 화면에서 멈춰 SAM 으로 라벨합니다:
드래그=바운딩박스, 좌클릭=포함점(positive), 우클릭=제외점(negative).
SAM 이 마스크를 만들면 클래스 숫자키를 눌러 확정합니다(0=Box, 1=Tote, 2=Wrap,
3=Pallet). 유효한 숫자키는 config.CLASSES 에서 자동으로 가져오므로 배포 클래스 수에
맞춰집니다. 한 화면에서 여러 객체를 확정할 수 있습니다.
Enter 로 저장하면(jpg + 마스크 png + json) 학습 파이프라인(`dr-onsite-ingest`)이
읽는 액티브러닝 폴더에 들어갑니다. ESC 는 취소입니다.
"""

import datetime
import json
import os

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


class SAMLabeler:
    """Owns the SAM model + per-session UI state. label_frame() runs one
    interactive pass on a paused frame."""

    WINDOW_NAME = "Manual Tagging (SAM)"

    def __init__(self, sam_checkpoint, save_dir, classes,
                 sam_type="vit_b", device="cpu"):
        if not os.path.isfile(sam_checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_checkpoint}. "
                f"Set DEPAL_SAM_WEIGHTS to the .pth path."
            )
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.classes = classes
        # 클래스 확정용 숫자 키 — config.CLASSES 에서 동적으로 끌어와 클래스 수가
        # 늘어도(예: 3=Pallet 추가) 깨지지 않게 한다. {ord('3'): 3, ...}
        self.class_keys = {ord(str(c)): c for c in classes if 0 <= c <= 9}
        print(f"Loading SAM ({sam_type}, {device}) ← {sam_checkpoint}")
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        print("SAM ready.")
        self._reset()

    def _reset(self):
        self.confirmed_img = None
        self.temp_img = None
        self.mask_canvas = None
        self.annotations = []
        self.current_mask = None
        self.current_polygon = None
        self.current_bbox = None
        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.saved_box = None
        self.input_points = []
        self.input_labels = []
        self.poly_mode = False      # 'p' 로 토글. True 면 클릭=폴리곤 점.
        self.poly_pts = []          # 수동 폴리곤 정점들

    def _draw_poly(self):
        """현재 폴리곤 정점/선을 미리보기로 그린다."""
        self.temp_img = self.confirmed_img.copy()
        if self.poly_pts:
            pts = np.array(self.poly_pts, np.int32)
            if len(self.poly_pts) >= 2:
                cv2.polylines(self.temp_img, [pts], False, (0, 255, 255), 2)
            for p in self.poly_pts:
                cv2.circle(self.temp_img, tuple(p), 4, (0, 255, 0), -1)
        cv2.imshow(self.WINDOW_NAME, self.temp_img)

    def _finish_poly(self):
        """모은 정점으로 폴리곤 마스크를 만들어 'SAM 마스크'와 동일한 확정 대기 상태로."""
        if len(self.poly_pts) < 3:
            print("  점 3개 이상 찍어야 합니다")
            return
        pts = np.array(self.poly_pts, np.int32)
        h, w = self.confirmed_img.shape[:2]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        self.current_mask = m.astype(bool)
        self.current_polygon = pts.reshape(-1, 1, 2)
        xb, yb, wb, hb = cv2.boundingRect(pts)
        self.current_bbox = [xb, yb, xb + wb, yb + hb]
        self.poly_pts = []
        # 미리보기
        self.temp_img = self.confirmed_img.copy()
        overlay = np.zeros_like(self.temp_img)
        overlay[self.current_mask] = (0, 255, 255)
        cv2.addWeighted(self.temp_img, 1.0, overlay, 0.5, 0, self.temp_img)
        cv2.polylines(self.temp_img, [self.current_polygon], True, (0, 255, 255), 2)
        cv2.imshow(self.WINDOW_NAME, self.temp_img)
        print("  폴리곤 완성 — 클래스 숫자키로 확정하세요")

    def _update_sam(self):
        if self.saved_box is None:
            return
        pts = np.array(self.input_points) if self.input_points else None
        lbls = np.array(self.input_labels) if self.input_labels else None
        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            box=self.saved_box[None, :],
            multimask_output=False,
        )
        mask = masks[0]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        self.temp_img = self.confirmed_img.copy()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            self.current_polygon = largest
            xb, yb, wb, hb = cv2.boundingRect(largest)
            self.current_bbox = [xb, yb, xb + wb, yb + hb]
            self.current_mask = mask
            overlay = np.zeros_like(self.temp_img)
            overlay[mask] = (0, 255, 255)
            cv2.addWeighted(self.temp_img, 1.0, overlay, 0.5, 0, self.temp_img)
            cv2.polylines(self.temp_img, [largest], True, (0, 255, 255), 2)
            cv2.rectangle(
                self.temp_img,
                (int(self.saved_box[0]), int(self.saved_box[1])),
                (int(self.saved_box[2]), int(self.saved_box[3])),
                (255, 0, 0), 2,
            )
        for i, pt in enumerate(self.input_points):
            col = (0, 255, 0) if self.input_labels[i] == 1 else (0, 0, 255)
            cv2.circle(self.temp_img, tuple(pt), 4, col, -1)
            cv2.circle(self.temp_img, tuple(pt), 5, (255, 255, 255), 1)
        cv2.imshow(self.WINDOW_NAME, self.temp_img)
        print(f"  mask updated (score {scores[0]:.2f})")

    def _on_mouse(self, event, x, y, flags, param):
        # 폴리곤 모드: 좌클릭=정점 추가, 우클릭=마지막 점 취소. (SAM 마우스 동작 우회)
        if self.poly_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.poly_pts.append([x, y])
                self._draw_poly()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.poly_pts:
                    self.poly_pts.pop()
                    self._draw_poly()
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.saved_box is None:
                print("  먼저 바운딩박스를 그리세요(좌클릭 드래그)")
                return
            self.input_points.append([x, y])
            self.input_labels.append(0)
            self._update_sam()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_img = self.confirmed_img.copy()
                cv2.rectangle(self.temp_img, (self.ix, self.iy), (x, y),
                              (255, 0, 0), 2)
                cv2.imshow(self.WINDOW_NAME, self.temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if abs(self.ix - x) < 5 and abs(self.iy - y) < 5:
                if self.saved_box is None:
                    print("  먼저 바운딩박스를 그리세요(좌클릭 드래그)")
                    return
                self.input_points.append([x, y])
                self.input_labels.append(1)
                self._update_sam()
            else:
                x1, y1 = min(self.ix, x), min(self.iy, y)
                x2, y2 = max(self.ix, x), max(self.iy, y)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    return
                print("  새 박스 → SAM 초기 마스크 생성")
                self.saved_box = np.array([x1, y1, x2, y2])
                self.input_points = []
                self.input_labels = []
                self._update_sam()

    def label_frame(self, bgr_frame):
        """Pause on `bgr_frame` and run the interactive tagging loop.
        Returns True if at least one annotation was saved to disk."""
        self._reset()
        self.confirmed_img = bgr_frame.copy()
        self.temp_img = self.confirmed_img.copy()
        self.mask_canvas = np.zeros(bgr_frame.shape[:2], dtype=np.uint8)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        keyhint = ", ".join(f"{c}={self.classes[c][0]}"
                             for c in sorted(self.class_keys.values()))
        print("\n-- SAM 라벨링 --")
        print("  [SAM] 드래그=박스, 좌클릭=포함점, 우클릭=제외점")
        print("  [폴리곤] 'p'=모드 토글, 좌클릭=점 찍기, 우클릭=점 취소, 'f'=완성")
        print(f"  마스크 확정용 숫자키: {keyhint} (반복 가능)")
        print("  Enter=저장  /  ESC=취소\n")

        self.predictor.set_image(rgb_frame)
        # AUTOSIZE 유지(창=이미지 1:1 → 마우스 좌표가 픽셀과 정확히 일치).
        # 좌상단으로 옮기고 TOPMOST 로 다른 창 뒤에 깔리지 않게만 한다.
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.moveWindow(self.WINDOW_NAME, 0, 0)
        try:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1.0)
        except Exception:
            pass
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
        cv2.imshow(self.WINDOW_NAME, self.temp_img)

        try:
            while True:
                k = cv2.waitKey(1) & 0xFF
                if k == ord("p"):
                    self.poly_mode = not self.poly_mode
                    self.poly_pts = []
                    self.temp_img = self.confirmed_img.copy()
                    cv2.imshow(self.WINDOW_NAME, self.temp_img)
                    print("  폴리곤 모드:",
                          "ON (좌클릭=점, f=완성, 우클릭=취소)" if self.poly_mode
                          else "OFF (드래그=박스 SAM)")
                elif k == ord("f") and self.poly_mode:
                    self._finish_poly()
                elif self.current_bbox is not None and k in self.class_keys:
                    cid = self.class_keys[k]
                    name, color = self.classes[cid]
                    self.annotations.append({
                        "class_id": cid,
                        "class_name": name,
                        "bbox": self.current_bbox,
                        "is_manual_mask": True,
                        "segmentation": self.current_polygon.flatten().tolist(),
                    })
                    self.mask_canvas[self.current_mask] = 255
                    overlay = np.zeros_like(self.confirmed_img)
                    overlay[self.current_mask] = color
                    cv2.addWeighted(self.confirmed_img, 1.0, overlay, 0.4, 0,
                                    self.confirmed_img)
                    cv2.polylines(self.confirmed_img, [self.current_polygon],
                                  True, color, 2)
                    print(f"  + {name}  (누적 {len(self.annotations)}개)")
                    self.current_mask = self.current_polygon = self.current_bbox = None
                    self.saved_box = None
                    self.input_points = []
                    self.input_labels = []
                    self.temp_img = self.confirmed_img.copy()
                    cv2.imshow(self.WINDOW_NAME, self.temp_img)
                elif k == 13:  # Enter
                    return self._save(bgr_frame)
                elif k == 27:  # ESC
                    print("  취소됨 — 저장 안 함")
                    return False
        finally:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except Exception:
                pass

    def _save(self, bgr_frame):
        if not self.annotations:
            print("  저장할 라벨이 없습니다")
            return False
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = f"sam_hard_{ts}.jpg"
        mask_file = f"sam_hard_{ts}_mask.png"
        json_file = f"sam_hard_{ts}.json"
        cv2.imwrite(os.path.join(self.save_dir, img_file), bgr_frame)
        cv2.imwrite(os.path.join(self.save_dir, mask_file), self.mask_canvas)
        with open(os.path.join(self.save_dir, json_file), "w") as f:
            json.dump({
                "image_file": img_file,
                "mask_file": mask_file,
                "annotations": self.annotations,
            }, f, indent=4)
        print(f"  {len(self.annotations)}개 객체 저장됨 → {self.save_dir}/{img_file}")
        return True
