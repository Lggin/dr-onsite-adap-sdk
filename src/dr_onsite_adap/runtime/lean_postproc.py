"""Lean 마스크 후처리 — RF-DETR seg 추론 속도 최적화 (버전 적응형).

기본 RF-DETR `PostProcess` 는 topk(=num_select, 기본 300) 마스크를 **전부** 입력 해상도로
업샘플(bilinear)·이진화한 뒤, 호출부(`predict`)가 `score > threshold` 로 거른다. 실제 장면에서
살아남는 박스는 보통 한 자릿수인데 300개를 풀해상도로 업샘플하는 게 후처리 비용의
대부분(Thor 1280x720 기준 ~20ms)이었다.

이 모듈은 후처리를 **사전선별(pre-select) 후 업샘플** 로 교체한다: topk 점수는 내림차순이라
`score > thr` 는 상위 연속 구간 → 그 구간만 업샘플/이진화.

**정확성**: 사전선별 임계 `thr` 가 호출부 검출 임계 `P` **이하**이면, 반환 마스크 집합은 최종
생존자의 상위집합이고 호출부의 `score > P` 가 같은 부분집합을 골라 결과가 비트 동일하다
(ganginenv/rfdetr 1.7.1 실측 검증). 그래서 `thr` 는 운영 임계 이하의 낮은 floor 로 둔다.

**버전 적응형**: rfdetr 1.8.x 는 `PostProcess._postprocess_masks`(staticmethod) 를, 1.7.x 는
`PostProcess.forward` 인라인을 교체한다.
"""
import os

import torch
import torch.nn.functional as F
from rfdetr.models.postprocess import PostProcess

_PRESELECT = {"thr": float(os.environ.get("DEPAL_MASK_PRESELECT", "0.05"))}
_orig = {"mode": None, "fn": None}


def set_preselect(thr: float) -> None:
    """사전선별 임계 설정 (호출부 검출 임계 이하여야 결과 동일)."""
    _PRESELECT["thr"] = float(thr)


# ---- rfdetr 1.8.x: _postprocess_masks(staticmethod) 교체 ----
def _lean_postprocess_masks(out_masks, scores, labels, boxes, topk_boxes, target_sizes):
    thr = _PRESELECT["thr"]
    results = []
    for i in range(out_masks.shape[0]):
        sc = scores[i]
        n = int((sc > thr).sum().item())
        k_idx = topk_boxes[i][:n]
        h, w = target_sizes[i].tolist()
        if n == 0:
            masks_i = torch.zeros((0, 1, int(h), int(w)), dtype=torch.bool, device=out_masks.device)
        else:
            masks_i = out_masks[i][k_idx]
            masks_i = F.interpolate(masks_i.unsqueeze(1), size=(int(h), int(w)),
                                    mode="bilinear", align_corners=False)
            masks_i = masks_i > 0.0
        results.append({"scores": sc[:n], "labels": labels[i][:n],
                        "boxes": boxes[i][:n], "masks": masks_i})
    return results


# ---- rfdetr 1.7.x: forward 교체 (box_ops 는 postprocess 모듈에서 가져옴) ----
def _make_lean_forward():
    import rfdetr.models.postprocess as _pp
    box_ops = _pp.box_ops

    @torch.no_grad()
    def _lean_forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks", None)
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        thr = _PRESELECT["thr"]
        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                sc = scores[i]
                n = int((sc > thr).sum().item())  # 상위 연속 구간만
                k_idx = topk_boxes[i][:n]
                h, w = target_sizes[i].tolist()
                if n == 0:
                    masks_i = torch.zeros((0, 1, int(h), int(w)), dtype=torch.bool,
                                          device=out_masks.device)
                else:
                    masks_i = out_masks[i][k_idx]
                    masks_i = F.interpolate(masks_i.unsqueeze(1), size=(int(h), int(w)),
                                            mode="bilinear", align_corners=False)
                    masks_i = masks_i > 0.0
                results.append({"scores": sc[:n], "labels": labels[i][:n],
                                "boxes": boxes[i][:n], "masks": masks_i})
        else:
            results = [{"scores": s, "labels": l, "boxes": b}
                       for s, l, b in zip(scores, labels, boxes)]
        return results

    return _lean_forward


def enable() -> None:
    """후처리를 lean 버전으로 교체(idempotent, 버전 적응형)."""
    if _orig["mode"] is not None:
        return
    if hasattr(PostProcess, "_postprocess_masks"):       # rfdetr 1.8.x
        _orig["mode"], _orig["fn"] = "masks", PostProcess._postprocess_masks
        PostProcess._postprocess_masks = staticmethod(_lean_postprocess_masks)
    else:                                                 # rfdetr 1.7.x
        _orig["mode"], _orig["fn"] = "forward", PostProcess.forward
        PostProcess.forward = _make_lean_forward()


def disable() -> None:
    """원래 후처리로 복원."""
    if _orig["mode"] == "masks":
        PostProcess._postprocess_masks = staticmethod(_orig["fn"])
    elif _orig["mode"] == "forward":
        PostProcess.forward = _orig["fn"]
    _orig["mode"], _orig["fn"] = None, None
