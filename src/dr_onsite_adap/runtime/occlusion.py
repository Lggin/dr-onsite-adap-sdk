import cv2
import numpy as np


def compute_occlusion_by_mask(collected, cfg):
    """Flag detections whose mask is too small relative to its own
    convex hull — i.e. enough of the silhouette is missing that the
    visible region almost certainly doesn't span the full top face.

    Single-mask test: we don't look at other detections to identify the
    occluder. Adjacent boxes with jittery mask boundaries used to trigger
    false positives in the cross-mask version; comparing each mask only
    to its own hull removes that failure mode entirely.

    Returns the set of indices (into `collected`) that should be hidden.
    """
    occluded = set()
    for j, det in enumerate(collected):
        mask = det["mask"]
        mask_area = int(mask.sum())
        if mask_area == 0:
            continue
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < 1.0:
            continue
        hidden_frac = 1.0 - (mask_area / hull_area)
        if hidden_frac > cfg.OCCLUSION_HIDDEN_FRACTION:
            occluded.add(j)
    return occluded
