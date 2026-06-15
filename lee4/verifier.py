"""
DINOv2 독립 검증 레이어.

RFDETR 이 confident 하게 틀리는 오분류(특히 '내용물 든 토트 -> 박스')를,
RFDETR 과 무관한 self-supervised 표현(DINOv2)으로 한 번 더 확인한다.

설계:
- 검출 마스크로 만든 masked 크롭 -> DINOv2 임베딩 -> 클래스 프로토타입 뱅크와
  cosine kNN 투표 -> 예측 클래스.
- RFDETR 라벨과 불일치하면 verdict["agree"]=False (flag-only: 막지 않고 표시만).

train/serve 일관성: 뱅크 빌드(build_bank.py)와 런타임이 이 파일의 동일한
masked_crop_from_mask + embed_crops_bgr 를 쓴다. (CPU 측정상 ViT-B/224 가
정확도 최고 + 픽당 ~1s 수준 — cpu_sweep 참고)
"""
import os
import numpy as np
import cv2
import torch
from PIL import Image

CLASS_NAMES = {0: "Box", 1: "Tote", 2: "Wrap", 3: "Pallet"}

# 크롭 규약 — 뱅크와 런타임이 반드시 동일해야 함
CROP_MARGIN = 0.15      # bbox 여유 비율
CROP_GRAY = 128         # 마스크 밖 배경 채움값


# ------------------------------------------------------------ 크롭 / 임베딩
def masked_crop_from_mask(color_bgr, mask, margin=CROP_MARGIN, gray=CROP_GRAY):
    """boolean mask + BGR 프레임 -> masked RGB 크롭 (마스크 밖은 회색).
    런타임은 RFDETR 예측 마스크, 뱅크는 GT 폴리곤 래스터 마스크를 넣는다."""
    mask = np.asarray(mask, bool)
    ys, xs = np.where(mask)
    if xs.size < 4:
        return None
    H, W = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    X0, Y0 = max(0, x0 - mx), max(0, y0 - my)
    X1, Y1 = min(W, x1 + mx + 1), min(H, y1 + my + 1)
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return None
    comp = np.where(mask[..., None], color_bgr, np.full_like(color_bgr, gray))
    crop = comp[Y0:Y1, X0:X1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def load_dinov2(model_name, img_size, device, weights_file=None):
    """timm DINOv2 로드. weights_file 이 있으면 그걸로(오프라인), 없으면 HF 캐시/다운로드."""
    import timm
    from timm.data import resolve_data_config, create_transform
    if weights_file and os.path.exists(weights_file):
        model = timm.create_model(model_name, pretrained=False,
                                  num_classes=0, img_size=img_size)
        model.load_state_dict(torch.load(weights_file, map_location="cpu",
                                         weights_only=True))
    else:
        model = timm.create_model(model_name, pretrained=True,
                                  num_classes=0, img_size=img_size)
    model.eval().to(device)
    cfg = resolve_data_config({}, model=model)
    cfg["input_size"] = (3, img_size, img_size)
    cfg["crop_pct"] = 1.0
    return model, create_transform(**cfg)


@torch.no_grad()
def embed_crops_rgb(model, tf, crops_rgb, device, batch=16):
    """list[np.ndarray RGB] -> (N, D) L2-normalized float32."""
    out = []
    for i in range(0, len(crops_rgb), batch):
        chunk = crops_rgb[i:i + batch]
        x = torch.stack([tf(Image.fromarray(c)) for c in chunk]).to(device)
        f = torch.nn.functional.normalize(model(x).float(), dim=1)
        out.append(f.cpu().numpy())
    return np.concatenate(out, 0) if out else np.empty((0, 0), np.float32)


# ------------------------------------------------------------ 검증기
class Verifier:
    """DINOv2 + 프로토타입 뱅크. verify() 는 검출 1건의 verdict 를 돌려준다."""

    def __init__(self, bank_path, model_name, img_size, device="cpu", k=5,
                 dino_weights=None):
        data = np.load(bank_path, allow_pickle=True)
        self.bank_emb = data["emb"].astype(np.float32)      # (N, D) L2-norm
        self.bank_lab = data["labels"].astype(int)          # (N,)
        self.k = int(min(k, len(self.bank_lab)))
        self.device = device
        self.model, self.tf = load_dinov2(model_name, img_size, device,
                                          weights_file=dino_weights)
        # 뱅크-런타임 설정 불일치 조기 경고
        meta = data["meta"].item() if "meta" in data else {}
        if meta.get("model_name") not in (None, model_name):
            print(f"[verifier] WARN bank model {meta.get('model_name')} != {model_name}")
        n_per = {CLASS_NAMES.get(c, c): int((self.bank_lab == c).sum())
                 for c in sorted(set(self.bank_lab))}
        print(f"[verifier] bank {len(self.bank_lab)} crops {n_per} | "
              f"{model_name}@{img_size} on {device} | k={self.k}")

    def verify(self, color_bgr, mask, rfdetr_cid):
        """returns verdict dict or None (크롭 실패 시)."""
        crop = masked_crop_from_mask(color_bgr, mask)
        if crop is None:
            return None
        emb = embed_crops_rgb(self.model, self.tf, [crop], self.device)[0]  # (D,)
        sims = self.bank_emb @ emb                              # cosine (둘 다 정규화)
        idx = np.argsort(-sims)[:self.k]
        votes = {}
        for j in idx:
            c = int(self.bank_lab[j])
            votes[c] = votes.get(c, 0.0) + float(max(sims[j], 1e-6))   # distance-weighted
        total = sum(votes.values()) or 1.0
        pred = max(votes, key=votes.get)
        return {
            "pred_cid": pred,
            "pred_name": CLASS_NAMES.get(pred, str(pred)),
            "agree": (pred == rfdetr_cid),
            "score": votes[pred] / total,                  # 예측 클래스 득표율
            "rf_score": votes.get(rfdetr_cid, 0.0) / total,  # RFDETR 라벨 득표율
            "margin": (votes[pred] / total) - (votes.get(rfdetr_cid, 0.0) / total),
        }
