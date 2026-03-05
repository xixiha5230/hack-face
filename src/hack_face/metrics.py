"""视觉质量评估与 AI 人脸检测指标。

提供：
  image_metrics(ref, result) → {ssim, psnr, mad, max_diff}
  detect_prob(img_path)      → (low_ok, low_prob, def_ok)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 视觉差异指标
# ---------------------------------------------------------------------------


def image_metrics(ref_path: str | Path, result_path: str | Path) -> dict:
    """计算融合结果与原载体之间的视觉差异指标。

    Args:
        ref_path: 载体原图（已裁剪缩放至与结果相同尺寸）。
        result_path: 融合结果图片。

    Returns:
        包含 ssim / psnr / mad / max_diff 的字典。
    """
    a = cv2.imread(str(ref_path))
    b = cv2.imread(str(result_path))
    if a is None or b is None:
        return {"ssim": 0.0, "psnr": 0.0, "mad": 0.0, "max_diff": 0}
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    mad = float(diff.mean())
    max_diff = int(diff.max())
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = 10.0 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
    ssim_val = _ssim(a, b)

    return {"ssim": ssim_val, "psnr": psnr, "mad": mad, "max_diff": max_diff}


def _ssim(
    a: np.ndarray, b: np.ndarray, k1: float = 0.01, k2: float = 0.03, win: int = 11
) -> float:
    """逐通道计算 SSIM 后取均值。"""
    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2
    ssims = []
    for ch in range(a.shape[2]):
        x = a[:, :, ch].astype(np.float64)
        y = b[:, :, ch].astype(np.float64)
        mu_x = cv2.GaussianBlur(x, (win, win), 1.5)
        mu_y = cv2.GaussianBlur(y, (win, win), 1.5)
        sigma_x = cv2.GaussianBlur(x**2, (win, win), 1.5) - mu_x**2
        sigma_y = cv2.GaussianBlur(y**2, (win, win), 1.5) - mu_y**2
        sigma_xy = cv2.GaussianBlur(x * y, (win, win), 1.5) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        ssims.append(float(np.mean(num / (den + 1e-12))))
    return float(np.mean(ssims))


# ---------------------------------------------------------------------------
# AI 人脸检测评估
# ---------------------------------------------------------------------------

_mtcnn_low = None
_mtcnn_def = None


def _get_detectors():
    global _mtcnn_low, _mtcnn_def
    if _mtcnn_low is None:
        import torch
        from facenet_pytorch import MTCNN

        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _mtcnn_low = MTCNN(device=_dev, keep_all=True, thresholds=[0.3, 0.4, 0.4])
        _mtcnn_def = MTCNN(device=_dev, keep_all=True)
    return _mtcnn_low, _mtcnn_def


def detect_prob(img_path: str | Path) -> tuple[bool, float, bool]:
    """检测图片中的人脸。

    Returns:
        (低阈值通过, 最高置信度, 默认阈值通过)
    """
    from PIL import Image

    mtcnn_low, mtcnn_def = _get_detectors()
    img = Image.open(img_path).convert("RGB")

    boxes_l, probs_l = mtcnn_low.detect(img)
    boxes_d, _ = mtcnn_def.detect(img)

    low_ok = False
    low_prob = 0.0
    if boxes_l is not None and probs_l is not None:
        valid = [float(p) for p in probs_l if p is not None and p > 0.3]
        if valid:
            low_ok = True
            low_prob = max(valid)

    def_ok = boxes_d is not None and len(boxes_d) > 0
    return low_ok, low_prob, def_ok
