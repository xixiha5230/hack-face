"""人脸特征提取模块。

使用 MTCNN 检测人脸，InceptionResnetV1 提取 512 维嵌入向量。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ---------------------------------------------------------------------------
# 模型初始化（全局单例，避免重复加载）
# ---------------------------------------------------------------------------

_device: torch.device | None = None
_mtcnn: MTCNN | None = None
_resnet: InceptionResnetV1 | None = None


def _get_models() -> tuple[MTCNN, InceptionResnetV1, torch.device]:
    global _device, _mtcnn, _resnet
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _mtcnn is None:
        _mtcnn = MTCNN(device=_device, keep_all=False)
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
    return _mtcnn, _resnet, _device


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def get_face_embedding(image_path: str | Path) -> np.ndarray:
    """从图片中检测人脸并返回 512 维 float32 特征向量。

    Args:
        image_path: 目标图片路径。

    Returns:
        shape 为 (512,) 的 float32 numpy 数组。

    Raises:
        ValueError: 图片中未检测到人脸。
    """
    mtcnn, resnet, device = _get_models()

    img = Image.open(image_path).convert("RGB")
    img_cropped = mtcnn(img)  # 返回 [C, H, W] tensor 或 None

    if img_cropped is None:
        raise ValueError(f"在图片 '{image_path}' 中未检测到人脸")

    with torch.no_grad():
        embedding = resnet(img_cropped.unsqueeze(0).to(device))

    return embedding.cpu().numpy().flatten()  # shape: (512,), dtype: float32


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的余弦相似度。

    Returns:
        [-1, 1] 范围内的相似度值，越接近 1 越相似。
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def is_same_person(
    vec1: np.ndarray,
    vec2: np.ndarray,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """判断两个特征向量是否属于同一人。

    Args:
        vec1: 第一个人脸特征向量。
        vec2: 第二个人脸特征向量。
        threshold: 相似度阈值，默认 0.6。

    Returns:
        (是否匹配, 相似度得分) 元组。
    """
    score = cosine_similarity(vec1, vec2)
    return score > threshold, score


def crop_face(
    image_path: str | Path,
    margin: int = 40,
) -> Image.Image | None:
    """从图片中检测并裁剪人脸区域（带边距）。

    Args:
        image_path: 图片路径。
        margin: 人脸框外扩展边距（像素），让裁剪包含更多上下文。

    Returns:
        裁剪后的 PIL Image（仅人脸区域），或未检测到时返回 None。
    """
    mtcnn, _, _ = _get_models()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return None

    # 取置信度最高的人脸
    best = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[best]

    # 扩展边距
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(w, int(x2) + margin)
    y2 = min(h, int(y2) + margin)

    return img.crop((x1, y1, x2, y2))


def detect_faces(
    image_path: str | Path,
    threshold: float = 0.5,
) -> list[tuple[list[int], float]]:
    """检测图片中的所有人脸。

    Args:
        image_path: 图片路径。
        threshold: 置信度阈值。

    Returns:
        [(bbox_list, probability), ...] 列表。
    """
    # 使用 keep_all=True 的临时 MTCNN 来检测所有人脸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn_all = MTCNN(
        device=device, keep_all=True, thresholds=[threshold, threshold, threshold]
    )
    img = Image.open(image_path).convert("RGB")
    boxes, probs = mtcnn_all.detect(img)

    results = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob is not None and prob >= threshold:
                results.append((box.astype(int).tolist(), float(prob)))
    return results


def detect_faces_haar(
    image_or_path: str | Path | Image.Image,
    cascade_name: str = "haarcascade_frontalface_alt2",
    scale_factor: float = 1.1,
    min_neighbors: int = 3,
    min_size: tuple[int, int] = (30, 30),
) -> list[tuple[list[int], float]]:
    """使用 OpenCV Haar Cascade 检测人脸。

    Haar 比 MTCNN 宽松，更贴近一些企业头像系统的检测模型。

    Args:
        image_or_path: 图片路径或 PIL Image。
        cascade_name: Haar 级联分类器名称。
        scale_factor: 图像缩放系数。
        min_neighbors: 每个候选矩形最少邻居数。
        min_size: 最小检测尺寸。

    Returns:
        [(bbox_list, 1.0), ...] 列表（Haar 不提供概率，固定为 1.0）。
    """
    import cv2

    if isinstance(image_or_path, Image.Image):
        img_arr = np.array(image_or_path.convert("RGB"))
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(cv2.imread(str(image_or_path)), cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + cascade_name + ".xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    results = []
    for x, y, w, h in faces:
        results.append(([x, y, x + w, y + h], 1.0))
    return results


def verify_face_detectable(
    image_or_path: str | Path | Image.Image,
) -> dict:
    """多模型联合验证图片中是否可检测到人脸。

    依次使用：
      1. Haar Cascade alt2（最宽松，贴近企业系统）
      2. MTCNN 低阈值 (0.3)
      3. MTCNN 默认阈值 (0.6)

    Args:
        image_or_path: 图片路径或 PIL Image。

    Returns:
        {
            "haar": bool,        # Haar 是否检测到
            "mtcnn_low": bool,   # MTCNN 低阈值是否检测到
            "mtcnn_default": bool,  # MTCNN 默认阈值是否检测到
            "haar_count": int,
            "mtcnn_low_count": int,
            "mtcnn_low_prob": float | None,
            "pass_company": bool,  # 综合判断：能否通过企业检测
        }
    """
    if isinstance(image_or_path, (str, Path)):
        pil_img = Image.open(image_or_path).convert("RGB")
    else:
        pil_img = image_or_path.convert("RGB")

    # Haar Cascade
    haar_faces = detect_faces_haar(pil_img)

    # MTCNN 低阈值
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn_low = MTCNN(device=device, keep_all=True, thresholds=[0.3, 0.4, 0.4])
    boxes_low, probs_low = mtcnn_low.detect(pil_img)
    mtcnn_low_ok = boxes_low is not None and len(boxes_low) > 0
    mtcnn_low_count = len(boxes_low) if mtcnn_low_ok else 0
    mtcnn_low_prob = float(probs_low[0]) if mtcnn_low_ok else None

    # MTCNN 默认阈值
    mtcnn_def = MTCNN(device=device, keep_all=True)
    boxes_def, probs_def = mtcnn_def.detect(pil_img)
    mtcnn_def_ok = boxes_def is not None and len(boxes_def) > 0

    # 综合判断：MTCNN 低阈值通过即可（公司实测用深度学习模型）
    pass_company = mtcnn_low_ok

    return {
        "haar": len(haar_faces) > 0,
        "mtcnn_low": mtcnn_low_ok,
        "mtcnn_default": mtcnn_def_ok,
        "haar_count": len(haar_faces),
        "mtcnn_low_count": mtcnn_low_count,
        "mtcnn_low_prob": mtcnn_low_prob,
        "pass_company": pass_company,
    }
