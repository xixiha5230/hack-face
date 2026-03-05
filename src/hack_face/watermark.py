"""水印编解码模块。

使用 LSB（最低有效位）隐写术将人脸 512 维 float32 嵌入向量
藏入载体图片的像素最低位，实现无损嵌入与提取。

容量说明：
  嵌入数据  = 4 字节头 + 2048 字节负载 = 2052 字节 = 16416 bits
  1024×1024 图片容量 = 1024×1024×3 bits = 3,145,728 bits（远超需求）

隐蔽性：每像素每通道仅改变最低 1 bit，人眼完全无法察觉。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_EMBEDDING_DIM = 512
_EMBEDDING_BYTES = _EMBEDDING_DIM * 4  # 2048 bytes（512 个 float32）
_HEADER_BYTES = 4  # 小端序 uint32：负载字节数


# ---------------------------------------------------------------------------
# 内部辅助：纯 numpy 位操作
# ---------------------------------------------------------------------------


def _bytes_to_bits(data: bytes) -> np.ndarray:
    """将字节序列转换为 uint8 bit 数组（MSB first）。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)  # shape: (len(data)*8,)
    return bits


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """将 uint8 bit 数组（MSB first）重新打包为 bytes。"""
    return np.packbits(bits).tobytes()


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def encode_face_into_image(
    embedding: np.ndarray,
    carrier_path: str | Path,
    output_path: str | Path,
) -> None:
    """将人脸特征向量以 LSB 隐写形式嵌入载体图片。

    每像素每通道仅修改最低 1 bit，人眼无法察觉差异。
    输出必须保存为无损格式（PNG），有损压缩（JPEG）会破坏水印。

    Args:
        embedding: shape (512,) 的 float32 人脸特征向量。
        carrier_path: 载体图片路径（任意格式可读）。
        output_path: 输出图片路径，建议 .png 后缀。

    Raises:
        ValueError: 嵌入向量维度不为 512 或图片容量不足。
        FileNotFoundError: 载体图片不存在。
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (_EMBEDDING_DIM,):
        raise ValueError(f"嵌入向量维度应为 {_EMBEDDING_DIM}，实际为 {embedding.shape}")

    carrier_path = Path(carrier_path)
    if not carrier_path.exists():
        raise FileNotFoundError(f"载体图片不存在：{carrier_path}")

    bgr_img = cv2.imread(str(carrier_path))
    if bgr_img is None:
        raise ValueError(f"无法读取图片：{carrier_path}")

    # 构造负载：4 字节长度头 + 2048 字节浮点数据
    payload = _EMBEDDING_BYTES.to_bytes(_HEADER_BYTES, "little") + embedding.tobytes()
    payload_bits = _bytes_to_bits(payload)  # shape: (16416,)

    flat = bgr_img.flatten()
    capacity_bits = len(flat)
    if len(payload_bits) > capacity_bits:
        raise ValueError(
            f"图片容量不足：需要 {len(payload_bits)} bits，"
            f"图片仅有 {capacity_bits} bits 可用"
        )

    # 将 payload bits 写入像素 LSB
    flat[: len(payload_bits)] = (flat[: len(payload_bits)] & 0xFE) | payload_bits
    bgr_encoded = flat.reshape(bgr_img.shape)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), bgr_encoded)
    print(f"✓ 已保存带隐藏人脸的图片：{output_path}")
    print(
        f"  嵌入方式：LSB 隐写  |  修改像素通道数：{len(payload_bits)}  |  "
        f"占总容量：{len(payload_bits) / capacity_bits * 100:.3f}%"
    )


def decode_face_from_image(watermarked_path: str | Path) -> np.ndarray:
    """从 LSB 隐写图片中提取人脸特征向量。

    Args:
        watermarked_path: 经过 encode_face_into_image 处理的 PNG 图片路径。

    Returns:
        shape (512,) 的 float32 numpy 数组。

    Raises:
        FileNotFoundError: 图片不存在。
        ValueError: 图片未包含有效嵌入数据（长度校验失败）。
    """
    watermarked_path = Path(watermarked_path)
    if not watermarked_path.exists():
        raise FileNotFoundError(f"图片不存在：{watermarked_path}")

    bgr_img = cv2.imread(str(watermarked_path))
    if bgr_img is None:
        raise ValueError(f"无法读取图片：{watermarked_path}")

    flat = bgr_img.flatten()

    # 1. 读取头部 4×8=32 bits → uint32 负载长度
    header_bits = flat[: _HEADER_BYTES * 8] & 1
    payload_len = int.from_bytes(_bits_to_bytes(header_bits), "little")

    if payload_len != _EMBEDDING_BYTES:
        raise ValueError(
            f"水印头校验失败：期望 {_EMBEDDING_BYTES} 字节，读到 {payload_len}。"
            "图片可能未经 hack-face-encode 处理，或经过了有损压缩。"
        )

    # 2. 读取数据位
    total_bits = (_HEADER_BYTES + payload_len) * 8
    data_bits = flat[_HEADER_BYTES * 8 : total_bits] & 1

    # 3. bits → bytes → float32
    raw_bytes = _bits_to_bytes(data_bits)
    vector = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    return vector  # shape: (512,)


# ---------------------------------------------------------------------------
# 人脸视觉融合（低透明度叠加，使 AI 检测器识别人脸）
# ---------------------------------------------------------------------------


def blend_face_into_image(
    face_path: str | Path,
    carrier_path: str | Path,
    output_path: str | Path,
    alpha: float = 0.10,
    face_scale: float = 0.15,
    position: str = "auto",
    blur_radius: int = 15,
    contrast: float = 1.0,
    output_size: int = 640,
    auto_alpha: bool = True,
    feather: int = 0,
    color_match: bool = False,
    blend_mode: str = "poisson",
    hf_scale: float = 1.0,
) -> None:
    """将人脸融合进载体图片，使 AI 人脸检测可识别。

    融合策略（按隐蔽性从强到弱）：
      adaptive (推荐): 纹理自适应高频嵌入——仅在载体纹理复杂处叠加人脸高频分量，
                       光滑区域（天空、墙面）几乎不变，视觉隐蔽性最强。
                       建议配合 face_scale≥0.35 以分散每像素变化量。
      poisson        : 泊松无缝克隆——用泊松方程将人脸梯度场（五官边缘结构）
                       无缝融入背景，颜色完全由背景决定，肉眼看不到色块边界。
      lum            : Lab 亮度融合——只叠加明暗结构，保留背景颜色。
      hf             : 高频叠加——只叠加边缘/细节，保留背景色调。
      full           : 传统 alpha 融合（auto_alpha 自动搜索最低可检测透明度）。

    Args:
        face_path: 人脸照片路径。
        carrier_path: 载体（风景）图片路径。
        output_path: 输出图片路径（建议 PNG）。
        alpha: 融合强度起始值。full/lum/hf 默认 0.10；adaptive 自动从 0.05 搜索。
        face_scale: 人脸占输出边长比例。默认 0.15；adaptive 推荐 0.35+。
        position: 放置位置。默认 auto=优先纹理丰富且色差较小的区域。
        blur_radius: 载体覆盖区高斯模糊半径。默认 15（仅 full 有效）。
        contrast: 人脸对比度增强倍数。默认 1.0。
        output_size: 输出正方形边长。默认 640。
        auto_alpha: 自动调参（MTCNN 通过即停）。默认 True。
        feather: 羽化/遮罩内缩像素。0=自动。
        color_match: 色调匹配（仅 full 模式有效）。默认 False。
        blend_mode: 融合模式 adaptive/poisson/lum/hf/full。默认 poisson。
        hf_scale: hf/adaptive 模式高频分量缩放系数。默认 1.0。

    Raises:
        FileNotFoundError: 图片路径不存在。
    """
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    from hack_face.face import crop_face, detect_faces_haar

    face_path = Path(face_path)
    carrier_path = Path(carrier_path)
    if not face_path.exists():
        raise FileNotFoundError(f"人脸图片不存在：{face_path}")
    if not carrier_path.exists():
        raise FileNotFoundError(f"载体图片不存在：{carrier_path}")

    carrier = Image.open(carrier_path).convert("RGB")

    # Step 0: 载体裁剪/缩放为正方形
    cw, ch = carrier.size
    short_side = min(cw, ch)
    left = (cw - short_side) // 2
    top = (ch - short_side) // 2
    carrier = carrier.crop((left, top, left + short_side, top + short_side))
    carrier = carrier.resize((output_size, output_size), Image.LANCZOS)
    cw, ch = output_size, output_size

    # Step 1: 裁剪纯人脸区域
    face_cropped = crop_face(face_path, margin=60)
    if face_cropped is None:
        print("⚠ 源图片未检测到人脸，将使用整张图片融合")
        face_cropped = Image.open(face_path).convert("RGB")
    else:
        print(f"  已裁剪人脸区域：{face_cropped.size[0]}×{face_cropped.size[1]}")

    fw, fh = face_cropped.size

    # 缩放人脸
    target_h = int(output_size * face_scale)
    scale_ratio = target_h / fh
    target_w = int(fw * scale_ratio)
    target_w = min(target_w, cw)
    target_h = min(target_h, ch)
    face_resized = face_cropped.resize((target_w, target_h), Image.LANCZOS)

    # 计算放置位置（边距 3%）
    pad = int(output_size * 0.03)
    positions = {
        "center": ((cw - target_w) // 2, (ch - target_h) // 2),
        "top-left": (pad, pad),
        "top-right": (cw - target_w - pad, pad),
        "bottom-left": (pad, ch - target_h - pad),
        "bottom-right": (cw - target_w - pad, ch - target_h - pad),
    }
    if position == "auto":
        # 自动选位：在色差较小的候选区中优先选纹理能量最高的区域
        # 纹理丰富区可更好地掩盖人脸信号（视觉遮蔽效应）
        import cv2 as _cv2_pos

        face_mean = np.array(face_resized, dtype=np.float32).mean(axis=(0, 1))
        carrier_np = np.array(carrier, dtype=np.float32)

        # 计算局部色差（与人脸均值的差）
        diff_sq = np.zeros(carrier_np.shape[:2], dtype=np.float32)
        for _c in range(3):
            local_mean = _cv2_pos.blur(carrier_np[:, :, _c], (target_w, target_h))
            diff_sq += (local_mean - face_mean[_c]) ** 2

        # 计算局部纹理能量（局部标准差均值）
        _tex_k = max(11, (min(target_w, target_h) // 6) | 1)
        if _tex_k % 2 == 0:
            _tex_k += 1
        texture_energy = np.zeros(carrier_np.shape[:2], dtype=np.float32)
        for _c in range(3):
            _lm = _cv2_pos.boxFilter(
                carrier_np[:, :, _c], -1, (_tex_k, _tex_k), normalize=True
            )
            _lsq = _cv2_pos.boxFilter(
                carrier_np[:, :, _c] ** 2, -1, (_tex_k, _tex_k), normalize=True
            )
            texture_energy += np.maximum(_lsq - _lm**2, 0.0)
        texture_energy = np.sqrt(texture_energy / 3.0)

        # 有效中心点范围（留出边距）
        cy_min = pad + target_h // 2
        cy_max = ch - target_h // 2 - pad
        cx_min = pad + target_w // 2
        cx_max = cw - target_w // 2 - pad

        # 综合评分：纹理高→视觉遮蔽强；色差小→不突兀（score 越大越好）
        valid_diff = diff_sq[cy_min:cy_max, cx_min:cx_max]
        valid_tex = texture_energy[cy_min:cy_max, cx_min:cx_max]
        _d_max = float(valid_diff.max()) + 1e-6
        _t_max = float(valid_tex.max()) + 1e-6
        score = valid_tex / _t_max - 0.3 * valid_diff / _d_max
        best_vy, best_vx = np.unravel_index(np.argmax(score), score.shape)
        x = (cx_min + best_vx) - target_w // 2
        y = (cy_min + best_vy) - target_h // 2
        print(f"  自动选位：({x}, {y}) — 纹理丰富且色差较小")
    else:
        x, y = positions.get(position, positions["bottom-right"])

    # ── 泊松无缝克隆（默认，最隐蔽）────────────────────────────────────────
    if blend_mode == "poisson":
        import cv2 as _cv2_p

        if contrast != 1.0:
            face_resized = ImageEnhance.Contrast(face_resized).enhance(contrast)

        # 椭圆遮罩：内缩 feather 像素避免边缘溢出
        if feather == 0:
            feather = max(int(min(target_w, target_h) * 0.12), 3)
        pmask = np.zeros((target_h, target_w), dtype=np.uint8)
        _cv2_p.ellipse(
            pmask,
            (target_w // 2, target_h // 2),
            (max(target_w // 2 - feather, 1), max(target_h // 2 - feather, 1)),
            0,
            0,
            360,
            255,
            -1,
        )

        face_bgr = np.array(face_resized, dtype=np.uint8)[:, :, ::-1].copy()
        carrier_bgr = np.array(carrier, dtype=np.uint8)[:, :, ::-1].copy()
        center_pt = (x + target_w // 2, y + target_h // 2)

        import torch
        from facenet_pytorch import MTCNN as _MTCNN

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _mtcnn_low = _MTCNN(device=_device, keep_all=True, thresholds=[0.3, 0.4, 0.4])

        def _mtcnn_check(img: Image.Image) -> tuple[bool, float]:
            ok, prob = False, 0.0
            boxes, probs = _mtcnn_low.detect(img)
            if boxes is not None:
                for box, p in zip(boxes, probs):
                    bx1, by1, bx2, by2 = box.astype(int)
                    ix1 = max(bx1, x)
                    iy1 = max(by1, y)
                    ix2 = min(bx2, x + target_w)
                    iy2 = min(by2, y + target_h)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if inter / max((bx2 - bx1) * (by2 - by1), 1) > 0.3 and p > 0.3:
                            ok, prob = True, float(p)
                            break
            return ok, prob

        def _haar_check(img: Image.Image) -> bool:
            for (bx1, by1, bx2, by2), _ in detect_faces_haar(img):
                ix1 = max(bx1, x)
                iy1 = max(by1, y)
                ix2 = min(bx2, x + target_w)
                iy2 = min(by2, y + target_h)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter / max((bx2 - bx1) * (by2 - by1), 1) > 0.3:
                        return True
            return False

        poisson_ok = False
        result = None
        carrier_arr_f = np.array(carrier, dtype=np.float64)  # 原始载体用于混合

        # 先尝试 MIXED_CLONE（脸颊平坦区透出背景纹理，视觉最隐蔽）
        for clone_flag, flag_name in [
            (_cv2_p.MIXED_CLONE, "MIXED"),
            (_cv2_p.NORMAL_CLONE, "NORMAL"),
        ]:
            try:
                out_bgr = _cv2_p.seamlessClone(
                    face_bgr, carrier_bgr, pmask, center_pt, clone_flag
                )
                poisson_full = Image.fromarray(out_bgr[:, :, ::-1])
            except Exception as e:
                print(f"  ⚠ 泊松 {flag_name}_CLONE 异常：{e}")
                continue

            poisson_arr = np.array(poisson_full, dtype=np.float64)

            # 泊松完整克隆后，搜索最小 mix 使 MTCNN 刚好通过
            # mix=1.0 = 纯泊松结果；mix=0.5 = 泊松与原始载体各半
            # 越小人脸结构越弱，视觉越隐蔽
            for mix in np.arange(0.05, 1.01, 0.025):
                mix = round(float(mix), 2)
                blended_arr = mix * poisson_arr + (1.0 - mix) * carrier_arr_f
                candidate = Image.fromarray(
                    np.clip(blended_arr, 0, 255).astype(np.uint8)
                )
                mtcnn_ok, mtcnn_prob = _mtcnn_check(candidate)
                if mtcnn_ok:
                    haar_ok = _haar_check(candidate)
                    haar_tag = "Haar: ✓" if haar_ok else "Haar: ✗"
                    print(
                        f"  ✓ 泊松 {flag_name}_CLONE mix={mix:.2f} 通过"
                        f"（prob={mtcnn_prob:.4f}, {haar_tag}）"
                    )
                    result = candidate
                    poisson_ok = True
                    break
                else:
                    if mix >= 0.70:
                        print(
                            f"  · 泊松 {flag_name}_CLONE mix={mix:.3f} prob={mtcnn_prob:.4f} 未通过"
                        )
            if poisson_ok:
                break
            print(f"  · {flag_name}_CLONE 所有 mix 均未通过，继续尝试")

        if not poisson_ok:
            print(f"  ⚠ 泊松克隆两种模式均未通过，退回 full 模式")
            blend_mode = "full"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(str(output_path))
            print(f"✓ 已保存融合图片：{output_path}")
            print(
                f"  输出: {output_size}×{output_size} | 泊松无缝克隆 | "
                f"contrast={contrast}x | 人脸={target_w}×{target_h} | 位置=({x},{y})"
            )
            return

    # ── alpha 混合模式（full / lum / hf）────────────────────────────────────
    # Step 2: 增强对比度
    if contrast != 1.0:
        face_resized = ImageEnhance.Contrast(face_resized).enhance(contrast)

    # Step 2b: color_match（仅 full 模式）
    if color_match and blend_mode == "full":
        carrier_region = np.array(
            carrier.crop((x, y, x + target_w, y + target_h)), dtype=np.float64
        )
        face_f = np.array(face_resized, dtype=np.float64)
        for c in range(3):
            src_mean, src_std = face_f[:, :, c].mean(), face_f[:, :, c].std() + 1e-6
            tgt_mean, tgt_std = (
                carrier_region[:, :, c].mean(),
                carrier_region[:, :, c].std() + 1e-6,
            )
            matched = (face_f[:, :, c] - src_mean) / src_std * tgt_std + tgt_mean
            face_f[:, :, c] = 0.5 * face_f[:, :, c] + 0.5 * matched
        face_resized = Image.fromarray(np.clip(face_f, 0, 255).astype(np.uint8))

    # Step 3: 预处理融合数据
    import cv2 as _cv2

    face_rgb = np.array(face_resized, dtype=np.uint8)  # H×W×3 uint8 RGB

    if blend_mode == "lum":
        # Lab 亮度融合：只提取人脸 L 通道，背景 a/b 通道完全保留
        face_lab = _cv2.cvtColor(face_rgb[:, :, ::-1], _cv2.COLOR_BGR2Lab).astype(
            np.float64
        )
        face_L = face_lab[:, :, 0]  # 0-255 范围的亮度通道
    elif blend_mode == "hf":
        # 高频分量：人脸 - 低频模糊版本（均值约 0 的边缘信息）
        sigma = max(int(min(target_w, target_h) * 0.15), 3)
        face_low = face_resized.filter(ImageFilter.GaussianBlur(radius=sigma))
        face_hf = np.array(face_resized, dtype=np.float64) - np.array(
            face_low, dtype=np.float64
        )
    elif blend_mode == "adaptive":
        # 纹理自适应 HF：用较小 σ（6% 人脸短边）保留中高频五官结构（眼眉鼻唇边缘）
        # σ 越小高频越丰富，AI 识别越容易；同时每像素变化量通过纹理遮罩控制
        sigma_a = max(int(min(target_w, target_h) * 0.06), 3)
        ksize_a = sigma_a * 6 + 1
        face_bgr_float = face_rgb[:, :, ::-1].astype(np.float64)
        face_blur_a = _cv2.GaussianBlur(face_bgr_float, (ksize_a, ksize_a), sigma_a)
        face_hf_adapt = (face_bgr_float - face_blur_a)[:, :, ::-1]  # RGB HF，均值≈0
    else:  # full
        face_arr_full = np.array(face_resized, dtype=np.float64)

    # Step 4: 构建羽化遮罩
    if feather == 0:
        feather = max(int(min(target_w, target_h) * 0.18), 2)
    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([feather, feather, target_w - feather, target_h - feather], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    feather_mask = np.array(mask, dtype=np.float64) / 255.0  # shape: (h, w)

    # 预计算模糊后的载体
    carrier_blurred = None
    if blur_radius > 0:
        carrier_blurred = carrier.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _build_result(a: float) -> Image.Image:
        """用指定 alpha 值生成融合图。"""
        carrier_arr = np.array(carrier, dtype=np.float64)
        if carrier_blurred is not None and blend_mode == "full":
            # full 模式才模糊载体（lum/hf 模式保留原始载体纹理更自然）
            cb_arr = np.array(carrier_blurred, dtype=np.float64)
            carrier_arr[y : y + target_h, x : x + target_w] = cb_arr[
                y : y + target_h, x : x + target_w
            ]
        region = carrier_arr[y : y + target_h, x : x + target_w]  # float64 RGB
        eff = a * feather_mask[:, :, np.newaxis]  # (h, w, 1)

        if blend_mode == "lum":
            # 将载体区域转 Lab，替换 L 通道，转回 RGB
            region_bgr_u8 = np.clip(region, 0, 255).astype(np.uint8)[:, :, ::-1]
            region_lab = _cv2.cvtColor(region_bgr_u8, _cv2.COLOR_BGR2Lab).astype(
                np.float64
            )
            # 在 Lab 空间中对 L 通道做 alpha 融合（保留 a/b 颜色通道）
            region_lab[:, :, 0] = (1 - eff[:, :, 0]) * region_lab[:, :, 0] + eff[
                :, :, 0
            ] * face_L
            region_lab_u8 = np.clip(region_lab, 0, 255).astype(np.uint8)
            blended_bgr = _cv2.cvtColor(region_lab_u8, _cv2.COLOR_Lab2BGR)
            blended = blended_bgr[:, :, ::-1].astype(np.float64)  # BGR→RGB
        elif blend_mode == "hf":
            # 加性融合高频分量（无色调变化，只叠加边缘/纹理）
            blended = np.clip(region + face_hf * eff * hf_scale, 0, 255)
        elif blend_mode == "adaptive":
            # 纹理自适应：用载体区域局部标准差作为视觉遮蔽权重
            # 在纹理丰富处叠加人脸高频；光滑处（天空/墙面）几乎不变
            _k = max(9, (min(target_w, target_h) // 8) | 1)
            if _k % 2 == 0:
                _k += 1
            r32 = region.astype(np.float32)
            _lmean = _cv2.boxFilter(r32, -1, (_k, _k), normalize=True).astype(
                np.float64
            )
            _lsqm = _cv2.boxFilter((r32**2), -1, (_k, _k), normalize=True).astype(
                np.float64
            )
            _lvar = np.maximum(_lsqm - _lmean**2, 0.0)
            _tex_e = np.sqrt(_lvar.mean(axis=2))  # (h, w)
            _t_max = _tex_e.max()
            _tex_norm = (
                (_tex_e / _t_max)
                if _t_max > 1e-6
                else np.ones((target_h, target_w), dtype=np.float64)
            )
            # 有效强度 = a × 纹理权重 × 羽化遮罩；光滑区趋近 0，纹理区趋近 a
            eff_adapt = a * _tex_norm[:, :, np.newaxis] * feather_mask[:, :, np.newaxis]
            blended = np.clip(region + face_hf_adapt * eff_adapt * hf_scale, 0, 255)
        else:  # full
            blended = (1 - eff) * region + eff * face_arr_full

        carrier_arr[y : y + target_h, x : x + target_w] = np.clip(blended, 0, 255)
        return Image.fromarray(carrier_arr.astype(np.uint8))

    # Step 3: 自动调参或固定 alpha
    if auto_alpha:
        import torch
        from facenet_pytorch import MTCNN as _MTCNN

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _mtcnn_low = _MTCNN(device=_device, keep_all=True, thresholds=[0.3, 0.4, 0.4])

        def _detect_in_region(img: Image.Image) -> tuple[bool, float, bool]:
            """检测融合区域内的人脸。

            以 MTCNN 为主要判据（公司系统实测用深度学习模型），
            Haar 仅作为辅助参考。

            Returns:
                (mtcnn_ok, mtcnn_prob, haar_ok)
            """
            # MTCNN 低阈值检测 — 主判据
            mtcnn_ok = False
            mtcnn_prob = 0.0
            boxes, probs = _mtcnn_low.detect(img)
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    bx1, by1, bx2, by2 = box.astype(int)
                    ix1, iy1 = max(bx1, x), max(by1, y)
                    ix2, iy2 = min(bx2, x + target_w), min(by2, y + target_h)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        box_area = max((bx2 - bx1) * (by2 - by1), 1)
                        if inter / box_area > 0.3 and prob > 0.3:
                            mtcnn_ok = True
                            mtcnn_prob = float(prob)
                            break

            # Haar 检测 — 辅助参考
            haar_ok = False
            for (bx1, by1, bx2, by2), _ in detect_faces_haar(img):
                ix1, iy1 = max(bx1, x), max(by1, y)
                ix2, iy2 = min(bx2, x + target_w), min(by2, y + target_h)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    haar_area = max((bx2 - bx1) * (by2 - by1), 1)
                    if inter / haar_area > 0.3:
                        haar_ok = True
                        break

            return mtcnn_ok, mtcnn_prob, haar_ok

        # 从指定 alpha 开始搜索，MTCNN 通过即停
        # adaptive 模式：从极小值起步，步进 0.05，允许搜索到 3.0（纹理遮罩使有效强度低）
        # 其他模式：从 alpha 起步，步进 0.02，最大 1.0
        if blend_mode == "adaptive":
            _search_start = min(alpha, 0.05)
            _search_range = np.arange(_search_start, 3.01, 0.05)
            _max_alpha = 3.0
        else:
            _search_range = np.arange(alpha, 1.01, 0.02)
            _max_alpha = 1.0
        best_alpha = alpha
        result = None
        _log_cnt = 0
        for test_alpha in _search_range:
            test_alpha = round(float(test_alpha), 4)
            candidate = _build_result(test_alpha)
            mtcnn_ok, mtcnn_prob, haar_ok = _detect_in_region(candidate)

            if mtcnn_ok:
                best_alpha = test_alpha
                result = candidate
                haar_tag = "Haar: ✓" if haar_ok else "Haar: ✗"
                print(
                    f"  ✓ alpha={test_alpha:.4f} MTCNN 通过"
                    f"（prob={mtcnn_prob:.4f}, {haar_tag}）"
                )
                break
            _log_cnt += 1
            if _log_cnt % 10 == 0:
                print(f"  · alpha={test_alpha:.4f} 未通过（prob={mtcnn_prob:.4f}）")
        else:
            best_alpha = _max_alpha
            result = _build_result(_max_alpha)
            print(f"  ⚠ 自动调参到 alpha={_max_alpha} 仍未通过 MTCNN")
        alpha = best_alpha
    else:
        result = _build_result(alpha)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    print(f"✓ 已保存融合图片：{output_path}")
    print(
        f"  输出: {output_size}×{output_size} | alpha={alpha} | blur={blur_radius} | "
        f"contrast={contrast}x | 人脸={target_w}×{target_h} | 位置={position}"
    )
