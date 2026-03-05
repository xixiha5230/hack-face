"""生成最优方案的视觉对比图（原图 vs 融合结果 vs 10×差异放大图）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SWEEP_DIR = Path(__file__).parent.parent / "images/output/sweep"
OUT_DIR = Path(__file__).parent.parent / "images/output"
REF = SWEEP_DIR / "_carrier_ref.png"

# 按 SSIM 排序最优方案（取 Top 6）
TOP = [
    ("poisson_fs0.25", 0.99983, 56.98, 0.042, 12),
    ("adaptive_fs0.30_hs1.0", 0.99966, 52.93, 0.082, 29),
    ("hf_fs0.30_hs1.0", 0.99956, 53.42, 0.102, 13),
    ("hf_fs0.30_hs1.5", 0.99952, 53.12, 0.105, 13),
    ("adaptive_fs0.30_hs2.0", 0.99959, 52.09, 0.089, 32),
    ("poisson_fs0.35", 0.99822, 48.01, 0.157, 26),
]


def put_text(img: np.ndarray, text: str, y: int = 18, scale: float = 0.52) -> None:
    cv2.putText(
        img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (6, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


ref_bgr = cv2.imread(str(REF))
H, W = ref_bgr.shape[:2]  # 640×640

rows = []
for label, ssim, psnr, mad, maxd in TOP:
    src = SWEEP_DIR / f"{label}.png"
    if not src.exists():
        print(f"跳过 {label}（文件不存在）")
        continue
    blended = cv2.imread(str(src))

    # 差异图 ×10 放大
    diff = np.abs(blended.astype(np.int32) - ref_bgr.astype(np.int32))
    diff_amp = np.clip(diff * 10, 0, 255).astype(np.uint8)

    # 拼三列
    row_img = np.zeros((H, W * 3 + 8, 3), dtype=np.uint8)
    row_img[:, :W] = ref_bgr
    row_img[:, W + 4 : W * 2 + 4] = blended
    row_img[:, W * 2 + 8 : W * 3 + 8] = diff_amp

    # 标注
    put_text(row_img, "原始载体", 22)
    put_text(row_img, f"融合结果  [{label}]", 22, 0.50)
    cv2.putText(
        row_img,
        f"融合结果  [{label}]",
        (W + 4 + 6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        row_img,
        f"融合结果  [{label}]",
        (W + 4 + 6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        row_img,
        f"差异×10放大  SSIM={ssim:.5f} PSNR={psnr:.1f}dB MAD={mad:.3f} MaxD={maxd}",
        (W * 2 + 8 + 6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        row_img,
        f"差异×10放大  SSIM={ssim:.5f} PSNR={psnr:.1f}dB MAD={mad:.3f} MaxD={maxd}",
        (W * 2 + 8 + 6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    rows.append(row_img)
    print(f"✓ {label}")

if rows:
    gap = np.full((4, rows[0].shape[1], 3), 60, dtype=np.uint8)
    combined = rows[0]
    for r in rows[1:]:
        combined = np.vstack([combined, gap, r])
    out = OUT_DIR / "comparison_top6.png"
    cv2.imwrite(str(out), combined)
    print(f"\n✓ 对比图已保存：{out}")
    print(f"  尺寸：{combined.shape[1]}×{combined.shape[0]}")
