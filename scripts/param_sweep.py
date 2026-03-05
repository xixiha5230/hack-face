"""参数扫描测试：找出视觉最隐蔽 + AI 可识别的最优参数组合。

评测指标（越高越隐蔽）：
  SSIM   : 结构相似度（1.0=完全一致）
  PSNR   : 峰值信噪比（dB）
  MaxDiff: 最大像素差（0-255，越低越好）
  MAD    : 平均绝对差（越低越好）

AI 检测指标：
  MTCNN prob : MTCNN 低阈值检测置信度（>0.3 即通过）
  detected   : 是否通过检测
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 确保能找到 hack_face 包
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from hack_face.watermark import blend_face_into_image

# ── 路径配置 ────────────────────────────────────────────────────────────────
FACE_PATH = Path(__file__).parent.parent / "images/source/face.jpg"
CARRIER_PATH = Path(__file__).parent.parent / "images/target/target1.png"
OUT_DIR = Path(__file__).parent.parent / "images/output/sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── AI 检测器（全局单例）────────────────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mtcnn_low = MTCNN(device=_device, keep_all=True, thresholds=[0.3, 0.4, 0.4])
_mtcnn_def = MTCNN(device=_device, keep_all=True)


def detect_prob(img_path: str | Path) -> tuple[bool, float, bool]:
    """返回 (低阈值通过, 最高prob, 默认阈值通过)"""
    img = Image.open(img_path).convert("RGB")
    boxes_l, probs_l = _mtcnn_low.detect(img)
    boxes_d, probs_d = _mtcnn_def.detect(img)

    low_ok = False
    low_prob = 0.0
    if boxes_l is not None and probs_l is not None:
        valid = [p for p in probs_l if p is not None and p > 0.3]
        if valid:
            low_ok = True
            low_prob = float(max(valid))

    def_ok = boxes_d is not None and len(boxes_d) > 0

    return low_ok, low_prob, def_ok


def image_metrics(path_a: str | Path, path_b: str | Path) -> dict:
    """计算两图视觉差异指标（a=载体原图, b=融合结果）"""
    a = cv2.imread(str(path_a))
    b = cv2.imread(str(path_b))
    if a is None or b is None:
        return {}
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    mad = float(diff.mean())
    max_diff = int(diff.max())

    # PSNR
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")

    # SSIM (简化版: 逐通道均值再平均)
    ssim_val = _ssim(a, b)

    return {"ssim": ssim_val, "psnr": psnr, "mad": mad, "max_diff": max_diff}


def _ssim(a: np.ndarray, b: np.ndarray, k1=0.01, k2=0.03, win=11) -> float:
    """简化 SSIM（逐通道计算后平均）"""
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


# ── 测试用例 ────────────────────────────────────────────────────────────────
CASES: list[dict] = []

# adaptive 模式：大 face_scale（分散每像素变化）× hf_scale 组合
for fs in [0.30, 0.40, 0.50]:
    for hs in [1.0, 1.5, 2.0]:
        CASES.append(
            dict(
                label=f"adaptive_fs{fs:.2f}_hs{hs:.1f}",
                blend_mode="adaptive",
                face_scale=fs,
                hf_scale=hs,
                alpha=0.05,
                auto_alpha=True,
                output_size=640,
                feather=0,
                blur_radius=0,
                contrast=1.0,
                color_match=False,
            )
        )

# poisson 模式：不同 face_scale（新 mix 下界=0.05）
for fs in [0.25, 0.35, 0.45]:
    CASES.append(
        dict(
            label=f"poisson_fs{fs:.2f}",
            blend_mode="poisson",
            face_scale=fs,
            hf_scale=1.0,
            alpha=0.05,
            auto_alpha=True,
            output_size=640,
            feather=0,
            blur_radius=0,
            contrast=1.0,
            color_match=False,
        )
    )

# lum 模式：亮度融合
for fs in [0.30, 0.40]:
    CASES.append(
        dict(
            label=f"lum_fs{fs:.2f}",
            blend_mode="lum",
            face_scale=fs,
            hf_scale=1.0,
            alpha=0.01,
            auto_alpha=True,
            output_size=640,
            feather=0,
            blur_radius=0,
            contrast=1.0,
            color_match=False,
        )
    )

# hf 模式：高频叠加（去除了原来的 *2.0）
for fs in [0.30, 0.40]:
    for hs in [1.0, 1.5]:
        CASES.append(
            dict(
                label=f"hf_fs{fs:.2f}_hs{hs:.1f}",
                blend_mode="hf",
                face_scale=fs,
                hf_scale=hs,
                alpha=0.01,
                auto_alpha=True,
                output_size=640,
                feather=0,
                blur_radius=0,
                contrast=1.0,
                color_match=False,
            )
        )


# ── 执行测试 ────────────────────────────────────────────────────────────────
def run_case(case: dict) -> dict | None:
    label = case["label"]
    out_path = OUT_DIR / f"{label}.png"
    print(f"\n{'=' * 60}")
    print(f"[{label}]")

    t0 = time.time()
    try:
        blend_face_into_image(
            face_path=FACE_PATH,
            carrier_path=CARRIER_PATH,
            output_path=out_path,
            blend_mode=case["blend_mode"],
            face_scale=case["face_scale"],
            hf_scale=case["hf_scale"],
            alpha=case["alpha"],
            auto_alpha=case["auto_alpha"],
            output_size=case["output_size"],
            feather=case["feather"],
            blur_radius=case["blur_radius"],
            contrast=case["contrast"],
            color_match=case["color_match"],
        )
    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
        return None

    elapsed = time.time() - t0

    if not out_path.exists():
        print("  ✗ 输出文件不存在")
        return None

    # 视觉指标（与原载体缩放为相同尺寸后对比）
    carrier_resized = OUT_DIR / "_carrier_ref.png"
    if not carrier_resized.exists():
        _ref = Image.open(CARRIER_PATH).convert("RGB")
        cw, ch = _ref.size
        ss = min(cw, ch)
        _ref = _ref.crop(
            ((cw - ss) // 2, (ch - ss) // 2, (cw + ss) // 2, (ch + ss) // 2)
        )
        _ref = _ref.resize((640, 640), Image.LANCZOS)
        _ref.save(str(carrier_resized))

    metrics = image_metrics(carrier_resized, out_path)

    # AI 检测
    low_ok, low_prob, def_ok = detect_prob(out_path)

    result = {
        "label": label,
        "mode": case["blend_mode"],
        "face_scale": case["face_scale"],
        "hf_scale": case["hf_scale"],
        "detected_low": low_ok,
        "detected_def": def_ok,
        "prob": low_prob,
        "ssim": metrics.get("ssim", 0),
        "psnr": metrics.get("psnr", 0),
        "mad": metrics.get("mad", 0),
        "max_diff": metrics.get("max_diff", 0),
        "elapsed": elapsed,
    }

    status = "✓ AI通过" if low_ok else "✗ AI未通过"
    print(
        f"  {status} | prob={low_prob:.4f} | SSIM={metrics.get('ssim', 0):.5f} | "
        f"PSNR={metrics.get('psnr', 0):.2f}dB | MAD={metrics.get('mad', 0):.3f} | "
        f"MaxDiff={metrics.get('max_diff', 0)} | {elapsed:.1f}s"
    )
    return result


def main() -> None:
    results = []
    for i, case in enumerate(CASES):
        print(f"\n进度: {i + 1}/{len(CASES)}")
        r = run_case(case)
        if r:
            results.append(r)

    # ── 汇总报告 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("汇总报告（仅 AI 通过的方案，按 SSIM 降序）")
    print("=" * 80)

    detected = [r for r in results if r["detected_low"]]
    detected.sort(key=lambda r: r["ssim"], reverse=True)

    print(
        f"{'标签':<32} {'SSIM':>7} {'PSNR':>7} {'MAD':>6} {'MaxD':>5} {'prob':>6} {'def':>4}"
    )
    print("-" * 80)
    for r in detected:
        def_tag = "✓" if r["detected_def"] else "✗"
        print(
            f"{r['label']:<32} {r['ssim']:>7.5f} {r['psnr']:>7.2f} "
            f"{r['mad']:>6.3f} {r['max_diff']:>5d} {r['prob']:>6.4f} {def_tag:>4}"
        )

    if not detected:
        print("  (无方案通过 AI 检测)")
    else:
        best = detected[0]
        print(f"\n★ 最优方案: {best['label']}")
        print(
            f"  SSIM={best['ssim']:.5f}  PSNR={best['psnr']:.2f}dB  "
            f"MAD={best['mad']:.3f}  MaxDiff={best['max_diff']}"
        )
        print(
            f"  AI置信度={best['prob']:.4f}  默认阈值通过={'是' if best['detected_def'] else '否'}"
        )
        print(f"  输出: {OUT_DIR / (best['label'] + '.png')}")

    print(
        "\n未通过 AI 检测的方案：",
        [r["label"] for r in results if not r["detected_low"]],
    )


if __name__ == "__main__":
    main()
