"""参数扫描模块：自动搜索最优的视觉隐蔽 + AI 可识别参数组合。

主入口：
  run_sweep(face_path, carrier_path, output_dir, output_size, top_n)
    → 返回按 SSIM 降序排列的 top_n 个通过 AI 检测的结果列表
    → 同时在 output_dir 保存各结果 PNG 和对比图
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from hack_face.metrics import detect_prob, image_metrics
from hack_face.watermark import blend_face_into_image

# ---------------------------------------------------------------------------
# 测试用例模板
# ---------------------------------------------------------------------------


def _build_cases(output_size: int) -> list[dict[str, Any]]:
    """构建全量参数组合用例。"""
    cases: list[dict[str, Any]] = []

    # poisson 模式：新 mix 下界 0.05，小 face_scale 视觉最干净
    for fs in [0.20, 0.25, 0.30, 0.35, 0.45]:
        cases.append(
            dict(
                label=f"poisson_fs{fs:.2f}",
                blend_mode="poisson",
                face_scale=fs,
                hf_scale=1.0,
                alpha=0.05,
                auto_alpha=True,
                output_size=output_size,
                feather=0,
                blur_radius=0,
                contrast=1.0,
                color_match=False,
            )
        )

    # adaptive 模式：纹理自适应 HF
    for fs in [0.25, 0.30, 0.40, 0.50]:
        for hs in [1.0, 1.5, 2.0]:
            cases.append(
                dict(
                    label=f"adaptive_fs{fs:.2f}_hs{hs:.1f}",
                    blend_mode="adaptive",
                    face_scale=fs,
                    hf_scale=hs,
                    alpha=0.05,
                    auto_alpha=True,
                    output_size=output_size,
                    feather=0,
                    blur_radius=0,
                    contrast=1.0,
                    color_match=False,
                )
            )

    # hf 模式：高频叠加
    for fs in [0.25, 0.30, 0.40]:
        for hs in [1.0, 1.5]:
            cases.append(
                dict(
                    label=f"hf_fs{fs:.2f}_hs{hs:.1f}",
                    blend_mode="hf",
                    face_scale=fs,
                    hf_scale=hs,
                    alpha=0.01,
                    auto_alpha=True,
                    output_size=output_size,
                    feather=0,
                    blur_radius=0,
                    contrast=1.0,
                    color_match=False,
                )
            )

    # lum 模式：Lab 亮度融合（通常需要较高 alpha，作为对照）
    for fs in [0.25, 0.30]:
        cases.append(
            dict(
                label=f"lum_fs{fs:.2f}",
                blend_mode="lum",
                face_scale=fs,
                hf_scale=1.0,
                alpha=0.01,
                auto_alpha=True,
                output_size=output_size,
                feather=0,
                blur_radius=0,
                contrast=1.0,
                color_match=False,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# 参考图缓存：将原载体裁剪缩放到 output_size 正方形
# ---------------------------------------------------------------------------


def _make_ref(carrier_path: Path, output_dir: Path, output_size: int) -> Path:
    ref_path = output_dir / "_ref.png"
    if not ref_path.exists():
        img = Image.open(carrier_path).convert("RGB")
        cw, ch = img.size
        ss = min(cw, ch)
        img = img.crop(((cw - ss) // 2, (ch - ss) // 2, (cw + ss) // 2, (ch + ss) // 2))
        img = img.resize((output_size, output_size), Image.LANCZOS)
        img.save(str(ref_path))
    return ref_path


# ---------------------------------------------------------------------------
# 对比图生成
# ---------------------------------------------------------------------------


def _make_comparison(
    ref_path: Path,
    top_results: list[dict],
    output_dir: Path,
) -> Path:
    """生成 Top-N 对比图：每行 = 原图 | 融合结果 | ×10差异放大。"""
    ref_bgr = cv2.imread(str(ref_path))
    H, W = ref_bgr.shape[:2]
    gap_h = 6
    col_gap = 6

    def _text(img: np.ndarray, line1: str, line2: str = "") -> None:
        for text, y in [(line1, 22), (line2, 44)]:
            if not text:
                continue
            cv2.putText(
                img,
                text,
                (6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                text,
                (6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    rows = []
    for rank, r in enumerate(top_results, 1):
        img_path = output_dir / f"{r['label']}.png"
        if not img_path.exists():
            continue
        blended = cv2.imread(str(img_path))
        diff_amp = np.clip(
            np.abs(blended.astype(np.int32) - ref_bgr.astype(np.int32)) * 10, 0, 255
        ).astype(np.uint8)

        row = np.zeros((H, W * 3 + col_gap * 2, 3), dtype=np.uint8)

        # 列1：原图
        orig_col = ref_bgr.copy()
        _text(orig_col, "原始载体")
        row[:, :W] = orig_col

        # 列2：融合结果（带排名标注）
        blend_col = blended.copy()
        ai_tag = f"AI {'✓' if r['detected_low'] else '✗'} prob={r['prob']:.3f}"
        _text(blend_col, f"#{rank} {r['label']}", ai_tag)
        row[:, W + col_gap : W * 2 + col_gap] = blend_col

        # 列3：差异×10放大（带指标）
        diff_col = diff_amp.copy()
        _text(
            diff_col,
            f"差异×10  SSIM={r['ssim']:.5f}",
            f"PSNR={r['psnr']:.1f}dB  MAD={r['mad']:.3f}  MaxD={r['max_diff']}",
        )
        row[:, W * 2 + col_gap * 2 :] = diff_col

        rows.append(row)

    if not rows:
        return output_dir / "comparison.png"

    gap = np.full((gap_h, rows[0].shape[1], 3), 50, dtype=np.uint8)
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.vstack([canvas, gap, r])

    out_path = output_dir / "comparison_top.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


# ---------------------------------------------------------------------------
# 主扫描函数
# ---------------------------------------------------------------------------


def run_sweep(
    face_path: str | Path,
    carrier_path: str | Path,
    output_dir: str | Path,
    output_size: int = 640,
    top_n: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """执行全量参数扫描，返回 top_n 个最优结果（按 SSIM 降序）。

    Args:
        face_path: 人脸图片路径。
        carrier_path: 载体图片路径。
        output_dir: 结果输出目录（自动创建）。
        output_size: 输出正方形边长，默认 640。
        top_n: 返回/保存的最优方案数量，默认 5。
        verbose: 是否打印详细进度，默认 True。

    Returns:
        按 SSIM 降序排列的结果字典列表，每项包含：
        label, mode, face_scale, hf_scale, detected_low, detected_def,
        prob, ssim, psnr, mad, max_diff, elapsed, output_path
    """
    face_path = Path(face_path)
    carrier_path = Path(carrier_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = _make_ref(carrier_path, output_dir, output_size)
    cases = _build_cases(output_size)
    total = len(cases)
    results: list[dict] = []

    for i, case in enumerate(cases, 1):
        label = case["label"]
        out_path = output_dir / f"{label}.png"
        if verbose:
            print(f"\n[{i}/{total}] {label}")

        t0 = time.time()
        try:
            blend_face_into_image(
                face_path=face_path,
                carrier_path=carrier_path,
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
            if verbose:
                print(f"  ✗ 失败: {e}")
            continue

        elapsed = time.time() - t0
        if not out_path.exists():
            continue

        m = image_metrics(ref_path, out_path)
        low_ok, low_prob, def_ok = detect_prob(out_path)

        r = {
            "label": label,
            "mode": case["blend_mode"],
            "face_scale": case["face_scale"],
            "hf_scale": case["hf_scale"],
            "detected_low": low_ok,
            "detected_def": def_ok,
            "prob": low_prob,
            "ssim": m.get("ssim", 0.0),
            "psnr": m.get("psnr", 0.0),
            "mad": m.get("mad", 0.0),
            "max_diff": m.get("max_diff", 0),
            "elapsed": elapsed,
            "output_path": str(out_path),
        }
        results.append(r)

        if verbose:
            status = "✓ AI通过" if low_ok else "✗ AI未通过"
            print(
                f"  {status} | prob={low_prob:.4f} | SSIM={m.get('ssim', 0):.5f} | "
                f"PSNR={m.get('psnr', 0):.2f}dB | MAD={m.get('mad', 0):.3f} | "
                f"MaxDiff={m.get('max_diff', 0)} | {elapsed:.1f}s"
            )

    # 已通过 AI 检测的按 SSIM 降序，未通过的排最后
    passed = sorted(
        [r for r in results if r["detected_low"]], key=lambda x: x["ssim"], reverse=True
    )
    failed = [r for r in results if not r["detected_low"]]
    ranked = (passed + failed)[: top_n * 2]  # 保留更多供参考

    top = passed[:top_n]

    # 生成对比图
    if top:
        comp_path = _make_comparison(ref_path, top, output_dir)
        if verbose:
            print(f"\n✓ 对比图已保存：{comp_path}")

    return top
