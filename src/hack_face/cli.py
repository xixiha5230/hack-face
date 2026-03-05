"""命令行接口模块。

提供以下命令：
  hack-face-run    —— 【推荐】输入两张图片，自动扫描所有参数组合，输出 Top-N 最优结果
  hack-face-blend  —— 单次融合（指定模式，自动调参）
  hack-face-embed  —— 从人脸图片提取特征向量，保存为 .npy 文件
  hack-face-encode —— 将人脸特征嵌入风景照（LSB 隐写）
  hack-face-decode —— 从带隐写图片中还原特征并与已知人脸对比
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# hack-face-embed
# ---------------------------------------------------------------------------


def embed() -> None:
    """提取人脸特征向量并保存为 .npy 文件。"""
    parser = argparse.ArgumentParser(
        prog="hack-face-embed",
        description="从人脸图片中提取 512 维特征向量，保存为 .npy 文件",
    )
    parser.add_argument(
        "--face",
        "-f",
        required=True,
        metavar="IMAGE",
        help="目标人脸图片路径（jpg/png）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="NPY_FILE",
        help="输出 .npy 文件路径（默认与输入同名，后缀替换为 .npy）",
    )
    args = parser.parse_args()

    from hack_face.face import get_face_embedding

    face_path = Path(args.face)
    out_path = Path(args.output) if args.output else face_path.with_suffix(".npy")

    print(f"→ 正在提取人脸特征：{face_path}")
    try:
        embedding = get_face_embedding(face_path)
    except ValueError as e:
        print(f"✗ 错误：{e}", file=sys.stderr)
        sys.exit(1)

    np.save(str(out_path), embedding)
    print(f"✓ 特征向量已保存：{out_path}  （维度：{embedding.shape}）")


# ---------------------------------------------------------------------------
# hack-face-encode
# ---------------------------------------------------------------------------


def encode() -> None:
    """将人脸特征向量嵌入风景照。"""
    parser = argparse.ArgumentParser(
        prog="hack-face-encode",
        description="将人脸特征以不可见水印形式嵌入载体图片",
    )
    parser.add_argument(
        "--face",
        "-f",
        required=True,
        metavar="IMAGE_OR_NPY",
        help="人脸图片路径（自动提取特征）或已有的 .npy 特征文件",
    )
    parser.add_argument(
        "--carrier",
        "-c",
        required=True,
        metavar="IMAGE",
        help="载体（风景）图片路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_with_hidden_face.png",
        metavar="PNG_FILE",
        help="输出图片路径（默认：output_with_hidden_face.png）",
    )
    args = parser.parse_args()

    from hack_face.face import get_face_embedding
    from hack_face.watermark import encode_face_into_image

    face_path = Path(args.face)
    if face_path.suffix.lower() == ".npy":
        print(f"→ 从 .npy 文件加载特征：{face_path}")
        embedding = np.load(str(face_path))
    else:
        print(f"→ 正在提取人脸特征：{face_path}")
        try:
            embedding = get_face_embedding(face_path)
        except ValueError as e:
            print(f"✗ 错误：{e}", file=sys.stderr)
            sys.exit(1)

    print(f"→ 正在将特征嵌入载体图片：{args.carrier}")
    try:
        encode_face_into_image(embedding, args.carrier, args.output)
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ 错误：{e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# hack-face-decode
# ---------------------------------------------------------------------------


def decode() -> None:
    """从带隐写图片中还原人脸特征，并可选地与已知人脸对比。"""
    parser = argparse.ArgumentParser(
        prog="hack-face-decode",
        description="从带隐写图片中还原人脸特征向量，并与已知人脸比对",
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        metavar="PNG_FILE",
        help="带隐写水印的图片路径",
    )
    parser.add_argument(
        "--known-face",
        "-k",
        default=None,
        metavar="IMAGE_OR_NPY",
        help="用于对比的已知人脸（图片或 .npy 文件，可选）",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="余弦相似度阈值（默认：0.6）",
    )
    parser.add_argument(
        "--save-vector",
        "-s",
        default=None,
        metavar="NPY_FILE",
        help="将还原的特征向量保存为 .npy 文件（可选）",
    )
    args = parser.parse_args()

    from hack_face.watermark import decode_face_from_image

    print(f"→ 正在从图片中提取隐藏特征：{args.image}")
    try:
        extracted = decode_face_from_image(args.image)
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ 错误：{e}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ 成功还原特征向量，维度：{extracted.shape}，dtype：{extracted.dtype}")

    if args.save_vector:
        np.save(args.save_vector, extracted)
        print(f"✓ 特征向量已保存：{args.save_vector}")

    if args.known_face:
        from hack_face.face import get_face_embedding, is_same_person

        known_path = Path(args.known_face)
        if known_path.suffix.lower() == ".npy":
            print(f"→ 从 .npy 文件加载已知特征：{known_path}")
            known_vec = np.load(str(known_path))
        else:
            print(f"→ 正在提取已知人脸特征：{known_path}")
            try:
                known_vec = get_face_embedding(known_path)
            except ValueError as e:
                print(f"✗ 错误：{e}", file=sys.stderr)
                sys.exit(1)

        matched, score = is_same_person(known_vec, extracted, threshold=args.threshold)
        status = "✓ 匹配成功" if matched else "✗ 匹配失败"
        print(f"\n识别结果：{status}")
        print(f"余弦相似度：{score:.4f}  （阈值：{args.threshold}）")
    else:
        print("\n提示：使用 --known-face 参数可与已知人脸进行比对识别")


# ---------------------------------------------------------------------------
# hack-face-blend  (单次融合)
# ---------------------------------------------------------------------------

# 每种模式对应的最优默认参数（来自扫描测试）
_MODE_DEFAULTS: dict[str, dict] = {
    "poisson": dict(face_scale=0.25, hf_scale=1.0, alpha=0.05),
    "adaptive": dict(face_scale=0.25, hf_scale=1.0, alpha=0.05),
    "hf": dict(face_scale=0.25, hf_scale=1.0, alpha=0.01),
    "lum": dict(face_scale=0.25, hf_scale=1.0, alpha=0.01),
    "full": dict(face_scale=0.25, hf_scale=1.0, alpha=0.10),
}


def blend() -> None:
    """将人脸融合进载体图片（单次，自动调参）。"""
    parser = argparse.ArgumentParser(
        prog="hack-face-blend",
        description=(
            "将人脸融合到载体图片中——人眼几乎看不出差异，但 AI 人脸检测器能识别到。\n"
            "提示：使用 hack-face-run 可自动扫描所有参数组合并输出 Top-N 最优结果。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--face", "-f", required=True, metavar="IMAGE", help="人脸照片路径"
    )
    parser.add_argument(
        "--carrier",
        "-c",
        required=True,
        metavar="IMAGE",
        help="载体图片路径（风景/生活照）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="blended_output.png",
        metavar="PNG",
        help="输出路径（默认：blended_output.png）",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["poisson", "adaptive", "hf", "lum", "full"],
        default="poisson",
        help="融合模式（默认 poisson）：\n"
        "  poisson  泊松无缝克隆，隐蔽性最佳\n"
        "  adaptive 纹理自适应高频叠加\n"
        "  hf       高频叠加（边缘/细节）\n"
        "  lum      Lab 亮度融合\n"
        "  full     传统 alpha 混合",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        metavar="INT",
        help="输出正方形边长（默认 640）",
    )
    parser.add_argument(
        "--with-lsb",
        action="store_true",
        default=False,
        help="同时嵌入 LSB 特征向量（可用 hack-face-decode 还原）",
    )
    args = parser.parse_args()

    from hack_face.watermark import blend_face_into_image, encode_face_into_image

    defaults = _MODE_DEFAULTS[args.mode]
    print(f"→ 融合模式：{args.mode}  |  使用最优默认参数")

    try:
        blend_face_into_image(
            face_path=args.face,
            carrier_path=args.carrier,
            output_path=args.output,
            blend_mode=args.mode,
            face_scale=defaults["face_scale"],
            hf_scale=defaults["hf_scale"],
            alpha=defaults["alpha"],
            auto_alpha=True,
            output_size=args.size,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ 错误：{e}", file=sys.stderr)
        sys.exit(1)

    if args.with_lsb:
        from hack_face.face import get_face_embedding

        print("→ 嵌入 LSB 特征向量...")
        try:
            embedding = get_face_embedding(args.face)
            encode_face_into_image(embedding, args.output, args.output)
        except (ValueError, FileNotFoundError) as e:
            print(f"✗ LSB 嵌入错误：{e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# hack-face-run  (自动扫描 + Top-N 报告)
# ---------------------------------------------------------------------------


def run() -> None:
    """自动扫描所有参数组合，输出 Top-N 视觉最隐蔽且 AI 可识别的结果。"""
    parser = argparse.ArgumentParser(
        prog="hack-face-run",
        description=(
            "输入人脸图片 + 载体图片，自动测试所有融合模式与参数组合，\n"
            "按照视觉隐蔽性（SSIM）排序，输出 Top-N 最佳结果及对比图。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--face", "-f", required=True, metavar="IMAGE", help="人脸照片路径"
    )
    parser.add_argument(
        "--carrier",
        "-c",
        required=True,
        metavar="IMAGE",
        help="载体图片路径（风景/生活照）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="DIR",
        help="结果输出目录（默认：载体图片同目录下的 hack_face_out/）",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        metavar="INT",
        help="输出正方形边长（默认 640）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="INT",
        help="保存并展示的最优结果数量（默认 5）",
    )
    args = parser.parse_args()

    from hack_face.sweep import run_sweep

    carrier_path = Path(args.carrier)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (carrier_path.parent / "hack_face_out")
    )

    print(f"┌{'─' * 58}┐")
    print(f"│  hack-face-run  自动参数扫描{' ' * 28}│")
    print(f"├{'─' * 58}┤")
    print(f"│  人脸图片：{args.face:<46}│")
    print(f"│  载体图片：{args.carrier:<46}│")
    print(f"│  输出目录：{str(out_dir):<46}│")
    print(f"│  输出尺寸：{args.size:<3}px   Top-N：{args.top:<27}│")
    print(f"└{'─' * 58}┘")

    try:
        top_results = run_sweep(
            face_path=args.face,
            carrier_path=args.carrier,
            output_dir=out_dir,
            output_size=args.size,
            top_n=args.top,
            verbose=True,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ 错误：{e}", file=sys.stderr)
        sys.exit(1)

    if not top_results:
        print("\n✗ 所有参数组合均未通过 AI 检测，请检查人脸图片质量。", file=sys.stderr)
        sys.exit(1)

    # ── 排名表 ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  Top-{args.top} 最优方案（按 SSIM 降序，SSIM 越高视觉越隐蔽）")
    print(f"{'═' * 80}")
    print(
        f"  {'排名':<4} {'方案':<32} {'SSIM':>7} {'PSNR':>7} {'MAD':>6} {'MaxDiff':>7} {'AI置信度':>8}"
    )
    print(f"  {'─' * 76}")
    for rank, r in enumerate(top_results, 1):
        star = "★" if rank == 1 else f"#{rank}"
        print(
            f"  {star:<4} {r['label']:<32} {r['ssim']:>7.5f} {r['psnr']:>7.2f} "
            f"{r['mad']:>6.3f} {r['max_diff']:>7d} {r['prob']:>8.4f}"
        )

    best = top_results[0]
    print(f"\n  ★ 最优方案：{best['label']}")
    print(
        f"     SSIM={best['ssim']:.5f}  PSNR={best['psnr']:.2f}dB  "
        f"MAD={best['mad']:.3f}  MaxDiff={best['max_diff']}"
    )
    print(f"     文件：{best['output_path']}")
    print(f"\n  对比图：{out_dir / 'comparison_top.png'}")
