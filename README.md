# hack-face

将人脸信息隐写进任意图片，使 AI 人脸识别系统能够检测到指定人脸，而人眼几乎察觉不到任何差异。

## 原理

```
人脸图片 ──► 提取五官边缘特征 ──► 以极低强度融合进载体图片
                                        │
                                        ▼
                              SSIM ≈ 0.9999（人眼无感）
                              AI 人脸检测 ✓（MTCNN 通过）
```

融合采用多种算法（泊松无缝克隆、高频叠加、Lab 亮度融合等），并通过自动参数搜索找到「视觉最隐蔽 + AI 可识别」的最优平衡点。

---

## 安装

**要求：** Python 3.12+

```bash
git clone https://github.com/your-org/hack-face.git
cd hack-face
```

**方式 A：pip（标准）**

```bash
pip install -e .
```

**方式 B：uv（推荐，更快）**

```bash
# 安装 uv（若未安装）
pip install uv

# 创建虚拟环境并安装所有依赖
uv sync

# 激活环境
source .venv/bin/activate
```

> PyTorch 默认安装 CPU 版本。若需 GPU 加速，请先手动安装对应 CUDA 版本的 torch/torchvision，再执行安装步骤。

---

## 快速开始

### 方式一：自动扫描（推荐）

输入人脸 + 载体图片，自动测试 25 种参数组合，输出 Top-5 最优结果及对比图：

```bash
hack-face-run \
  --face  images/source/face.jpg \
  --carrier images/target/landscape.png
```

输出示例：

```
════════════════════════════════════════════════════════════════════
  Top-5 最优方案（按 SSIM 降序，SSIM 越高视觉越隐蔽）
════════════════════════════════════════════════════════════════════
  排名   方案                    SSIM     PSNR    MAD  MaxDiff  AI置信度
  ★    hf_fs0.25_hs1.0      0.99990   59.42  0.044        7    0.4957
  #2   hf_fs0.25_hs1.5      0.99988   58.94  0.046        8    0.4957
  #3   poisson_fs0.25       0.99983   56.98  0.042       12    0.4957
  #4   lum_fs0.25           0.99972   51.09  0.122       14    0.4957
  #5   adaptive_fs0.25_hs1.0 0.99970  53.24  0.065       31    0.4957

  ★ 最优方案：hf_fs0.25_hs1.0
     文件：hack_face_out/hf_fs0.25_hs1.0.png
  对比图：hack_face_out/comparison_top.png
```

**参数说明：**

| 参数 | 说明 | 默认 |
|------|------|------|
| `--face / -f` | 人脸照片路径 | 必填 |
| `--carrier / -c` | 载体图片路径（风景/生活照） | 必填 |
| `--output-dir / -o` | 结果输出目录 | 载体图片同目录下的 `hack_face_out/` |
| `--size` | 输出正方形边长（px） | `640` |
| `--top` | 输出最优结果数量 | `5` |

---

### 方式二：单次融合

已知最优参数时，直接生成结果：

```bash
hack-face-blend \
  --face images/source/face.jpg \
  --carrier images/target/landscape.png \
  --mode hf \
  --output result.png
```

**参数说明：**

| 参数 | 说明 | 默认 |
|------|------|------|
| `--face / -f` | 人脸照片路径 | 必填 |
| `--carrier / -c` | 载体图片路径 | 必填 |
| `--output / -o` | 输出路径 | `blended_output.png` |
| `--mode / -m` | 融合模式（见下表） | `poisson` |
| `--size` | 输出正方形边长（px） | `640` |
| `--with-lsb` | 同时嵌入 LSB 不可见特征向量 | 关闭 |

**融合模式：**

| 模式 | 算法 | 特点 |
|------|------|------|
| `hf` | 高频叠加 | **视觉最隐蔽**（测试 SSIM=0.9999，MaxDiff=7） |
| `poisson` | 泊松无缝克隆 | 边缘无缝，颜色由背景决定 |
| `adaptive` | 纹理自适应高频 | 光滑区域（天空）几乎不变 |
| `lum` | Lab 亮度融合 | 只改明暗，保留背景颜色 |
| `full` | 传统 alpha 混合 | 直观可控 |

---

### LSB 不可见水印（进阶）

将 512 维人脸特征向量以 LSB 隐写形式嵌入图片（每像素仅改变最低 1 bit，肉眼完全无法察觉），支持后续机器验证身份：

```bash
# 嵌入
hack-face-encode \
  --face images/source/face.jpg \
  --carrier result.png \
  --output result_lsb.png

# 提取并验证
hack-face-decode \
  --image result_lsb.png \
  --known-face images/source/face.jpg
```

---

## 评测指标说明

| 指标 | 含义 | 优秀阈值 |
|------|------|---------|
| **SSIM** | 结构相似度（越接近 1.0 越隐蔽） | > 0.999 |
| **PSNR** | 峰值信噪比 dB（越高越隐蔽） | > 50 dB |
| **MAD** | 平均绝对像素差（越低越隐蔽） | < 0.1 |
| **MaxDiff** | 最大单像素差（越低越隐蔽） | < 15 |
| **AI 置信度** | MTCNN 低阈值检测置信度（> 0.3 通过） | > 0.4 |

---

## 项目结构

```
hack-face/
├── src/hack_face/
│   ├── cli.py        # 命令行入口
│   ├── watermark.py  # 融合算法核心（poisson/adaptive/hf/lum/full）
│   ├── face.py       # MTCNN + InceptionResnetV1 人脸检测与特征提取
│   ├── sweep.py      # 自动参数扫描 + 对比图生成
│   └── metrics.py    # SSIM / PSNR / MAD + AI 检测评估
├── images/
│   ├── source/       # 人脸图片
│   └── target/       # 载体图片
└── pyproject.toml
```
