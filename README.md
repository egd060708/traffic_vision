# Traffic Vision — 交通场景视觉感知系统

本仓库实现两个计算机视觉任务：

| 任务 | 内容 | 方法 |
|------|------|------|
| **任务一** | 机动车速度与 TTC（Time-to-Collision）预测 | YOLO11 + ByteTrack + 场景标定 |
| **任务二** | 动态车牌识别（检测 → 跟踪 → 识别） | YOLO-Pose + CRNN（CNN+BiLSTM+CTC） |

---

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [任务一：车速与 TTC 预测](#任务一车速与-ttc-预测)
  - [数据准备](#11-数据准备)
  - [模型（无需训练）](#12-模型无需训练)
  - [推理流程](#13-推理流程)
  - [GUI 使用](#14-gui-使用)
- [任务二：动态车牌识别](#任务二动态车牌识别)
  - [数据下载与处理](#21-数据下载与处理)
  - [模型训练](#22-模型训练)
  - [推理流程](#23-推理流程)
  - [GUI 使用](#24-gui-使用)
- [输出文件说明](#输出文件说明)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

---

## 项目结构

```
traffic_vision/
├── configs/
│   └── calibration.json              # 任务一：场景标定参数
├── docs/                             # 详细技术文档
│   ├── speed_ttc_guide.md            # 任务一完整指南
│   ├── execution_guide.md            # 任务一 GUI 操作流程
│   ├── yolo_speed_ttc_gui_plan.md    # 任务一设计方案
│   ├── license_plate_guide.md        # 任务二完整指南
│   ├── license_plate_gui.md          # 任务二 GUI 实现记录
│   ├── ccpd_dataset_processing.md    # CCPD 数据处理记录
│   └── other_plate_training.md       # 多省均衡模型训练
├── models/                           # 模型权重
│   ├── yolo11n.pt                    # 任务一：预训练检测模型
│   ├── yolo26n.pt                    # 备选模型
│   ├── yolov8n-pose.pt               # 任务二：检测器预训练起点
│   ├── license_plate_det.pt          # 任务二：训练好的车牌检测器
│   ├── plate_recognizer.pt           # 任务二：CCPD 训练的识别器（偏皖）
│   └── plate_recognizer_other.pt     # 任务二：多省均衡识别器
├── src/                              # 源代码
│   ├── yolo_speed_ttc_gui.py         # 任务一 GUI 主程序
│   ├── speed_ttc_gui.py              # 任务一 简化版（MOG2，无需 GPU）
│   ├── license_plate_gui.py          # 任务二 GUI 主程序
│   ├── plate_pipeline.py             # 任务二 核心库（CRNN/检测/裁剪/绘制）
│   ├── prepare_ccpd.py               # CCPD 数据预处理
│   ├── prepare_other_plate.py        # other 数据集预处理
│   ├── train_plate_detector.py       # 车牌检测器训练脚本
│   └── train_plate_recognizer.py     # 车牌识别器训练脚本
├── videos/
│   ├── predict_line/                 # 任务一测试视频
│   │   ├── car/                      # 汽车场景
│   │   ├── bike/                     # 自行车/电动车场景
│   │   └── mix/                      # 混合场景
│   └── check_number/                 # 任务二测试视频
├── outputs/                          # 输出目录
│   ├── videos/                       # 标注视频
│   └── csv/                          # 逐帧检测/识别结果
├── data/processed/                   # 处理后数据（gitignored）
├── datasets/                         # 原始数据集（gitignored）
├── run_yolo_gui.bat                  # 任务一 GUI 启动脚本
├── run_plate_gui.bat                 # 任务二 GUI 启动脚本
├── environment.yml                   # Conda 环境定义
└── plan.md                           # 任务二原始方案设计
```

---

## 环境配置

### 方式一：使用已有的 Conda 环境

项目已配置好 `traffic-yolo` 环境（Python 3.10 + PyTorch CUDA 12.6），启动脚本直接调用：

```
D:\conda_envs\traffic-yolo\python.exe
```

验证环境：

```bash
D:\conda_envs\traffic-yolo\python.exe -c "import cv2, torch; from ultralytics import YOLO; print('ok')"
```

### 方式二：从 environment.yml 重建

```bash
conda env create -f environment.yml
conda activate traffic-yolo
```

关键依赖：

| 包 | 版本 | 用途 |
|----|------|------|
| `ultralytics` | ≥8.3.0 | YOLO 检测/跟踪框架 |
| `opencv-python` | ≥4.8.0 | 视频读写、图像处理 |
| `torch` | 2.12.1+cu126 | 深度学习后端 |
| `numpy` | ≥1.24.0 | 数值计算 |
| `pandas` | ≥2.0.0 | CSV 输出 |
| `Pillow` | ≥10.0.0 | GUI 图像显示 |
| `lap` | ≥0.5.12 | ByteTrack 线性指派 |

### 方式三：手动安装（CPU 版本）

```bash
conda create -n traffic-yolo python=3.10 -y
conda activate traffic-yolo
pip install torch torchvision torchaudio
pip install ultralytics opencv-python pillow numpy pandas lap
```

### 跨平台快速启动

Windows 和 Linux/macOS 的环境、字体、Tkinter、RAR 解压依赖说明见 `docs/cross_platform.md`。

通用 CPU/default PyPI 安装：

```bash
python -m pip install -r requirements.txt
```

Windows CUDA 12.6 安装：

```bash
python -m pip install -r requirements-cuda126.txt
```

启动命令：

```bash
# Windows
run_yolo_gui.bat
run_plate_gui.bat

# Linux/macOS
sh run_yolo_gui.sh
sh run_plate_gui.sh

# 所有平台
python src/yolo_speed_ttc_gui.py
python src/license_plate_gui.py
```

---

## 任务一：车速与 TTC 预测

### 概述

利用 YOLO11 目标检测 + ByteTrack 多目标跟踪 + 场景一维标定，从 dashcam 视频中实时估计车辆的行驶速度（km/h）和到达人行横道目标线的剩余时间（TTC）。

### 1.1 数据准备

**无需训练数据**。YOLO11 使用 Ultralytics 官方 COCO 预训练权重直接推理。

测试视频位于 `videos/predict_line/`：

```
videos/predict_line/
├── car/     (4 个 mp4)    # 汽车场景
├── bike/    (3 个 mp4)    # 自行车/电动自行车场景
├── mix/     (3 个 mp4)    # 混合交通场景
├── cali.jpg               # 场景标定参考图
├── cali_arrow.jpg         # 带标定标注的参考图
└── calibration_reference.md
```

### 1.2 模型（无需训练）

| 模型 | 大小 | 来源 | 用途 |
|------|------|------|------|
| `models/yolo11n.pt` | 5.6 MB | [Ultralytics 官方](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt) | 目标检测 + ByteTrack 跟踪 |

YOLO 自动检测 COCO 类别并映射到项目标签：

| COCO 类 | 项目标签 | 条件 |
|---------|---------|------|
| car | 汽车 | 默认 |
| bicycle | 自行车 | 视频路径含 "bike" |
| motorcycle | 电动自行车 | 视频路径含 "e-bike" |
| bus | 巴士 | 默认 |
| truck | 卡车 | 默认 |

### 1.3 推理流程

```
读取视频帧
  → YOLO11 检测 (car/bicycle/motorcycle/bus/truck) + ByteTrack 分配 track_id
  → 取检测框底边中心点作为车辆接地点
  → ROI 多边形过滤（仅保留道路区域内目标）
  → 根据 calibration.json 将像素位置映射为真实距离（米）
  → 维护每个 track 的距离-时间历史（最近 12 帧）
  → 线性回归拟合距离曲线 → 计算速度 (km/h)
  → TTC = 当前距离 / 接近速度
  → 绘制：检测框、track ID、类别、速度、TTC、目标线、标定轴
  → 可选输出：标注视频 + CSV
```

#### 场景标定（calibration.json）

标定文件 `configs/calibration.json` 定义了一维道路坐标轴：

| 参数 | 值 | 说明 |
|------|-----|------|
| `reference_size` | `[1200, 683]` | 标定参考图尺寸 |
| `target_point` | `[92, 516]` | 目标线（人行横道）参考点 |
| `far_point` | `[1046, 434]` | 蓝色标定箭头远端 |
| `axis_length_m` | `12.10` | 两点间真实距离（米） |
| `target_line` | `[[70,540],[210,410]]` | 目标线像素位置 |
| `roi_polygon` | 4 点坐标 | 道路感兴趣区域 |
| `speed_window` | `12` | 速度拟合历史帧数 |
| `min_approach_speed_mps` | `0.15` | TTC 最小接近速度阈值 |

#### 速度与 TTC 计算

```
对每个 track_id 维护: deque[(timestamp, distance_m), ...], maxlen=12
线性回归: distance = a * time + b
speed_mps = |a|
speed_kmh = speed_mps * 3.6
TTC = distance_m / (-a)                     (若 a < 0 且 speed > 0.15 m/s)
TTC = "--"                                  (接近速度过小或历史不足)
TTC = "crossed"                             (已越过目标线)
```

### 1.4 GUI 使用

#### 启动

```bash
# 方式一：双击
run_yolo_gui.bat

# 方式二：终端
D:\conda_envs\traffic-yolo\python.exe src\yolo_speed_ttc_gui.py
```

#### 界面布局

```
┌──────────────────────────────────────────────────────────┐
│ [Open Video]  [Play/Pause]  [Reset]                      │
│ Class: [Auto ▼]  conf: [0.25 ▼]  imgsz: [640 ▼]         │
│ device: [auto ▼]  ☐ Save video/csv                       │
│ ☐ Dynamic line  ☐ Lock line                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                    视频显示区域                            │
│            检测框 + 类别 + ID + 速度 + TTC                 │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ video.mp4 | frame=342 | tracking=3 | infer=45ms           │
└──────────────────────────────────────────────────────────┘
```

#### 控件说明

| 控件 | 默认 | 说明 |
|------|------|------|
| **Open Video** | — | 选择 `videos/predict_line/` 下的测试视频 |
| **Play/Pause** | Play | 播放/暂停 |
| **Reset** | — | 回到视频开头 |
| **Class** | Auto | 类别覆盖：Auto/car/bicycle/e-bike/vehicle |
| **conf** | 0.25 | 置信度阈值（0.15~0.50） |
| **imgsz** | 640 | 推理尺寸（480/640/800） |
| **device** | auto | 推理设备：auto/cpu/0 |
| **Save video/csv** | off | 勾选后输出到 `outputs/` |
| **Dynamic line** | off | 自动检测目标线（白线检测 + HoughLinesP） |
| **Lock line** | off | 锁定检测到的目标线 |

#### 操作步骤

1. 双击 `run_yolo_gui.bat` 启动 GUI
2. 点击 **Open Video** → 选择 `videos/predict_line/car/`、`bike/` 或 `mix/` 下的视频
3. 视频自动播放，实时显示检测框、速度、TTC
4. 如需保存结果 → 先勾选 **Save video/csv**，再打开视频
5. 参数调优：
   - 漏检多 → 降低 conf（0.15~0.20）
   - 误检多 → 调高 conf（0.30~0.40）
   - 卡顿 → 降低 imgsz（480）
   - 分类错误 → 手动选择 Class 覆盖

#### 标注颜色

| 颜色 | 目标类型 |
|------|---------|
| 🟢 绿色 | 汽车 (car) |
| 🔵 青色 | 自行车 (bicycle) |
| 🟠 橙色 | 电动自行车 (e-bike / motorcycle) |

#### TTC 显示规则

| 显示 | 条件 |
|------|------|
| `X.Xs` | 正常接近目标线 |
| `crossed` | 已越过目标线 (distance ≤ 0) |
| `--` | 接近速度 < 0.15 m/s 或历史帧数 < 5 |

#### 备选方案

`src/speed_ttc_gui.py` — 基于 **MOG2 背景减除**的简化版，不依赖 YOLO/GPU，当 YOLO 版本无法运行时备用。

---

## 任务二：动态车牌识别

### 概述

完整实现视频中车牌的 **检测 → 跟踪 → 裁剪 → 字符识别** 全流程：
- **检测**：YOLO-Pose 同时输出车牌边界框和 4 个角点
- **校正**：4 角点透视变换（优于仿射变换）
- **识别**：CRNN（4 层 CNN + 2 层 BiLSTM + CTC Loss）
- **稳定**：同一车牌多帧识别结果投票融合

### 2.1 数据下载与处理

#### 数据来源

| 数据集 | 存放位置 | 数量 | 说明 |
|--------|---------|------|------|
| CCPD2019 (ccpd_base) | `datasets/CCPD2019/ccpd_base/` | 13,940 张 | 蓝牌（7 位），皖为主 |
| CCPD2020 (ccpd_green) | `datasets/CCPD2020/CCPD2020/ccpd_green/` | 11,776 张 | 绿牌（8 位新能源），皖为主 |
| other/git_plate | `datasets/other/git_plate/train.zip` | 63,196 张 | 多省均衡，含特殊车牌 |

> **注意**：原始数据集需自行下载并放入 `datasets/` 目录。该目录已在 `.gitignore` 中排除。

#### CCPD 数据预处理

CCPD 图像为**整车照片**，标注信息编码在文件名中。预处理脚本将文件名解析为 YOLO-Pose 标签和裁剪车牌。

```bash
cd D:\研一课内\模式识别与机器视觉\交通任务\traffic_vision

D:\conda_envs\traffic-yolo\python.exe src\prepare_ccpd.py \
  --ccpd-root datasets\CCPD2019\ccpd_base datasets\CCPD2020\CCPD2020\ccpd_green \
  --out data\processed\ccpd_plate \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --seed 42 --clean
```

**输出结构**（`data/processed/ccpd_plate/`）：

```
ccpd_plate/
├── yolo_pose/                        # 检测模型训练数据
│   ├── data.yaml                     # YOLO 数据集配置
│   ├── images/{train,val,test}/      # 原始整车图像 (25,716 张)
│   └── labels/{train,val,test}/      # YOLO-Pose 标注
├── recognition/                      # 识别模型训练数据
│   ├── alphabet.txt                  # 字母表（68 tokens）
│   ├── {train,val,test}.txt          # 清单文件（路径\t车牌文本）
│   └── {train,val,test}/             # 透视裁剪后的车牌图像
└── splits/                           # 数据划分记录
```

数据划分（seed=42）：

| 集合 | 数量 |
|------|------|
| train | 20,572 |
| val | 2,571 |
| test | 2,573 |
| **合计** | **25,716** |

#### other/git_plate 数据预处理

"other" 数据集已是裁剪好的车牌图（约 178×66 px），文件名直接包含车牌号。省份分布更均衡（粤 14.9%、川 11.6%、苏 10.7%、皖仅 6.4%）。

```bash
D:\conda_envs\traffic-yolo\python.exe src\prepare_other_plate.py --clean
```

**输出结构**（`data/processed/other_plate/recognition/`）：

```
recognition/
├── alphabet.txt     # 字母表（76 tokens，含使/挂/民/港/澳/航/领）
├── train.txt        # 训练清单（56,876 行）
├── val.txt          # 验证清单（6,320 行）
└── train/           # 裁剪好的车牌图像
```

### 2.2 模型训练

#### 架构总览

```
┌─────────────────────────┐     ┌──────────────────────────┐
│   检测器 (YOLO-Pose)      │     │   识别器 (PlateCRNN)       │
├─────────────────────────┤     ├──────────────────────────┤
│ 基础: yolov8n-pose.pt    │     │ 输入: 160×48 RGB          │
│ 输入: 640×640 整车图      │     │ CNN: 4层 Conv+BN+ReLU+Pool │
│ 输出: bbox + 4 个角点     │     │ RNN: 2层 BiLSTM(256→128)  │
│ 跟踪: ByteTrack          │     │ Head: Linear(256, N_tokens)│
│                          │     │ 损失: CTC Loss             │
└──────────────────────────┘     └───────────────────────────┘
         │                              │
         └──────────┬───────────────────┘
                    ▼
         ┌─────────────────────┐
         │  license_plate_gui   │
         │  视频 → 检测 → 裁剪 → │
         │  识别 → 投票 → 输出   │
         └─────────────────────┘
```

#### 训练车牌检测器

```bash
D:\conda_envs\traffic-yolo\python.exe src\train_plate_detector.py \
  --data data\processed\ccpd_plate\yolo_pose\data.yaml \
  --model yolov8n-pose.pt \
  --epochs 100 --imgsz 640 --batch 16 \
  --save-to models\license_plate_det.pt
```

| 参数 | 默认/推荐 | 说明 |
|------|----------|------|
| `--data` | `data/processed/ccpd_plate/yolo_pose/data.yaml` | 数据集配置 |
| `--model` | `yolov8n-pose.pt` | 预训练权重 |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 640 | 输入尺寸 |
| `--batch` | 16 | 批次大小 |
| `--device` | auto | 训练设备（GPU 用 `0` 或 `cuda:0`） |
| `--workers` | 0 | Windows 建议设为 0 |
| `--save-to` | `models/license_plate_det.pt` | 输出路径 |

#### 训练车牌识别器

##### CCPD 版本（68 tokens，偏向皖）

```bash
D:\conda_envs\traffic-yolo\python.exe src\train_plate_recognizer.py \
  --train data\processed\ccpd_plate\recognition\train.txt \
  --val data\processed\ccpd_plate\recognition\val.txt \
  --out models\plate_recognizer.pt \
  --epochs 30 --batch 64 --lr 1e-3 --device cuda:0
```

##### other 版本（76 tokens，多省均衡，推荐）

```bash
D:\conda_envs\traffic-yolo\python.exe src\train_plate_recognizer.py \
  --train data\processed\other_plate\recognition\train.txt \
  --val data\processed\other_plate\recognition\val.txt \
  --out models\plate_recognizer_other.pt \
  --epochs 30 --batch 64 --lr 1e-3 --device cuda:0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train` | — | 训练清单路径 |
| `--val` | — | 验证清单路径 |
| `--epochs` | 30 | 训练轮数 |
| `--batch` | 64 | 批次大小 |
| `--lr` | 1e-3 | 学习率（AdamW） |
| `--device` | cuda:0 | 训练设备（CPU 用 `cpu`） |
| `--out` | `models/plate_recognizer.pt` | 输出路径 |

训练监控指标：
- **val_char_acc**：字符级准确率
- **val_plate_acc**：整牌完全匹配率
- 每次 `val_plate_acc` 创新高时自动保存检查点

#### 模型对比

| 特性 | `plate_recognizer.pt` | `plate_recognizer_other.pt` |
|------|----------------------|---------------------------|
| 训练数据 | CCPD（25,716 张） | other/git_plate（56,876 张） |
| 皖占比 | ~50%+ | 6.4% |
| 字母表 | 68 tokens | 76 tokens |
| 特殊车牌 | 不支持 | 支持（使馆/挂车/港澳等） |
| 省份偏差 | 严重偏向皖 | 多省均衡 |
| **GUI 默认使用** | 否 | **是** |

#### PlateCRNN 详细架构

```
输入: (B, 3, 48, 160)
  │
  ├─ Conv(3→32, k3, p1)  → BN → ReLU → MaxPool(2,2)   → (B, 32, 24, 80)
  ├─ Conv(32→64, k3, p1) → BN → ReLU → MaxPool(2,2)   → (B, 64, 12, 40)
  ├─ Conv(64→128, k3, p1) → BN → ReLU → MaxPool(2,1)  → (B, 128, 6, 40)
  ├─ Conv(128→256, k3, p1) → BN → ReLU → MaxPool(2,1) → (B, 256, 3, 40)
  │
  ├─ AdaptiveAvgPool2d(1, None) → squeeze → permute   → (B, 40, 256)
  ├─ BiLSTM(256→128, 2 layers, bidirectional)         → (B, 40, 256)
  └─ Linear(256, N_tokens)                             → (B, 40, N_tokens)
```

设计要点：后两层池化核 `(2,1)` 只压缩高度不压缩宽度，保留约 40 个时间步的序列特征。

#### 字母表

```
PROVINCES（33个）:  皖 沪 津 渝 冀 晋 蒙 辽 吉 黑 苏 浙 京 闽 赣 鲁
                    豫 鄂 湘 粤 桂 琼 川 贵 云 藏 陕 甘 青 宁 新 警 学
SPECIAL（7个）:      使 挂 民 港 澳 航 领        (仅 other 版本)
LETTERS（25个）:     A B C D E F G H J K L M N P Q R S T U V W X Y Z
DIGITS（10个）:      0 1 2 3 4 5 6 7 8 9
BLANK（1个）:        <blank> → CTC 空白标记
```

### 2.3 推理流程

```
视频帧 (BGR, 1920×1080)
  │
  ├─ 1. YOLO-Pose 检测 + ByteTrack 分配 track_id
  │      detector.track(conf=0.25, imgsz=640, persist=True)
  │      输出: bbox + 4 角点 + track_id + confidence
  │
  ├─ 2. 车牌裁剪 (crop_plate)
  │      ├─ 有 4 角点 → 透视变换 (order_corners → getPerspectiveTransform)
  │      └─ 无角点   → 矩形裁剪（bbox 扩展 8% 水平 + 18% 垂直）
  │      输出: 矫正车牌图像
  │
  ├─ 3. CRNN 识别 (每 interval=2 帧一次)
  │      输入: 160×48 RGB, [0,1] 归一化
  │      CNN → BiLSTM → Linear → softmax → argmax
  │      CTC 贪婪解码 → 文本 + 置信度
  │
  ├─ 4. 多帧投票 (vote_window=15)
  │      a) 找出最常见的文本长度
  │      b) 筛选该长度的候选项
  │      c) 逐字符取众数
  │      例: ["皖AD62208","皖AD82208","皖AD62208"] → "皖AD62208"
  │
  ├─ 5. 绘制标注
  │      黄色矩形框 + 青色角点多边形
  │      框上方: #trackID raw_text (confidence)
  │      框右侧: 绿色大字稳定车牌号
  │
  └─ 6. 可选输出: 标注视频 + CSV
```

### 2.4 GUI 使用

#### 启动

```bash
# 方式一：双击
run_plate_gui.bat

# 方式二：终端
D:\conda_envs\traffic-yolo\python.exe src\license_plate_gui.py
```

启动后终端会显示模型加载信息：

```
[GUI] main() starting...
[GUI] load_models() starting...
[GUI] detector: ...\license_plate_det.pt  exists=True
[GUI] recognizer: ...\plate_recognizer_other.pt  exists=True
[GUI] pipeline created in 0.8s
[GUI] Models ready! recognizer loaded
```

#### 界面布局

```
┌──────────────────────────────────────────────────────────┐
│ [Open Video]  [Play/Pause]  [Reset]                      │
│ conf: [0.25 ▼]  imgsz: [640 ▼]  device: [auto ▼]        │
│ interval: [2 ▼]  vote: [15 ▼]  ☐ Save video/csv          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                    视频显示区域                            │
│         黄色检测框 + 青色角点 + 车牌号叠加                  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ video.mp4 | frame=123 | plates=2 | infer+draw=45ms        │
└──────────────────────────────────────────────────────────┘
```

#### 控件说明

| 控件 | 默认 | 说明 |
|------|------|------|
| **Open Video** | — | 选择 `videos/check_number/` 下的视频 |
| **Play/Pause** | Play | 播放/暂停 |
| **Reset** | — | 回到视频开头 |
| **conf** | 0.25 | 检测置信度（0.15~0.50） |
| **imgsz** | 640 | 检测推理尺寸（480/640/800） |
| **device** | auto | 推理设备：auto/cpu/0 |
| **interval** | 2 | 识别间隔（每隔 N 帧识别一次，提高帧率） |
| **vote** | 15 | 投票窗口大小（越大越稳定，响应越慢） |
| **Save video/csv** | off | 勾选后输出到 `outputs/` |

#### 操作步骤

1. 双击 `run_plate_gui.bat` 启动 GUI
2. 点击 **Open Video** → 选择 `videos/check_number/` 下的视频
3. 观察识别效果：
   - **黄色框**：检测到的车牌位置
   - **青色点**：车牌四角
   - **框上方**：`#trackID raw_text confidence`
   - **框右侧绿色大字**：投票稳定后的车牌号
4. 参数调优：
   - 误报多 → 调高 conf
   - 漏检多 → 降低 conf（0.15~0.20）
   - 帧率低 → 增大 interval（3 或 5）
   - 结果不稳定 → 增大 vote（20 或 30）
   - 响应慢 → 减小 vote（5 或 10）
5. 如需保存结果 → 勾选 **Save video/csv**

#### 切换识别模型

修改 `src/license_plate_gui.py` 第 32 行附近：

```python
# CCPD 训练的模型（68 tokens，偏向皖）
# RECOGNIZER_PATH = MODEL_DIR / "plate_recognizer.pt"

# other 训练的模型（76 tokens，多省均衡，GUI 默认）
RECOGNIZER_PATH = MODEL_DIR / "plate_recognizer_other.pt"
```

修改后重新启动 GUI 即可生效。

---

## 输出文件说明

两个任务的 GUI 勾选 **Save video/csv** 后均会输出：

### 标注视频

```
outputs/videos/{视频名}_annotated.mp4         # 任务一
outputs/videos/{视频名}_plate_annotated.mp4    # 任务二
```

### CSV 文件

```
outputs/csv/{视频名}_yolo_speed_ttc.csv        # 任务一
outputs/csv/{视频名}_plate_recognition.csv      # 任务二
```

#### 任务一 CSV 字段

| 字段 | 说明 |
|------|------|
| `video` | 视频文件名 |
| `frame` | 帧序号 |
| `time_s` | 帧时刻（秒） |
| `track_id` | 跟踪 ID |
| `label` | 项目标签（汽车/自行车等） |
| `yolo_class` | YOLO 原始类别号 |
| `x1, y1, x2, y2` | 检测框坐标 |
| `distance_m` | 距目标线距离（m） |
| `speed_kmh` | 速度（km/h） |
| `ttc_s` | TTC（秒） |

#### 任务二 CSV 字段

| 字段 | 说明 |
|------|------|
| `video` | 视频文件名 |
| `frame` | 帧序号 |
| `time_s` | 帧时刻（秒） |
| `track_id` | 跟踪 ID |
| `x1, y1, x2, y2` | 检测框坐标 |
| `plate_conf` | 检测置信度 |
| `raw_text` | 当前帧原始识别结果 |
| `raw_text_conf` | 当前帧识别置信度 |
| `stable_text` | 投票后稳定车牌号 |

---

## 常见问题

### 任务一

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 无检测框 | conf 太高 | 降低 conf 到 0.15~0.20 |
| 帧率过低 | imgsz 太大 | 降低 imgsz 到 480 |
| 分类错误 | 车型混淆 | 手动选择 Class 覆盖 |
| 标定不准 | 视频视角变化 | 重新标定并更新 `calibration.json` |
| 提示缺少模型 | `models/yolo11n.pt` 不存在 | 从 [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt) 下载 |
| GUI 打不开 | 依赖缺失 | 运行验证命令检查依赖 |

### 任务二

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 提示 Missing model | 模型文件缺失 | 检查 `models/` 下是否有 `.pt` 文件 |
| 显示 NO_RECOGNIZER | 识别模型加载失败 | 检查 PyTorch 版本和模型完整性 |
| 所有结果都是"皖" | 使用了 CCPD 模型 | 切换到 `plate_recognizer_other.pt` |
| OpenCV 导入失败 | 缺少 opencv | `pip install opencv-python` |
| 中文显示乱码 | 缺少中文字体 | 确保 `C:/Windows/Fonts/msyh.ttc` 存在 |
| 帧率过低 | GPU 不可用或 imgsz 太大 | 设 device=cpu 或降低 imgsz 到 480 |

### 环境验证命令

```bash
# 检查依赖完整性
D:\conda_envs\traffic-yolo\python.exe -c "import cv2, torch; from ultralytics import YOLO; print('ok')"

# 检查模型文件
D:\conda_envs\traffic-yolo\python.exe -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); [print(f'{f.name}: {f.exists()}') for f in [Path('models/yolo11n.pt'), Path('models/license_plate_det.pt'), Path('models/plate_recognizer_other.pt')]]"
```

---

## 参考资料

- Ultralytics YOLO 文档：<https://docs.ultralytics.com/>
- PyTorch 安装：<https://pytorch.org/get-started/locally/>
- CCPD 数据集论文：<https://github.com/detectRecog/CCPD>
- 详细技术文档见 `docs/` 目录下各 `.md` 文件
- 原始设计方案见 `plan.md`
