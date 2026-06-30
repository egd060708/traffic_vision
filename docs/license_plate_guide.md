# 任务二：动态车牌识别

## 概述

本任务实现视频中车牌的**检测 → 跟踪 → 裁剪 → 字符识别**全流程。检测模型使用 YOLO-Pose（定位车牌四角），识别模型使用 CRNN（卷积循环神经网络）做序列识别。支持中文车牌（含新能源绿牌、使馆等特殊车牌），对同一车辆的多帧识别结果进行投票融合，输出稳定车牌号。

## 目录

1. [数据处理](#1-数据处理)
2. [模型训练](#2-模型训练)
3. [推理验证](#3-推理验证)
4. [GUI 使用](#4-gui-使用)

---

## 1. 数据处理

### 1.1 数据来源

项目使用两个来源的车牌数据：

| 数据集 | 路径 | 数量 | 说明 |
|--------|------|------|------|
| CCPD2019 | `datasets/CCPD2019/` | 13,940 张 | 蓝牌（7 位），皖为主 |
| CCPD2020 | `datasets/CCPD2020/` | 11,776 张 | 绿牌（8 位新能源），皖为主 |
| other/git_plate | `datasets/other/git_plate/train.zip` | 63,196 张 | 多省均衡，含特殊车牌 |

### 1.2 CCPD 数据格式

CCPD 图像为**整车照片**，文件名编码了标注信息：

```
{area}-{tilt}-{bbox}-{corners}-{plate}-{brightness}-{blur}.jpg

示例: 036041...-90_89-351&564_451&606-463&608_353&618_353&561_463&560-0_0_3_30_26_26_24_32-100-13.jpg
```

各部分含义：

| 字段 | 格式 | 说明 |
|------|------|------|
| area | 数字 | 拍摄区域 |
| tilt | `水平_垂直` | 倾斜角度 |
| bbox | `x1&y1_x2&y2` | 车牌边界框 |
| corners | `x1&y1_x2&y2_x3&y3_x4&y4` | 车牌四角坐标 |
| **plate** | `p_l_c1_c2_c3_c4_c5_c6` | **车牌号索引**（7 或 8 个整数） |
| brightness | 数字 | 亮度 |
| blur | 数字 | 模糊程度 |

**车牌编码规则**：
- `indexes[0]` → PROVINCES 列表中的省份（例：`0` = 皖）
- `indexes[1]` → LETTERS 列表中的字母（例：`0` = A）
- `indexes[2:]` → ADS 列表中的字母数字（例：`3,30,26,26,24,32` = D62208）
- 结果：`0_0_3_30_26_26_24_32` → **皖AD62208**

### 1.3 other/git_plate 数据格式

图像为**已裁剪的车牌图**（约 178×66 像素），文件名直接包含完整车牌号：

| 格式 | 示例 | 提取的车牌 |
|------|------|-----------|
| `{idx}_{plate}_{suffix}.jpg` | `2700_皖AD11558_417411.jpg` | `皖AD11558` |
| `{plate}_{variant}.jpg` | `171001使_stretch2.jpg` | `171001使` |

### 1.4 数据处理脚本

#### CCPD 预处理

```bash
cd D:\研一课内\模式识别与机器视觉\交通任务\traffic_vision

# 处理 CCPD2019 + CCPD2020
D:\conda_envs\traffic-yolo\python.exe src\prepare_ccpd.py ^
  --ccpd-root datasets\CCPD2019 datasets\CCPD2020 ^
  --out data\processed\ccpd_plate ^
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 ^
  --seed 42 ^
  --clean
```

**输出结构**（`data/processed/ccpd_plate/`）：

```
ccpd_plate/
├── yolo_pose/                     # 检测模型训练数据
│   ├── data.yaml                  # YOLO 数据集配置
│   ├── images/{train,val,test}/   # 原始整车图像
│   └── labels/{train,val,test}/   # YOLO-Pose 标注文件
│                                  # 格式: 0 cx cy w h kp1_x kp1_y 2 kp2_x kp2_y 2 ...
│                                  #       kpt_shape: [4, 3]（4个角点，可见性=2）
├── recognition/                   # 识别模型训练数据
│   ├── alphabet.txt               # 字母表（76行，最后一行 <blank>）
│   ├── {train,val,test}.txt       # 清单文件(绝对值路径\t车牌文本)
│   └── {train,val,test}/          # 裁剪后的车牌图像（透视校正）
└── splits/                        # 原始图像分片记录
    └── {train,val,test}.txt
```

#### other/git_plate 预处理

```bash
D:\conda_envs\traffic-yolo\python.exe src\prepare_other_plate.py ^
  --train-zip datasets\other\git_plate\train.zip ^
  --val-rar datasets\other\git_plate\val.rar ^
  --out data\processed\other_plate\recognition ^
  --train-ratio 0.9 ^
  --seed 42 ^
  --clean
```

**输出结构**（`data/processed/other_plate/recognition/`）：

```
recognition/
├── alphabet.txt      # 字母表
├── train.txt         # 训练清单（56,876 行）
├── val.txt           # 验证清单（6,320 行）
└── train/            # 裁剪好的车牌图像（56,876 张）
```

### 1.5 清单文件格式

所有识别清单严格遵循：**每行 `image_path\tplate_text`**（制表符分隔）：

```
.../recognition/train/川BA4543_7439_a8d426381e.jpg	川BA4543
.../recognition/train/京AA09706_26_e01e45dbed.jpg	京AA09706
```

### 1.6 字母表

```
75 个可见字符 + 1 个空白标记 = 76 个 token

PROVINCES（33个）:  皖 沪 津 渝 冀 晋 蒙 辽 吉 黑 苏 浙 京 闽 赣 鲁
                    豫 鄂 湘 粤 桂 琼 川 贵 云 藏 陕 甘 青 宁 新 警 学

SPECIAL（7个）:      使 挂 民 港 澳 航 领

LETTERS（25个）:     A B C D E F G H J K L M N P Q R S T U V W X Y Z
                    （不含 I 和 O，避免与 1/0 混淆）

DIGITS（10个）:      0 1 2 3 4 5 6 7 8 9

BLANK（1个）:        <blank>  →  CTC Loss 的空白标记（索引 75）
```

### 1.7 省份分布对比

| 数据集 | 皖占比 | 特点 |
|--------|--------|------|
| CCPD（原） | ~50%+ | 严重偏向安徽 |
| other（新） | 6.4% | 粤14.9%/川11.6%/苏10.7%，更均衡 |

---

## 2. 模型训练

### 2.1 架构总览

```
┌──────────────────────┐     ┌──────────────────────┐
│   检测模型 (Detector)  │     │   识别模型 (Recognizer) │
├──────────────────────┤     ├──────────────────────┤
│ 基础模型: YOLO-Pose   │     │ 模型: PlateCRNN       │
│ 预训练: yolov8n-pose  │     │ 输入: 160×48 RGB     │
│ 输入: 640×640 整车图   │     │ CNN: 4层卷积+池化     │
│ 输出: bbox + 4个角点   │     │ RNN: 2层BiLSTM       │
│ 跟踪: ByteTrack       │     │ 分类: Linear(256, 76)│
│                      │     │ 损失: CTC Loss        │
└──────────────────────┘     └──────────────────────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
         ┌──────────────────────┐
         │  license_plate_gui.py │
         │  视频 → 检测 → 裁剪 → │
         │  识别 → 投票 → 输出   │
         └──────────────────────┘
```

### 2.2 检测模型训练

```bash
cd D:\研一课内\模式识别与机器视觉\交通任务\traffic_vision

D:\conda_envs\traffic-yolo\python.exe src\train_plate_detector.py ^
  --data data\processed\ccpd_plate\yolo_pose\data.yaml ^
  --model yolov8n-pose.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --save-to models\license_plate_det.pt
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/processed/ccpd_plate/yolo_pose/data.yaml` | YOLO-Pose 数据集配置 |
| `--model` | `yolov8n-pose.pt` | 预训练权重（Ultralytics） |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 640 | 输入尺寸 |
| `--batch` | 16 | 批次大小 |
| `--device` | auto | 训练设备 |
| `--workers` | 0 | DataLoader 工作进程数（Windows 设为 0） |
| `--save-to` | `models/license_plate_det.pt` | 训练完成后输出路径 |

**训练数据格式**（`data.yaml`）：

```yaml
path: data/processed/ccpd_plate/yolo_pose
train: images/train
val: images/val
test: images/test
names:
  0: license_plate
kpt_shape: [4, 3]
```

**标注格式**（每行）：

```
0 cx cy bw bh kp1_x kp1_y 2 kp2_x kp2_y 2 kp3_x kp3_y 2 kp4_x kp4_y 2
```

- 类别：`0`（license_plate）
- 坐标：归一化到 [0, 1]
- 关键点可见性：`2`（始终可见）

### 2.3 识别模型训练

#### 模型架构详情（PlateCRNN）

```
输入: (B, 3, 48, 160)
  │
  ├─ Conv(3→32, k3, p1) → BN → ReLU → MaxPool(2,2)    → (B, 32, 24, 80)
  ├─ Conv(32→64, k3, p1) → BN → ReLU → MaxPool(2,2)    → (B, 64, 12, 40)
  ├─ Conv(64→128, k3, p1) → BN → ReLU → MaxPool(2,1)   → (B, 128, 6, 40)
  ├─ Conv(128→256, k3, p1) → BN → ReLU → MaxPool(2,1)  → (B, 256, 3, 40)
  │
  ├─ AdaptiveAvgPool2d(1, None) → squeeze → permute    → (B, 40, 256)
  │
  ├─ BiLSTM(256→128, 2 layers, bidirectional)          → (B, 40, 256)
  │
  └─ Linear(256, 76)                                    → (B, 40, 76)
```

**设计要点**：
- 后两层池化核 `(2,1)` 只压缩高度不压缩宽度，保留序列长度（~40 个时间步）
- 自适应平均池化将高度统一为 1，得到序列特征
- 2 层双向 LSTM 建模字符间上下文依赖
- CTC Loss 自动学习字符对齐，无需字符级位置标注

#### CCPD 训练（旧模型，偏向皖）

```bash
D:\conda_envs\traffic-yolo\python.exe src\train_plate_recognizer.py ^
  --train data\processed\ccpd_plate\recognition\train.txt ^
  --val data\processed\ccpd_plate\recognition\val.txt ^
  --out models\plate_recognizer.pt ^
  --epochs 30 --batch 64 --lr 1e-3 --device cuda:0
```

#### other 数据集训练（新模型，多省均衡）

```bash
D:\conda_envs\traffic-yolo\python.exe src\train_plate_recognizer.py ^
  --train data\processed\other_plate\recognition\train.txt ^
  --val data\processed\other_plate\recognition\val.txt ^
  --out models\plate_recognizer_other.pt ^
  --epochs 30 --batch 64 --lr 1e-3 --device cuda:0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train` | — | 训练清单路径 |
| `--val` | — | 验证清单路径 |
| `--epochs` | 30 | 训练轮数 |
| `--batch` | 64 | 批次大小 |
| `--lr` | 1e-3 | 学习率（AdamW，weight_decay=1e-4） |
| `--device` | cuda:0 | 训练设备 |
| `--out` | `models/plate_recognizer.pt` | 输出模型路径 |

#### 训练监控

```
epoch=001 loss=1.2345 val_char_acc=0.9234 val_plate_acc=0.8567
epoch=002 loss=0.9567 val_char_acc=0.9412 val_plate_acc=0.8823
...
```

- **val_char_acc**：字符级准确率
- **val_plate_acc**：整牌完全匹配率
- 模型在 `val_plate_acc` 创新高时自动保存

#### 检查点格式

```python
{
    "alphabet": ["皖", "沪", ..., "9", "<blank>"],  # 完整字母表
    "model_state": OrderedDict(...)                   # model.state_dict()
}
```

---

## 3. 推理验证

### 3.1 核心管道

```
视频帧 (BGR, 1920×1080)
  │
  ├─ 1. YOLO-Pose 检测 + ByteTrack 多目标跟踪
  │      detector.track(source=frame, persist=True, tracker="bytetrack.yaml",
  │                       conf=0.25, imgsz=640, device="auto")
  │      输出: bbox (xyxy), confidence, track_id, keypoints (4个角点)
  │
  ├─ 2. 车牌裁剪 (crop_plate)
  │      ├─ 有4个角点 → 透视变换（order_corners → getPerspectiveTransform → warpPerspective）
  │      └─ 无角点   → 矩形裁剪（bbox 扩展 8%水平 + 18%垂直）
  │      输出: 矫正后的车牌图像
  │
  ├─ 3. CRNN 字符识别 (PlateRecognizer.recognize)
  │      输入: 160×48 RGB, [0,1] 归一化
  │      ├─ CNN → BiLSTM → Linear 输出 logits (1, 40, 76)
  │      ├─ softmax → argmax 得到每帧预测字符索引
  │      └─ CTC 贪婪解码 (ctc_greedy_decode)
  │           合并连续相同字符 → 跳过空白 → 输出文本 + 平均置信度
  │      识别间隔: 每 N 帧识别一次（interval=2）
  │
  ├─ 4. 时序投票 (PlateTrackState)
  │      observations: deque[str] 最大长度 = vote_window (15)
  │      投票策略:
  │        a) 找出最常见的文本长度
  │        b) 筛选该长度的候选项
  │        c) 逐字符取众数
  │        例: ["皖AD62208","皖AD82208","皖AD62208"] → "皖AD62208"
  │
  ├─ 5. 绘制标注 (draw_plate_detection)
  │      ├─ 黄色矩形框 (bounding box)
  │      ├─ 青色角点 + 多边形 (4 corners)
  │      ├─ 车牌号 + 置信度 (track ID 上方, 18px)
  │      └─ 稳定车牌号 (框右侧, 30px 绿色大字)
  │
  └─ 6. 输出（可选）
         标注视频: outputs/videos/{stem}_plate_annotated.mp4
         CSV 文件: outputs/csv/{stem}_plate_recognition.csv
```

### 3.2 CSV 输出字段

| 列名 | 说明 |
|------|------|
| `video` | 视频文件名 |
| `frame` | 帧序号 |
| `time_s` | 帧时刻（秒） |
| `track_id` | 跟踪 ID |
| `x1, y1, x2, y2` | 检测框坐标 |
| `plate_conf` | 检测置信度 |
| `raw_text` | 当前帧原始识别结果 |
| `raw_text_conf` | 当前帧识别置信度 |
| `stable_text` | 投票后的稳定车牌号 |

### 3.3 命令行快速测试

```bash
cd D:\研一课内\模式识别与机器视觉\交通任务\traffic_vision

# 测试单张图片
D:\conda_envs\traffic-yolo\python.exe -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from plate_pipeline import PlateRecognizer
import cv2

rec = PlateRecognizer(Path('models/plate_recognizer_other.pt'))
img = cv2.imread('data/processed/other_plate/recognition/train/粤ETJ565_3_13f099850e.jpg')
text, conf = rec.recognize(img)
print(f'Recognized: {text}  |  confidence: {conf:.3f}')
"
```

### 3.4 模型对比

| 特性 | `plate_recognizer.pt` | `plate_recognizer_other.pt` |
|------|----------------------|---------------------------|
| 训练数据 | CCPD（25,716 张） | other/git_plate（56,876 张） |
| 皖占比 | ~50%+ | 6.4% |
| 字母表 | 68 tokens | 76 tokens（+使挂民港澳航领） |
| 省份偏差 | 严重偏向皖 | 多省均衡 |
| 特殊车牌 | 不支持 | 支持使馆/挂车/港澳等 |

---

## 4. GUI 使用

### 4.1 启动方式

```bash
# 方式一：双击
run_plate_gui.bat

# 方式二：终端
D:\conda_envs\traffic-yolo\python.exe -u src\license_plate_gui.py
```

### 4.2 界面布局

```
┌─────────────────────────────────────────────────────────┐
│ [Open Video] [Play/Pause] [Reset]                       │
│ conf: [0.25 ▼]  imgsz: [640 ▼]  device: [auto ▼]      │
│ interval: [2 ▼]  vote: [15 ▼]  ☐ Save video/csv        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                   视频显示区域                            │
│          （检测框 + 角点 + 车牌号叠加）                   │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ video.mp4 | frame=123 | plates=2 | infer+draw=45ms      │
└─────────────────────────────────────────────────────────┘
```

### 4.3 控件说明

| 控件 | 默认值 | 说明 |
|------|--------|------|
| **Open Video** | — | 打开 `videos/check_number/` 下的测试视频 |
| **Play/Pause** | Play | 开始/暂停播放和推理 |
| **Reset** | — | 重置当前视频到第一帧 |
| **conf** | 0.25 | 检测置信度阈值（0.15~0.50） |
| **imgsz** | 640 | 检测推理尺寸（480/640/800） |
| **device** | auto | 推理设备：auto/cpu/0 |
| **interval** | 2 | 识别间隔帧数（每隔 N 帧识别一次，提高帧率） |
| **vote** | 15 | 投票窗口大小（帧数，越大结果越稳定但响应越慢） |
| **Save video/csv** | off | 勾选后保存到 `outputs/videos/` 和 `outputs/csv/` |

### 4.4 操作流程

```
1. 双击 run_plate_gui.bat 启动 GUI
   终端显示:
   [GUI] main() starting...
   [GUI] load_models() starting...
   [GUI] detector: ...\license_plate_det.pt  exists=True
   [GUI] recognizer: ...\plate_recognizer_other.pt  exists=True
   [GUI] pipeline created in 0.8s
   [GUI] Models ready! recognizer loaded
        ↓
2. 点击 Open Video → 选择 videos/check_number/ 下的视频
        ↓
3. 视频自动播放，观察识别效果
   - 黄色框 = 检测到的车牌位置
   - 青色点 = 车牌四角
   - 框上方小字 = #trackID raw_text confidence
   - 框右侧绿色大字 = 投票稳定后的车牌号
        ↓
4. 可选调整：
   - 检测太多误报？→ 调高 conf
   - 检测不到车牌？→ 调低 conf
   - 帧率太低？→ 增大 interval（如 3 或 5）
   - 结果不稳定？→ 增大 vote 窗口（如 20 或 30）
   - 需要实时响应？→ 减小 vote 窗口（如 5 或 10）
        ↓
5. 如需保存结果 → 勾选 Save video/csv
```

### 4.5 切换模型

修改 `src/license_plate_gui.py` 第 32 行：

```python
# CCPD 训练的模型（68 tokens，偏向皖）
# RECOGNIZER_PATH = MODEL_DIR / "plate_recognizer.pt"

# other 数据集训练的模型（76 tokens，多省均衡）
RECOGNIZER_PATH = MODEL_DIR / "plate_recognizer_other.pt"
```

修改后重新启动 GUI 即可。

### 4.6 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|--------|---------|
| 提示 Missing model | 检测/识别模型文件缺失 | 检查 `models/` 下是否有对应的 `.pt` 文件 |
| 显示 NO_RECOGNIZER | 识别模型加载失败 | 检查模型文件是否完整，PyTorch 版本是否匹配 |
| 所有结果都是皖 | 使用了 CCPD 训练的模型 | 切换到 `plate_recognizer_other.pt` |
| OpenCV 导入失败 | conda 环境缺少 opencv | `pip install opencv-python` |
| 中文显示乱码 | 缺少中文字体 | 确保 `C:/Windows/Fonts/msyh.ttc` 存在 |
| 帧率过低 | GPU 不可用或 imgsz 太大 | 设置 device=cpu 或降低 imgsz 到 480 |
