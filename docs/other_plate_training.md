# 车牌识别模型训练文档

## 背景

CCPD 数据集中安徽（皖）车牌占比过高，导致原始模型（`plate_recognizer.pt`）对皖牌产生严重偏好，识别其他省份车牌时容易误判为皖。

为解决此问题，使用 `datasets/other/git_plate/` 下的非 CCPD 车牌数据重新训练识别模型，并对字母表进行了扩展。

## 数据概况

| 项目 | 数值 |
|------|------|
| 数据来源 | `datasets/other/git_plate/train.zip`（355 MB） |
| 训练集 | 56,876 张已裁剪车牌图像 |
| 验证集 | 6,320 张（从训练集按 9:1 拆分） |
| 字符类别 | 75 个 token |
| 预处理脚本 | `src/prepare_other_plate.py` |
| 处理结果目录 | `data/processed/other_plate/recognition/` |

### 省份分布（Top 15）

```
粤: 14.9%    川: 11.6%    苏: 10.7%    皖:  6.4%
湘:  5.9%    鲁:  4.7%    浙:  4.6%    鄂:  4.5%
闽:  4.0%    豫:  3.6%    冀:  2.2%    沪:  2.2%
陕:  2.0%    津:  1.9%    藏:  1.9%
```

### 车牌长度分布

```
6位: 114     7位: 61,697     8位: 1,373     9位: 10
```

## 字母表

### 完整字母表（75 个 token）

原字母表（68 tokens）基础上新增 7 个特殊前缀：

```
新增字符: 使 挂 民 港 澳 航 领
```

完整结构：

```
PROVINCES（33个）: 皖 沪 津 渝 冀 晋 蒙 辽 吉 黑 苏 浙 京 闽 赣 鲁 豫 鄂 湘 粤 桂 琼 川 贵 云 藏 陕 甘 青 宁 新 警 学
SPECIAL（7个）:    使 挂 民 港 澳 航 领
LETTERS（25个）:   A B C D E F G H J K L M N P Q R S T U V W X Y Z
DIGITS（10个）:    0 1 2 3 4 5 6 7 8 9
BLANK（1个）:      <blank>
```

### 修改的文件

`src/plate_pipeline.py` 第 45-52 行新增 `SPECIAL_TOKENS` 常量。

## 数据预处理

### 文件名解析规则

"other" 数据集图片已经是裁剪好的车牌图（约 178×66 像素），车牌号直接嵌入在文件名中：

| 格式 | 示例文件名 | 提取的车牌号 |
|------|-----------|-------------|
| `{index}_{plate}_{suffix}` | `2700_皖AD11558_417411.jpg` | `皖AD11558` |
| `{plate}_{variant}` | `171001使_stretch2.jpg` | `171001使` |

### 预处理命令

```bash
cd D:\研一课内\模式识别与机器视觉\交通任务\traffic_vision

D:\conda_envs\traffic-yolo\python.exe src\prepare_other_plate.py --clean
```

选项说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-zip` | `datasets/other/git_plate/train.zip` | 训练数据压缩包 |
| `--val-rar` | `datasets/other/git_plate/val.rar` | 可选验证集（无工具时自动跳过并拆分训练集） |
| `--out` | `data/processed/other_plate/recognition` | 输出目录 |
| `--train-ratio` | `0.9` | 无 val 时的训练/验证拆分比例 |
| `--seed` | `42` | 随机种子 |
| `--clean` | — | 先清空输出目录 |

### 输出文件

```
data/processed/other_plate/recognition/
├── train/                  # 56,876 张训练图像
├── train.txt               # 训练清单（每行: image_path\tplate_text）
├── val.txt                 # 验证清单（每行: image_path\tplate_text）
└── alphabet.txt            # 75 行字母表（最后一行 <blank>）
```

清单格式示例：

```
.../recognition/train/川BA4543_7439_a8d426381e.jpg	川BA4543
.../recognition/train/粤ETJ565_3_13f099850e.jpg	粤ETJ565
```

## 训练

### 模型架构

```
PlateCRNN:
  CNN: Conv(3→32) → BN → ReLU → Pool(2,2)
       Conv(32→64) → BN → ReLU → Pool(2,2)
       Conv(64→128) → BN → ReLU → Pool(2,1)
       Conv(128→256) → BN → ReLU → Pool(2,1)
  RNN: BiLSTM(256→128, 2 layers)
  Head: Linear(256, 75)
  Input: 160×48 RGB
  Loss: CTCLoss(blank=74)
```

### 训练命令

```bash
cd D:\研一课内\模式识别与机器视觉\交通任务\traffic_vision

D:\conda_envs\traffic-yolo\python.exe src\train_plate_recognizer.py ^
  --train data\processed\other_plate\recognition\train.txt ^
  --val data\processed\other_plate\recognition\val.txt ^
  --out models\plate_recognizer_other.pt ^
  --epochs 30 ^
  --batch 64 ^
  --lr 1e-3 ^
  --device cuda:0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train` | — | 训练清单路径 |
| `--val` | — | 验证清单路径 |
| `--out` | `models/plate_recognizer.pt` | 输出模型路径 |
| `--epochs` | `30` | 训练轮数 |
| `--batch` | `64` | 批次大小 |
| `--lr` | `1e-3` | 学习率（AdamW，weight_decay=1e-4） |
| `--device` | `cuda:0` | 训练设备（CPU 可用 `cpu`） |

### 训练监控

每轮输出格式：

```
epoch=001 loss=1.2345 val_char_acc=0.9234 val_plate_acc=0.8567
```

- **val_char_acc**：字符级准确率（每个字符是否预测正确）
- **val_plate_acc**：整牌准确率（完整车牌号是否完全正确）
- 每次 `val_plate_acc` 创新高时自动保存检查点

## 验证

### GUI 中对比

修改 `src/license_plate_gui.py` 第 31 行，切换模型路径：

```python
# 旧模型（CCPD 训练）
# RECOGNIZER_PATH = MODEL_DIR / "plate_recognizer.pt"

# 新模型（other 训练）
RECOGNIZER_PATH = MODEL_DIR / "plate_recognizer_other.pt"
```

然后启动 GUI：

```bash
D:\conda_envs\traffic-yolo\python.exe src\license_plate_gui.py
```

在 `videos/check_number/` 下选择测试视频，对比新旧模型的识别结果（重点关注非皖车牌是否不再被误判为皖）。

### 命令行批量测试

```bash
D:\conda_envs\traffic-yolo\python.exe -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from plate_pipeline import PlateRecognizer
import cv2

# 加载模型
rec = PlateRecognizer(Path('models/plate_recognizer_other.pt'))
print(f'Model loaded: {rec.available}')
print(f'Alphabet size: {len(rec.alphabet)}')

# 测试单张图片
img = cv2.imread('data/processed/other_plate/recognition/train/川BA4543_7439_a8d426381e.jpg')
text, conf = rec.recognize(img)
print(f'Result: {text} (conf={conf:.3f})')
"
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/plate_pipeline.py` | 新增 `SPECIAL_TOKENS`，字母表 68→75 tokens |
| `src/prepare_other_plate.py` | **新建** — other 数据集预处理脚本 |
| `src/train_plate_recognizer.py` | 训练脚本（无需修改，通过参数指定新数据） |
| `models/plate_recognizer_other.pt` | **产出** — 新训练的车牌识别模型 |
| `data/processed/other_plate/recognition/` | 预处理后的训练数据 |
