# 任务一：机动车速度与 TTC 预测

## 概述

本任务利用 YOLO 目标检测 + ByteTrack 多目标跟踪 + 场景标定，实时估计视频中机动车/非机动车的行驶速度和到目标线的时间（TTC，Time-to-Collision）。

## 目录

1. [数据处理](#1-数据处理)
2. [模型训练](#2-模型训练)
3. [推理验证](#3-推理验证)
4. [GUI 使用](#4-gui-使用)

---

## 1. 数据处理

### 1.1 数据来源

测试视频存放在 `videos/predict_line/` 下，按车辆类型分目录：

```
videos/predict_line/
├── bike/                  # 自行车/电动自行车场景
│   ├── 2ad4a9805b000ce3a1595e8f533262c8.mp4
│   ├── 27324182d6014d15d376142c156a42ba.mp4
│   └── b3648868c2639e86d98cf499f4d6f66a.mp4
├── car/                   # 汽车场景
│   ├── 1d82a53ee97b52ce36f65174371ee0e1.mp4
│   ├── e2d9ac4d26b0b16998306b31c37a7a47.mp4
│   ├── 59488add108f8f33741a789effe1b5ca.mp4
│   └── b4b0ee7e0cb7d51bf5bbfff8f9ef74f7.mp4
├── mix/                   # 混合场景（汽车+非机动车）
│   ├── 183918aa66ff21715761a15677857573.mp4
│   ├── 24ba1ba70345987f3527359b04ba74cb.mp4
│   └── cf5656d278c9fe9ff5023f3488a7edf4.mp4
├── cali.jpg               # 场景标定参考图
├── cali_arrow.jpg         # 带标定标注的参考图
└── calibration_reference.md  # 标定参考文档
```

### 1.2 场景标定

标定文件位于 `configs/calibration.json`，是本任务的核心配置。它定义了将图像像素坐标映射为真实世界距离（米）的参数。

#### 标定原理

在场景中选取一条已知长度的道路参考线（12.10 米），将其端点映射到标定参考图的像素坐标上，建立起**一维道路坐标轴**。最终效果是将检测框底边中心点沿道路方向投影，计算其到目标线（人行横道）的距离。

#### 标定文件格式

```json
{
  "reference_size": [1200, 683],
  "target_point": [92, 516],
  "far_point": [1046, 434],
  "axis_length_m": 12.10,
  "target_line": [[70, 540], [210, 410]],
  "roi_polygon": [[0, 390], [1200, 370], [1200, 655], [0, 670]],
  "speed_window": 12,
  "min_approach_speed_mps": 0.15
}
```

#### 关键字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `reference_size` | `[宽, 高]` | 标定参考图的像素尺寸 |
| `target_point` | `[x, y]` | 道路参照轴在目标线（人行横道）上的端点 |
| `far_point` | `[x, y]` | 道路参照轴在远端的端点 |
| `axis_length_m` | 浮点数 | `target_point` 到 `far_point` 的真实世界距离（米） |
| `target_line` | `[[x1,y1],[x2,y2]]` | 目标线的像素位置（用于绘制和动态检测） |
| `roi_polygon` | `[[x,y], ...]` | 检测感兴趣区域多边形，只有 ROI 内的目标才计算速度/TTC |
| `speed_window` | 整数 | 用于线性回归的历史帧数（帧数越多越平滑，但响应越慢） |
| `min_approach_speed_mps` | 浮点数 | TTC 计算所需的最小接近速度（m/s），低于此值不显示 TTC |

#### 三种验证距离（calibration_reference.md）

在标定参考图 `cali_arrow.jpg` 中标注了三个验证箭头：

| 箭头颜色 | 真实长度 |
|----------|---------|
| 🔵 蓝色 | 12.10 m |
| 🟠 橙色 | 9.68 m |
| 🟢 绿色 | 5.00 m |

可用于验证标定精度。实际使用中采用**简化方案**：仅使用蓝色箭头的 12.10 m 轴定义一维道路坐标。

### 1.3 无训练数据

本任务**不需要训练**。YOLO 检测使用 Ultralytics 提供的预训练模型 `yolo11n.pt`（COCO 预训练），直接进行推理。

---

## 2. 模型训练

### 2.1 预训练模型

| 模型 | 大小 | 来源 | 用途 |
|------|------|------|------|
| `yolo11n.pt` | 5.6 MB | Ultralytics 官方 | 目标检测 + ByteTrack 跟踪 |

模型自动检测 COCO 数据集中的以下类别，并映射到项目标签：

| YOLO COCO 类 | COCO 含义 | 项目标签 | 条件 |
|-------------|----------|---------|------|
| 1 | bicycle | 自行车 | 默认；视频路径含 "bike" 时 |
| 2 | car | 汽车 | 默认 |
| 3 | motorcycle | 电动自行车 | 默认；视频路径含 "e-bike" 时 |
| 5 | bus | 巴士 | 默认 |
| 7 | truck | 卡车 | 默认 |

### 2.2 如需微调

若要在特定场景上微调 YOLO 模型（本任务当前无需此步骤）：

```bash
# 1. 用 LabelImg/LabelMe 标注目标检测数据（YOLO 格式）
# 2. 参考 Ultralytics 官方训练命令
yolo detect train data=custom.yaml model=yolo11n.pt epochs=50 imgsz=640
```

---

## 3. 推理验证

### 3.1 核心管道

```
视频帧
  │
  ├─ 1. YOLO11 检测 + ByteTrack 多目标跟踪
  │      model.track(source=frame, persist=True, tracker="bytetrack.yaml",
  │                   classes=[1,2,3,5,7], conf=0.25, imgsz=640)
  │
  ├─ 2. 类别映射（COCO → 项目标签）
  │      car→汽车, bicycle→自行车, motorcycle→电动自行车, bus→巴士, truck→卡车
  │
  ├─ 3. 提取底边中心点 + 估计前接触点
  │      bottom_center = ((x1+x2)/2, y2)  → 缩放到参考坐标系
  │      front_point = 沿底边9个采样点中最接近目标线的点
  │
  ├─ 4. ROI 检查
  │      Calibration.roi_mask() → 生成多边形二值掩码
  │      仅保留 ROI 内的检测框
  │
  ├─ 5. 距离计算
  │      signed_distance_from_target(point)
  │      步骤：
  │      a) 将 point 投影到 target_point→far_point 轴
  │      b) 按 axis_length_m / 像素轴长度 比例缩放
  │      c) 返回带符号距离（目标线前为 +，穿过为 -）
  │
  ├─ 6. TrackState 维护
  │      滑动窗口: deque[(timestamp, signed_distance)], maxlen=12
  │      update():
  │        对最近 ≤speed_window 个样本做线性回归
  │        np.polyfit(timestamps, distances, 1)
  │        speed_kmh = |slope| * 3.6
  │        ttc_s = distance_m / approach_speed_mps
  │        （min_approach_speed_mps = 0.15 m/s 以下不显示 TTC）
  │
  ├─ 7. 绘制标注
  │      不同颜色编码框（汽车绿、自行车青、电动车橙）
  │      Label: "ID: 车种"
  │      Speed: "XX km/h"
  │      TTC:   "XXs"  或  "crossed" 或  "--"
  │      动态目标线（可选）
  │
  └─ 8. 输出（可选）
         标注视频: outputs/videos/{stem}_annotated.mp4
         CSV 文件: outputs/csv/{stem}_yolo_speed_ttc.csv
```

### 3.2 CSV 输出字段

| 列名 | 说明 |
|------|------|
| `video` | 视频文件名 |
| `frame` | 帧序号 |
| `time_s` | 帧时刻（秒） |
| `track_id` | 跟踪 ID |
| `label` | 项目标签（汽车/自行车等） |
| `yolo_class` | YOLO 原始类别号 |
| `x1, y1, x2, y2` | 检测框坐标 |
| `front_x, front_y` | 前接触点坐标 |
| `distance_m` | 距目标线距离（m） |
| `signed_distance_m` | 带符号距离（m） |
| `speed_kmh` | 速度（km/h） |
| `ttc_s` | 到目标线时间（s） |
| `predicted_hit_time_s` | 预测到达时刻 |
| `target_line_source` | 目标线来源（static/dynamic） |
| `target_line_conf` | 目标线置信度 |

### 3.3 命令行测试

```bash
cd D:\研一课内\模式识别与机器视觉\交通任务\traffic_vision

# 直接运行，无 GUI（如需 batch 推断，可调 main 函数）
D:\conda_envs\traffic-yolo\python.exe -c "
from src.yolo_speed_ttc_gui import YoloSpeedTtcApp
# ... 自定义批量推理逻辑
"
```

---

## 4. GUI 使用

### 4.1 启动方式

```bash
# 方式一：双击 bat 文件
run_yolo_gui.bat

# 方式二：终端命令
D:\conda_envs\traffic-yolo\python.exe src\yolo_speed_ttc_gui.py
```

### 4.2 界面布局

```
┌─────────────────────────────────────────────────────────┐
│ [Open Video] [Play/Pause] [Reset]                       │
│ Class: [Auto ▼]  conf: [0.25 ▼]  imgsz: [640 ▼]       │
│ device: [auto ▼]  ☐ Save video/csv                      │
│ ☐ Dynamic line  ☐ Lock line                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                   视频显示区域                            │
│            （检测框 + 速度 + TTC 标注）                  │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ video.mp4 | frame=123 | tracking=3 | infer=45ms          │
└─────────────────────────────────────────────────────────┘
```

### 4.3 控件说明

| 控件 | 默认值 | 说明 |
|------|--------|------|
| **Open Video** | — | 打开 `videos/predict_line/` 下的测试视频 |
| **Play/Pause** | Play | 开始/暂停播放和推理 |
| **Reset** | — | 重置当前视频到第一帧 |
| **Class** | Auto | **类别覆盖**：Auto=根据视频路径自动判断，也可手动选择 car/bicycle/e-bike/vehicle |
| **conf** | 0.25 | YOLO 置信度阈值（0.15~0.50），降低→更多检测框，升高→更少误报 |
| **imgsz** | 640 | 推理图像尺寸（480/640/800），越小越快 |
| **device** | auto | 推理设备：auto=自动选择 GPU，cpu=强制 CPU，0=GPU 0 |
| **Save video/csv** | off | 勾选后保存标注视频和 CSV 到 `outputs/` |
| **Dynamic line** | off | 启用后自动检测目标线位置（视觉方法，基于白线检测 + HoughLinesP） |
| **Lock line** | off | 锁定当前检测到的目标线位置，不再更新 |

### 4.4 操作流程

```
1. 双击 run_yolo_gui.bat 启动 GUI
        ↓
2. 点击 Open Video → 选择 videos/predict_line/ 下的视频
        或直接选择 car/、bike/、mix/ 子目录的视频
        ↓
3. 界面自动开始播放，显示检测框、速度、TTC
        ↓
4. 可选调整：
   - conf 太低？→ 调高减少误检
   - 检测不到？→ 调低 conf、调整 imgsz
   - 分类错误？→ 手动选择 Class 覆盖
        ↓
5. 如需保存结果 → 勾选 Save video/csv
   输出: outputs/videos/{stem}_annotated.mp4
        outputs/csv/{stem}_yolo_speed_ttc.csv
        ↓
6. 使用 Pause/Reset 控制播放
```

### 4.5 状态栏解读

```
videos/predict_line/car/1d82a53e...mp4 | frame=342 | tracking=3 | infer=45ms
│                                         │           │             │
│                                         │           │             └─ 推理耗时
│                                         │           └─ 当前活跃目标数
│                                         └─ 当前帧序号
└─ 视频文件名
```

### 4.6 标注颜色含义

| 颜色 | 目标类型 |
|------|---------|
| 🟢 绿色 | 汽车（car） |
| 🔵 青色 | 自行车（bicycle） |
| 🟠 橙色 | 电动自行车（e-bike / motorcycle） |

### 4.7 TTC 显示规则

| 显示内容 | 条件 |
|---------|------|
| `X.Xs` | 正常接近中，显示预计到达目标线时间 |
| `crossed` | 已穿过目标线（`distance <= 0`） |
| `--` | 接近速度 < 0.15 m/s，或历史帧数 < 5 |

### 4.8 备选 GUI（基于运动检测）

`src/speed_ttc_gui.py` 是**基于背景减除（MOG2）的简化版本**，不依赖 YOLO，不使用 GPU。

特点：
- 使用 `cv2.createBackgroundSubtractorMOG2` 检测运动目标
- 自定义 IoU 跟踪器替代 ByteTrack
- 功能较简单，不保存视频/CSV

仅当 YOLO 版本无法运行时作为备用方案。

### 4.9 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|--------|---------|
| 无检测框 | conf 太高 | 降低 conf 到 0.15~0.20 |
| 帧率过低 | imgsz 太大 | 降低 imgsz 到 480 |
| 分类错误 | 车型混淆 | 手动选择 Class 覆盖 |
| 标定不准 | 视频视角变化 | 重新标定，更新 `calibration.json` |
| 提示缺少模型 | 模型文件丢失 | 确保 `models/yolo11n.pt` 存在 |
