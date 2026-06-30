# YOLO 车速与 TTC GUI 执行流程

## 1. 当前项目入口

项目已经实现 YOLO 版 GUI，主入口为：

```text
src/yolo_speed_ttc_gui.py
```

推荐直接运行：

```bat
run_yolo_gui.bat
```

该 bat 会调用已经创建好的 conda 环境：

```text
D:\conda_envs\traffic-yolo\python.exe
```

## 2. 执行前检查

确认以下文件存在：

```text
models/yolo11n.pt
configs/calibration.json
src/yolo_speed_ttc_gui.py
run_yolo_gui.bat
videos/predict_line/
```

如果要手动检查环境，可以在项目根目录运行：

```bat
D:\conda_envs\traffic-yolo\python.exe -c "import cv2, torch; from ultralytics import YOLO; print('ok')"
```

看到：

```text
ok
```

说明依赖正常。

## 3. 启动程序

方式一：双击项目根目录下的：

```text
run_yolo_gui.bat
```

方式二：在终端进入项目根目录后运行：

```bat
run_yolo_gui.bat
```

方式三：直接调用 Python：

```bat
D:\conda_envs\traffic-yolo\python.exe src\yolo_speed_ttc_gui.py
```

启动后会打开一个 GUI 窗口。

## 4. GUI 操作流程

1. 点击 `Open Video`
2. 选择视频文件，例如：

```text
videos/predict_line/bike/*.mp4
videos/predict_line/car/*.mp4
videos/predict_line/mix/*.mp4
```

3. 程序开始播放视频，并实时显示：

```text
检测框
目标 ID
类别
速度 km/h
TTC 到达人行道剩余时间
```

4. 点击 `Pause` 可以暂停。
5. 点击 `Play` 可以继续播放。
6. 点击 `Reset` 可以从视频开头重新运行。

## 5. GUI 参数说明

### Class

类别校正选项：

```text
Auto
car
bicycle
e-bike
bicycle/e-bike
vehicle
```

默认使用 `Auto`。

如果 YOLO 把电动车和自行车分错，可以手动选择：

```text
e-bike
bicycle
bicycle/e-bike
```

### conf

YOLO 检测置信度阈值。

推荐：

```text
0.25
```

如果漏检较多，可以调低：

```text
0.15 或 0.20
```

如果误检较多，可以调高：

```text
0.30 或 0.40
```

### imgsz

YOLO 推理图像尺寸。

推荐：

```text
640
```

CPU 卡顿时可以改为：

```text
480
```

想提高检测精度可以尝试：

```text
800
```

### device

推理设备：

```text
auto
cpu
0
```

当前环境是 CPU 版 PyTorch，建议使用：

```text
auto 或 cpu
```

如果以后安装 CUDA 版 PyTorch，并且电脑有 NVIDIA GPU，可以选择：

```text
0
```

### Save video/csv

勾选后会保存结果。

输出路径：

```text
outputs/videos/
outputs/csv/
```

注意：需要在打开视频前勾选，当前视频才会保存。

## 6. 输出文件

如果勾选 `Save video/csv`，程序会生成：

### 标注视频

```text
outputs/videos/视频名_annotated.mp4
```

视频中包含：

```text
检测框
目标 ID
类别
速度
TTC
目标线
标定参考轴
```

### CSV 表格

```text
outputs/csv/视频名_yolo_speed_ttc.csv
```

字段包括：

```text
video
frame
time_s
track_id
label
yolo_class
x1
y1
x2
y2
distance_m
speed_kmh
ttc_s
```

这些结果可以直接用于实验报告中的结果表和分析。

## 7. 程序内部流程

每一帧的执行流程如下：

```text
读取视频帧
-> YOLO 检测 car / bicycle / motorcycle / bus / truck
-> ByteTrack 关联目标并分配 track_id
-> 取检测框底边中心点作为车辆接地点
-> 判断该点是否在道路 ROI 内
-> 根据 calibration.json 将像素位置转换为真实距离
-> 保存该 track_id 的距离-时间历史
-> 使用最近若干帧做线性拟合估计速度
-> 根据接近速度计算 TTC
-> 绘制检测框、速度、TTC、目标线
-> 显示到 GUI
-> 可选写入视频和 CSV
```

## 8. 速度计算逻辑

每个目标维护一段历史：

```text
(time_s, distance_m)
```

程序使用最近 `speed_window` 帧做线性拟合：

```text
distance = a * time + b
```

速度：

```text
speed_mps = abs(a)
speed_kmh = speed_mps * 3.6
```

如果 `a < 0`，表示目标正在靠近人行道目标线。

## 9. TTC 计算逻辑

TTC 的计算公式：

```text
TTC = 当前剩余距离 / 接近速度
```

其中：

```text
当前剩余距离 = distance_m
接近速度 = -a
```

如果目标没有靠近人行道，或速度太小，则显示：

```text
TTC --
```

如果已经到达或越过目标线，则显示：

```text
crossed
```

## 10. 标定参数修改

标定文件：

```text
configs/calibration.json
```

关键字段：

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

含义：

```text
reference_size       标定图尺寸，对应 cali_arrow.jpg 的 1200x683
target_point         人行道目标线附近参考点
far_point            蓝色标定箭头远端点
axis_length_m        target_point 到 far_point 的真实距离，当前为 12.10m
target_line          GUI 中画出的人行道目标线
roi_polygon          只在该道路区域内统计目标
speed_window         速度拟合使用的历史帧数
min_approach_speed_mps 计算 TTC 的最小接近速度
```

如果速度整体偏大或偏小，优先检查：

```text
target_point
far_point
axis_length_m
```

如果检测到了道路外的行人或树影，可以缩小：

```text
roi_polygon
```

## 11. 常见问题

### GUI 打不开

先检查依赖：

```bat
D:\conda_envs\traffic-yolo\python.exe -c "import cv2, torch; from ultralytics import YOLO; print('ok')"
```

### 提示找不到 yolo11n.pt

确认文件存在：

```text
models/yolo11n.pt
```

如不存在，手动下载：

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
```

下载后放入：

```text
models/yolo11n.pt
```

### 播放卡顿

可以在 GUI 中调整：

```text
imgsz = 480
conf = 0.25
device = cpu
```

也可以后续安装 CUDA 版 PyTorch。

### 检测不到目标

可以尝试：

```text
conf = 0.15
imgsz = 800
```

### 类别不准

使用 `Class` 下拉框手动指定类别。

报告中说明：预训练 YOLO 没有严格的电动自行车类别，因此电动自行车与自行车需要结合场景先验或人工校正。

## 12. 建议实验执行顺序

建议依次测试：

1. `videos/predict_line/car`
2. `videos/predict_line/bike`
3. `videos/predict_line/mix`

每类至少保存一个标注视频和 CSV。

报告中可以挑选：

```text
汽车速度与 TTC 示例
自行车/电动车速度与 TTC 示例
混合场景多目标跟踪示例
失败案例或误差分析
```

