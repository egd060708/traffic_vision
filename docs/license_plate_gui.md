# 车牌识别 GUI 使用与实现记录

## 1. 目标

本任务实现一个视频车牌识别 GUI，形式与车辆速度和撞线时间预测 GUI 一致：

```text
加载视频 -> 逐帧检测车牌 -> 跟踪同一车牌 -> 裁剪车牌区域 -> 识别车牌文本 -> 在视频中实时叠加结果
```

GUI 入口：

```text
src/license_plate_gui.py
```

快捷启动脚本：

```text
run_plate_gui.bat
```

默认读取视频目录：

```text
videos/check_number
```

## 2. 模型文件

GUI 默认读取两个模型：

```text
models/license_plate_det.pt
models/plate_recognizer.pt
```

其中：

- `license_plate_det.pt`：YOLO-Pose 车牌检测模型，输出车牌框和四个角点。
- `plate_recognizer.pt`：CRNN 车牌字符识别模型，输出完整车牌号。

如果识别模型不存在，GUI 仍可加载检测模型，但识别文本会显示为 `NO_RECOGNIZER`。如果检测模型不存在，GUI 会提示先训练检测模型。

## 3. 数据来源

训练数据由 CCPD2019 和 CCPD2020 处理得到：

```text
data/processed/ccpd_plate/yolo_pose
data/processed/ccpd_plate/recognition
```

划分比例为：

```text
train : val : test = 8 : 1 : 1
```

实际数量：

```text
train: 20572
val:    2571
test:   2573
```

详细数据处理过程见：

```text
docs/ccpd_dataset_processing.md
```

## 4. 训练命令

训练车牌检测模型：

```powershell
D:\conda_envs\traffic-yolo\python.exe src\train_plate_detector.py `
  --data data\processed\ccpd_plate\yolo_pose\data.yaml `
  --save-to models\license_plate_det.pt
```

训练车牌识别模型：

```powershell
D:\conda_envs\traffic-yolo\python.exe src\train_plate_recognizer.py `
  --train data\processed\ccpd_plate\recognition\train.txt `
  --val data\processed\ccpd_plate\recognition\val.txt `
  --out models\plate_recognizer.pt
```

当前环境为 CPU 版 PyTorch，完整训练会比较慢。如果有 NVIDIA GPU，建议安装 CUDA 版 PyTorch 后使用：

```powershell
--device 0
```

## 5. GUI 操作流程

启动：

```powershell
run_plate_gui.bat
```

或者：

```powershell
D:\conda_envs\traffic-yolo\python.exe src\license_plate_gui.py
```

操作步骤：

1. 点击 `Open Video`。
2. 选择 `videos/check_number` 下的视频。
3. 视频播放时，界面会实时显示车牌检测框、跟踪 ID、单帧识别结果和投票后的稳定车牌号。
4. 点击 `Pause` 暂停，点击 `Play` 继续。
5. 点击 `Reset` 从头播放当前视频。

## 6. GUI 参数

```text
conf      检测置信度阈值
imgsz     YOLO 推理尺寸
device    推理设备，auto/cpu/0
interval  每隔多少帧做一次字符识别
vote      对同一 track 的最近若干次识别结果做投票平滑
```

推荐默认值：

```text
conf = 0.25
imgsz = 640
device = auto
interval = 2
vote = 15
```

如果视频卡顿，可以尝试：

```text
imgsz = 480
interval = 3 或 5
```

如果漏检较多，可以尝试：

```text
conf = 0.15 或 0.20
```

## 7. 输出文件

勾选 `Save video/csv` 后，会保存标注视频和逐帧识别结果：

```text
outputs/videos/视频名_plate_annotated.mp4
outputs/csv/视频名_plate_recognition.csv
```

CSV 字段包括：

```text
video
frame
time_s
track_id
x1 y1 x2 y2
plate_conf
raw_text
raw_text_conf
stable_text
```

其中：

- `raw_text` 是当前识别帧的直接输出。
- `stable_text` 是同一车牌 track 在时间窗口内投票后的稳定结果。

## 8. 实现流程

核心代码位于：

```text
src/plate_pipeline.py
src/license_plate_gui.py
```

每帧处理流程：

```text
OpenCV 读取视频帧
-> YOLO-Pose track 检测车牌并分配 track_id
-> 根据检测框和四角点裁剪车牌
-> CRNN 识别车牌字符
-> 对同一 track 的历史识别结果做投票
-> 绘制检测框、角点、单帧结果和稳定结果
-> 显示到 GUI
-> 可选写入视频和 CSV
```

## 9. 当前注意事项

- GUI 功能代码已经实现，但需要先训练或放入 `models/license_plate_det.pt` 才能实际检测车牌。
- `models/plate_recognizer.pt` 缺失时，GUI 仍能显示检测框，但不会给出真实车牌号。
- 中文车牌字符使用 Windows 中文字体绘制，优先读取 `msyh.ttc`、`simhei.ttf`、`simsun.ttc`。

