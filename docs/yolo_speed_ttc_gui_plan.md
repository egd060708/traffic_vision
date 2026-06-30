# YOLO 车速与到达人行道时间预测 GUI 方案

## 1. 目标

运行一个脚本后打开 GUI，用户可以导入 `videos/predict_line` 下的视频。视频播放时，系统实时显示：

- 目标检测框
- 目标 ID
- 目标类别：电动自行车 / 自行车 / 汽车
- 当前速度
- 到达人行道或停止线的剩余时间 TTC

最终可选输出：

- 带标注的视频
- 每帧检测结果 CSV
- 每个目标的平均速度、最大速度、最小 TTC

## 2. 推荐技术路线

采用 `YOLO + ByteTrack/BoT-SORT + 场景标定 + 轨迹速度估计`。

整体流程：

```text
GUI 选择视频
-> OpenCV 逐帧读取
-> YOLO 检测车辆/自行车/电动车候选目标
-> YOLO track 模式输出 track_id
-> 取检测框底边中心点作为路面接地点
-> 根据 cali_arrow.jpg 的标定线换算真实距离
-> 对同一 track_id 的距离-时间序列做平滑测速
-> TTC = 剩余距离 / 接近速度
-> 在视频画面上绘制框、速度、TTC
```

## 3. 为什么直接用 YOLO

YOLO 的优势是检测结果稳定、代码短、可复现性好。Ultralytics 官方 Python API 可以直接加载预训练模型，例如：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
```

`track` 模式会在检测框之外提供目标 ID，适合本实验的速度估计。官方文档也说明 YOLO tracking 输出包含标准检测结果和对象 ID，支持 BoT-SORT 与 ByteTrack。

## 4. 模型选择

优先推荐：

```text
yolo11n.pt
```

理由：

- 速度快，适合 GUI 实时播放
- 预训练 COCO 类别包含 `car`、`bicycle`、`motorcycle`
- 对本实验来说，检测框稳定性比极限精度更重要

备选：

```text
yolo11s.pt
```

如果你的电脑有 NVIDIA GPU，`yolo11s.pt` 会更准一些；如果 CPU 运行卡顿，用 `yolo11n.pt`。

也可以尝试新版：

```text
yolo26n.pt
```

不过为了课程作业复现稳定，我建议先用 `yolo11n.pt`，因为它更成熟、资料更多。

## 5. 类别处理策略

COCO 预训练模型没有严格的“电动自行车”类别，因此采用下面策略：

| YOLO 原始类别 | 本项目类别 |
| --- | --- |
| `car` | 汽车 |
| `bicycle` | 自行车 |
| `motorcycle` | 电动自行车候选 |

对于 `bike` 文件夹里的视频，可以在 GUI 里提供一个类别下拉框：

```text
Auto / car / bicycle / e-bike / bicycle-or-e-bike
```

这样报告中可以说明：由于公开预训练模型没有电动自行车专类，本文使用 `motorcycle` 和 `bicycle` 检测结果，并结合视频文件夹先验或人工下拉选项进行类别校正。

## 6. 场景标定方案

使用 `videos/predict_line/cali_arrow.jpg`。

已知真实距离：

```text
蓝色箭头：12.10 m
橙色箭头：9.68 m
绿色箭头：5.00 m
```

推荐第一版采用一维道路坐标：

1. 在 GUI 或配置文件中记录人行道目标线位置。
2. 记录蓝色箭头两端点，建立从人行道到道路右侧的参考轴。
3. 每个检测框取底边中心点。
4. 将该点投影到参考轴上。
5. 按蓝色箭头 `12.10 m` 换算剩余距离。

简化公式：

```text
s = projection(pixel_point - target_point, road_axis_unit)
distance_m = s / road_axis_pixel_length * 12.10
```

其中：

- `target_point` 是人行道/斑马线边界点
- `road_axis_unit` 是沿车辆靠近人行道方向的单位向量
- `distance_m` 是车辆到底线的剩余距离

如果后续要提高精度，可以加入橙色、绿色标定线，做分段比例尺或透视变换。

## 7. 速度估计

对每个 `track_id` 维护历史：

```text
[(timestamp_0, distance_0),
 (timestamp_1, distance_1),
 ...
 (timestamp_n, distance_n)]
```

不用相邻两帧直接差分，因为检测框会抖动。推荐取最近 8 到 15 帧做线性拟合：

```text
distance = a * time + b
speed_mps = abs(a)
speed_kmh = speed_mps * 3.6
```

若 `a < 0`，说明车辆在靠近人行道；若 `a >= 0`，说明目标远离或横向运动，不预测 TTC。

## 8. TTC 预测

```text
TTC = remaining_distance / approach_speed
```

其中：

```text
remaining_distance = 当前目标到底线距离
approach_speed = -a
```

异常处理：

```text
approach_speed < 0.15 m/s -> TTC 显示 --
remaining_distance <= 0 -> TTC 显示 0.0s 或 crossed
track 历史帧少于 5 帧 -> 速度和 TTC 暂不显示
```

## 9. GUI 设计

使用 `tkinter + OpenCV + PIL.ImageTk`，优点是 Python 自带 tkinter，依赖少，老师电脑更容易运行。

界面控件：

```text
Open Video      打开视频
Play/Pause      播放/暂停
Reset           从头播放
Class Override  类别校正
Save Output     可选：保存标注视频
```

画面叠加：

```text
#3 car 22.4 km/h TTC 1.8s
#5 e-bike 14.2 km/h TTC 2.6s
```

同时画出：

- 人行道目标线
- 标定参考轴
- 检测框底边中心点

## 10. 项目结构

建议结构：

```text
traffic_vision/
├─ videos/predict_line/
├─ models/
│  └─ yolo11n.pt
├─ configs/
│  └─ calibration.json
├─ src/
│  ├─ yolo_speed_ttc_gui.py
│  ├─ calibration_tool.py
│  └─ speed_utils.py
├─ outputs/
│  ├─ videos/
│  └─ csv/
├─ environment.yml
└─ docs/
   └─ yolo_speed_ttc_gui_plan.md
```

第一版可以只实现：

```text
src/yolo_speed_ttc_gui.py
configs/calibration.json
environment.yml
```

## 11. Conda 环境

环境名建议：

```text
traffic-yolo
```

CPU 版本：

```bash
conda create -n traffic-yolo python=3.10 -y
conda activate traffic-yolo
pip install torch torchvision torchaudio
pip install ultralytics opencv-python pillow numpy pandas
```

NVIDIA GPU 版本：

```bash
conda create -n traffic-yolo python=3.10 -y
conda activate traffic-yolo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pillow numpy pandas
```

如果你的显卡驱动支持 CUDA 12.6 或 12.8，可以到 PyTorch 官网复制对应安装命令。

环境验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
```

## 12. 需要下载的东西

### 必须安装

1. Anaconda 或 Miniconda

   官网：

   ```text
   https://www.anaconda.com/download
   https://docs.conda.io/projects/miniconda/en/latest/
   ```

2. PyTorch

   官网安装选择器：

   ```text
   https://pytorch.org/get-started/locally/
   ```

3. Ultralytics YOLO

   官方文档：

   ```text
   https://docs.ultralytics.com/quickstart/
   ```

   GitHub：

   ```text
   https://github.com/ultralytics/ultralytics
   ```

### 模型权重

推荐使用自动下载：第一次运行下面代码时，Ultralytics 会自动下载权重。

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
```

如果要手动下载，可以把权重放到项目的 `models/` 目录：

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

第一版建议下载：

```text
models/yolo11n.pt
```

## 13. 运行方式

计划最终运行命令：

```bash
conda activate traffic-yolo
python src/yolo_speed_ttc_gui.py
```

本项目当前已经创建了 `run_yolo_gui.bat`，在当前电脑上可以直接双击或在终端运行：

```bash
run_yolo_gui.bat
```

如果不使用 bat，也可以直接运行：

```bash
D:\conda_envs\traffic-yolo\python.exe src\yolo_speed_ttc_gui.py
```

GUI 打开后：

1. 点击 `Open Video`
2. 选择 `videos/predict_line/bike`、`car` 或 `mix` 下的视频
3. 播放时查看检测框、速度、TTC
4. 如类别不准，使用 `Class Override` 校正类别

## 14. 实现优先级

第一阶段：

- GUI 打开视频
- YOLO 检测和跟踪
- 显示检测框、类别、ID
- 根据默认标定参数显示速度和 TTC

当前已实现第一阶段功能，并额外实现：

- `Save video/csv` 开关
- 标注视频输出到 `outputs/videos/`
- CSV 输出到 `outputs/csv/`
- 默认 YOLO 权重放在 `models/yolo11n.pt`

第二阶段：

- 保存标注视频和 CSV
- 加入标定点可视化编辑
- 对速度做更好的异常值过滤

第三阶段：

- 如果电动车和自行车区分不理想，准备少量本场景截图做 YOLO 微调
- 加入更精确的透视变换标定

## 15. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 电动车和自行车类别混淆 | 类别显示不准 | 使用 GUI 类别校正，报告中说明预训练模型限制 |
| 检测框抖动 | 速度跳变 | 最近 8 到 15 帧线性拟合 |
| 车辆遮挡 | track_id 可能切换 | 使用 ByteTrack，短时丢失不立刻删除轨迹 |
| 标定不准 | 速度和 TTC 有系统误差 | 保留标定点配置，后续可手动微调 |
| CPU 推理慢 | GUI 卡顿 | 使用 `yolo11n.pt`，降低 `imgsz=480` 或隔帧检测 |

## 16. 报告可写的创新点

- 使用 YOLO 跟踪代替传统帧差法，提高复杂背景下的检测稳定性。
- 使用检测框底边中心点作为路面接地点，降低车辆高度带来的透视误差。
- 使用标定线建立真实距离映射，实现像素运动到实际速度的转换。
- 使用多帧线性拟合估计速度，减少单帧检测抖动。
- 同时输出速度和 TTC，满足实验二“车速和撞线时间估计”的要求。
