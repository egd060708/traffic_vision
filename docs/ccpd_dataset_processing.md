# CCPD 数据集划分与整理记录

## 1. 数据来源

本项目当前使用的车牌数据主要来自两个 CCPD 数据集目录：

```text
datasets/CCPD2019/ccpd_base
datasets/CCPD2020/CCPD2020/ccpd_green
```

其中：

- `CCPD2019/ccpd_base` 为普通蓝牌数据，当前共 `13940` 张图像。
- `CCPD2020/ccpd_green` 为新能源绿牌数据，当前共 `11776` 张图像。
- 两部分合计 `25716` 张图像。

原始数据目录只作为 raw data 保存，不在其中生成标签、不移动图片。处理后的训练数据统一输出到：

```text
data/processed/ccpd_plate
```

该目录已加入 `.gitignore`，避免大量生成图片和标签进入 Git。

## 2. 划分策略

本次未采用 CCPD2020 下载目录中已有的原始比例，因为其 test 占比偏高。最终统一采用常规训练集划分比例：

```text
train : val : test = 8 : 1 : 1
```

划分使用固定随机种子：

```text
seed = 42
```

这样每次重新处理数据时，只要输入数据不变，就能复现相同的 train、val、test 列表。

最终划分结果：

```text
total: 25716
train: 20572
val:    2571
test:   2573
```

划分清单保存在：

```text
data/processed/ccpd_plate/splits/train.txt
data/processed/ccpd_plate/splits/val.txt
data/processed/ccpd_plate/splits/test.txt
```

## 3. CCPD 文件名解析

CCPD 的标注信息嵌入在文件名中。脚本会从文件名中解析：

- 车牌检测框 bbox
- 车牌四个角点 keypoints
- 车牌字符序列

典型文件名结构如下：

```text
area-tilt-bbox-corners-plate-brightness-blur.jpg
```

其中：

- 第 3 段 `bbox` 形如 `351&564_451&606`，表示左上角和右下角坐标。
- 第 4 段 `corners` 表示车牌四个顶点坐标。
- 第 5 段 `plate` 为车牌字符索引，脚本根据 CCPD 字符表还原成真实车牌文本。

处理脚本位置：

```text
src/prepare_ccpd.py
```

## 4. 输出数据结构

脚本会同时生成两类任务的数据：车牌检测数据和车牌识别数据。

### 4.1 YOLO-Pose 检测数据

输出目录：

```text
data/processed/ccpd_plate/yolo_pose
```

目录结构：

```text
yolo_pose/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

标签格式为 YOLO-Pose：

```text
class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
```

当前只有一个类别：

```text
0: license_plate
```

关键点数量为 4，每个关键点包含：

```text
x y visibility
```

`data.yaml` 内容：

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: license_plate
kpt_shape: [4, 3]
```

检测数据校验结果：

```text
train images: 20572, labels: 20572
val   images:  2571, labels:  2571
test  images:  2573, labels:  2573
```

### 4.2 车牌识别数据

输出目录：

```text
data/processed/ccpd_plate/recognition
```

目录结构：

```text
recognition/
  alphabet.txt
  train.txt
  val.txt
  test.txt
  train/
  val/
  test/
```

处理流程为：

1. 读取原始图像。
2. 根据 CCPD 文件名解析 bbox 和四角点。
3. 对车牌区域做透视裁剪。
4. 保存裁剪后的车牌小图。
5. 在 manifest 中记录图片路径和车牌文本。

manifest 格式：

```text
plate_crop_path<TAB>plate_text
```

识别数据校验结果：

```text
train.txt: 20572
val.txt:    2571
test.txt:   2573
alphabet.txt: 68
```

本次处理得到的车牌长度统计：

```text
7 位车牌: 13940
8 位车牌: 11776
```

这对应 CCPD2019 普通蓝牌和 CCPD2020 新能源绿牌。

## 5. 实际执行命令

完整数据处理命令：

```powershell
D:\conda_envs\traffic-yolo\python.exe src\prepare_ccpd.py `
  --ccpd-root datasets\CCPD2019\ccpd_base datasets\CCPD2020\CCPD2020\ccpd_green `
  --out data\processed\ccpd_plate `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1 `
  --seed 42 `
  --clean
```

参数说明：

- `--ccpd-root`：输入 CCPD 图像目录，可以传入多个目录。
- `--out`：处理后数据输出目录。
- `--train-ratio`、`--val-ratio`、`--test-ratio`：数据划分比例。
- `--seed`：固定随机种子，用于可复现划分。
- `--clean`：处理前删除旧输出目录，避免混入历史结果。

## 6. 处理结果日志

本次完整处理结果：

```text
found=25716
requested_splits={'train': 20572, 'val': 2571, 'test': 2573}
converted_splits={'train': 20572, 'val': 2571, 'test': 2573}
kept=25716 skipped=0
plate_lengths={7: 13940, 8: 11776}
```

说明：

- `found` 表示找到的原始图片数量。
- `requested_splits` 表示划分后每个集合应处理的图片数量。
- `converted_splits` 表示实际成功转换的图片数量。
- `kept=25716` 表示全部图片成功处理。
- `skipped=0` 表示没有图片因文件名非法、读取失败或标注异常被跳过。

## 7. 后续训练入口

训练车牌检测模型：

```powershell
D:\conda_envs\traffic-yolo\python.exe src\train_plate_detector.py `
  --data data\processed\ccpd_plate\yolo_pose\data.yaml
```

训练车牌识别模型：

```powershell
D:\conda_envs\traffic-yolo\python.exe src\train_plate_recognizer.py `
  --train data\processed\ccpd_plate\recognition\train.txt `
  --val data\processed\ccpd_plate\recognition\val.txt
```

## 8. 其他数据集处理计划

当前 `datasets/other/git_plate` 下还有额外压缩包：

```text
datasets/other/git_plate/train.zip
datasets/other/git_plate/val.rar
```

参考仓库 `PRMV-Course/Practice2/ccpd_license_preprocess.py` 中的解压方式可以继续处理这些数据。后续接入前需要先确认其标注格式：

- 如果文件名仍为 CCPD 风格，可以复用 `src/prepare_ccpd.py` 的解析逻辑。
- 如果有独立的 txt、xml 或 json 标注，需要先编写格式转换 adapter。
- 如果只有图片没有标注，则不直接混入训练集，可作为人工标注、测试展示或半监督数据来源。

