# Cross-platform setup

This project is intended to run on Windows and Linux from the repository root.
Most source paths are resolved with `pathlib`, so commands should use relative
paths such as `src/yolo_speed_ttc_gui.py`, `models/yolo11n.pt`, and
`videos/predict_line/`.

## Python environment

CPU or default PyPI build:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS shell
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda CPU environment:

```bash
conda env create -f environment-cpu.yml
conda activate traffic-yolo
```

Windows CUDA 12.6 environment:

```bash
conda env create -f environment.yml
conda activate traffic-yolo
```

Or install the CUDA-specific pip requirements into an existing environment:

```bash
python -m pip install -r requirements-cuda126.txt
```

## GUI launch

Windows:

```powershell
.\run_yolo_gui.bat
.\run_plate_gui.bat
```

Linux/macOS:

```bash
sh run_yolo_gui.sh
sh run_plate_gui.sh
```

Direct Python commands work on all platforms:

```bash
python src/yolo_speed_ttc_gui.py
python src/license_plate_gui.py
```

## Linux system packages

Tkinter is part of the Python standard library, but some Linux distributions
ship it as a separate package. Install it before launching the GUI if
`import tkinter` fails.

Ubuntu/Debian:

```bash
sudo apt-get install python3-tk
```

For Chinese license plate text rendering on Linux, install a CJK font package if
labels appear as boxes.

Ubuntu/Debian:

```bash
sudo apt-get install fonts-noto-cjk
```

For `prepare_other_plate.py`, extracting `val.rar` requires either `7z` or
`unrar` on `PATH`.

Ubuntu/Debian:

```bash
sudo apt-get install p7zip-full
```

## Output directories

Generated videos and CSV files are written under `outputs/`. Dataset
preprocessing outputs are written under `data/processed/`. These directories are
kept out of version control by `.gitignore`.
