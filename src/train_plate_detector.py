from __future__ import annotations

import argparse
import os
from pathlib import Path

LOCAL_ULTRALYTICS_CONFIG = Path(__file__).resolve().parents[1] / ".ultralytics"
LOCAL_ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_ULTRALYTICS_CONFIG))

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO-Pose license plate detector on converted CCPD data.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/ccpd_plate/yolo_pose/data.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows if workers crash.")
    parser.add_argument("--project", type=Path, default=Path("outputs/train_plate_detector"))
    parser.add_argument("--save-to", type=Path, default=Path("models/license_plate_det.pt"))
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else ROOT_DIR / args.data
    project_path = args.project if args.project.is_absolute() else ROOT_DIR / args.project
    model = YOLO(args.model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name="yolo_pose_plate",
        exist_ok=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    target = args.save_to if args.save_to.is_absolute() else ROOT_DIR / args.save_to
    target.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        target.write_bytes(best.read_bytes())
        print(f"Saved detector to {target}")
    else:
        print(f"Training finished, but best.pt was not found under {best}")


if __name__ == "__main__":
    main()
