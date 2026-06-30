from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from plate_pipeline import DEFAULT_ALPHABET, PROVINCES, crop_plate


LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")


def parse_point(text: str) -> tuple[float, float]:
    x, y = text.split("&")
    return float(x), float(y)


def parse_ccpd_name(path: Path) -> dict:
    parts = path.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Invalid CCPD filename: {path.name}")
    bbox_a, bbox_b = parts[2].split("_")
    x1, y1 = parse_point(bbox_a)
    x2, y2 = parse_point(bbox_b)
    corners = np.array([parse_point(p) for p in parts[3].split("_")], dtype=np.float32)
    indexes = [int(v) for v in parts[4].split("_")]
    if len(indexes) < 7:
        raise ValueError(f"Invalid CCPD plate code: {path.name}")
    if indexes[0] >= len(PROVINCES) or indexes[1] >= len(LETTERS):
        raise ValueError(f"Invalid CCPD province or letter index: {path.name}")
    if any(i >= len(ADS) for i in indexes[2:]):
        raise ValueError(f"Invalid CCPD alphanumeric index: {path.name}")
    plate = PROVINCES[indexes[0]] + LETTERS[indexes[1]]
    plate += "".join(ADS[i] for i in indexes[2:])
    return {
        "bbox": (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
        "corners": corners,
        "plate": plate,
    }


def yolo_pose_line(width: int, height: int, bbox: tuple[float, float, float, float], corners: np.ndarray) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    values = [0, cx, cy, bw, bh]
    for x, y in corners[:4]:
        values.extend([x / width, y / height, 2])
    return " ".join(str(int(v)) if i == 0 or (i >= 5 and (i - 5) % 3 == 2) else f"{v:.6f}" for i, v in enumerate(values))


def iter_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(root.rglob(ext))
    return sorted(paths)


def collect_images(roots: list[Path]) -> list[Path]:
    images: list[Path] = []
    for root in roots:
        images.extend(iter_images(root))
    return images


def output_name(src: Path) -> str:
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{src.stem}_{digest}{src.suffix.lower()}"


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive.")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    images = list(images)
    random.Random(seed).shuffle(images)
    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)
    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CCPD images to YOLO-Pose and CRNN plate recognition data.")
    parser.add_argument(
        "--ccpd-root",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to downloaded CCPD root directories. Pass multiple roots to mix blue and green plates.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/ccpd_plate"), help="Output directory.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images to sample. Use 0 for all images.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before writing converted data.")
    args = parser.parse_args()

    images = collect_images(args.ccpd_root)
    if args.max_images > 0:
        random.Random(args.seed).shuffle(images)
        images = images[: args.max_images]
    splits = split_images(images, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    yolo_root = args.out / "yolo_pose"
    recog_root = args.out / "recognition"
    split_root = args.out / "splits"
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    for split in splits:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (recog_root / split).mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    for split, split_paths in splits.items():
        (split_root / f"{split}.txt").write_text(
            "".join(f"{p.resolve().as_posix()}\n" for p in split_paths),
            encoding="utf-8",
        )

    manifest_lines: dict[str, list[str]] = {split: [] for split in splits}
    converted_counts: Counter[str] = Counter()
    kept = 0
    skipped = 0
    plate_lengths: Counter[int] = Counter()
    for split, split_paths in splits.items():
        for index, src in enumerate(split_paths, start=1):
            try:
                meta = parse_ccpd_name(src)
                image = cv2.imread(str(src))
                if image is None:
                    raise ValueError("cannot read image")
                h, w = image.shape[:2]
                label = yolo_pose_line(w, h, meta["bbox"], meta["corners"])
                dst_name = output_name(src)
                dst_img = yolo_root / "images" / split / dst_name
                dst_label = yolo_root / "labels" / split / f"{Path(dst_name).stem}.txt"
                shutil.copy2(src, dst_img)
                dst_label.write_text(label + "\n", encoding="utf-8")

                crop = crop_plate(image, meta["bbox"], meta["corners"])
                if crop is not None and crop.size > 0:
                    crop_name = f"{Path(dst_name).stem}.jpg"
                    crop_path = recog_root / split / crop_name
                    cv2.imwrite(str(crop_path), crop)
                    manifest_lines[split].append(f"{crop_path.as_posix()}\t{meta['plate']}\n")
                kept += 1
                converted_counts[split] += 1
                plate_lengths[len(meta["plate"])] += 1
            except Exception:
                skipped += 1
            if index % 1000 == 0:
                print(f"{split}: processed {index}/{len(split_paths)}")

    for split, lines in manifest_lines.items():
        (recog_root / f"{split}.txt").write_text("".join(lines), encoding="utf-8")

    data_yaml = yolo_root / "data.yaml"
    yaml_path = yolo_root.as_posix()
    data_yaml.write_text(
        "\n".join([
            f"path: {yaml_path}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: license_plate",
            "kpt_shape: [4, 3]",
        ]) + "\n",
        encoding="utf-8",
    )
    alphabet_path = recog_root / "alphabet.txt"
    alphabet_path.write_text("\n".join(DEFAULT_ALPHABET) + "\n", encoding="utf-8")
    print(f"found={len(images)}")
    print(f"requested_splits={dict((k, len(v)) for k, v in splits.items())}")
    print(f"converted_splits={dict(converted_counts)}")
    print(f"kept={kept} skipped={skipped}")
    print(f"plate_lengths={dict(sorted(plate_lengths.items()))}")
    print(f"YOLO data: {data_yaml}")
    print(f"Recognizer manifests: {recog_root / 'train.txt'}, {recog_root / 'val.txt'}, {recog_root / 'test.txt'}")


if __name__ == "__main__":
    main()
