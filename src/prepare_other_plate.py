from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from plate_pipeline import DEFAULT_ALPHABET

ROOT_DIR = Path(__file__).resolve().parents[1]

_INVALID_CHARS_RE = re.compile(r"[()\s]")


def _decode_zip_filename(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    """Decode a ZIP filename that may be GBK-encoded."""
    raw = info.filename.encode("cp437")
    if info.flag_bits & 0x800:
        return raw.decode("utf-8")
    try:
        return raw.decode("gbk")
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def _parse_plate_from_stem(stem: str) -> str | None:
    """Extract plate text from a filename stem.

    Format 1: ``{plate}_{variant}``     → plate = parts[0]
    Format 2: ``{idx}_{plate}_{suffix}`` → plate = parts[1]
    """
    parts = stem.split("_")
    if len(parts) == 2:
        return parts[0]
    if len(parts) >= 3:
        return parts[1]
    return None


def _is_valid_plate(text: str, valid_chars: set[str]) -> bool:
    if not text:
        return False
    return all(ch in valid_chars for ch in text)


def _clean_plate(text: str) -> str:
    return _INVALID_CHARS_RE.sub("", text)


def _safe_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}{path.suffix.lower()}"


def _extract_rar(rar_path: Path, dest_dir: Path) -> list[Path]:
    """Extract RAR archive using 7z or unrar. Returns sorted image paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("7z") or shutil.which("7z.exe") or shutil.which("unrar")
    if exe is None:
        raise RuntimeError(
            "Cannot find 7z or unrar. Install 7-Zip and add to PATH."
        )
    cmd = [exe, "x", str(rar_path.resolve()), f"-o{str(dest_dir.resolve())}", "-y"]
    if "unrar" in exe.lower():
        cmd = [exe, "x", str(rar_path.resolve()), str(dest_dir.resolve()) + "/", "-y"]
    subprocess.run(cmd, check=True, capture_output=True)

    images: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(dest_dir.rglob(ext))
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 'other' plate dataset for CRNN recognition training."
    )
    parser.add_argument("--train-zip", type=Path, default=Path("datasets/other/git_plate/train.zip"))
    parser.add_argument("--val-rar", type=Path, default=Path("datasets/other/git_plate/val.rar"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/other_plate/recognition"))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    train_zip = args.train_zip if args.train_zip.is_absolute() else ROOT_DIR / args.train_zip
    val_rar = args.val_rar if args.val_rar.is_absolute() else ROOT_DIR / args.val_rar
    out_dir = args.out if args.out.is_absolute() else ROOT_DIR / args.out

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_chars = set(DEFAULT_ALPHABET)
    valid_chars.discard("<blank>")

    # ── Extract train.zip ────────────────────────────────────────────
    print("=" * 60)
    print("Extracting train.zip ...")
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    train_samples: list[tuple[Path, str]] = []
    if not train_zip.exists():
        raise FileNotFoundError(f"train.zip not found: {train_zip}")

    with zipfile.ZipFile(train_zip, "r") as zf:
        total = sum(1 for i in zf.infolist()
                    if i.filename.lower().endswith((".jpg", ".jpeg", ".png")))
        processed = 0
        for info in zf.infolist():
            orig_name = _decode_zip_filename(zf, info)
            if not orig_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = orig_name.rsplit(".", 1)[0]
            plate = _parse_plate_from_stem(stem)
            if plate is None:
                continue
            plate = _clean_plate(plate)
            if not _is_valid_plate(plate, valid_chars):
                continue

            dst_path = train_dir / _safe_name(Path(orig_name))
            if not dst_path.exists():
                tmp = train_dir / f"_{dst_path.name}"
                with zf.open(info) as src_f, open(tmp, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                tmp.rename(dst_path)

            train_samples.append((dst_path, plate))
            processed += 1
            if processed % 5000 == 0:
                print(f"  processed {processed}/{total} ...")

    print(f"  kept {len(train_samples)} / {total} images")

    # ── Extract val.rar ──────────────────────────────────────────────
    val_samples: list[tuple[Path, str]] = []
    if val_rar.exists():
        print("=" * 60)
        print("Extracting val.rar ...")
        val_dir = out_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        try:
            val_images = _extract_rar(val_rar, val_dir)
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            print("  Skipping val.rar — will split train data for validation instead.")
            val_images = []
        else:
            for img_path in val_images:
                stem = img_path.stem
                plate = _parse_plate_from_stem(stem)
                if plate is None:
                    continue
                plate = _clean_plate(plate)
                if not _is_valid_plate(plate, valid_chars):
                    continue
                val_samples.append((img_path, plate))
            print(f"  kept {len(val_samples)} val images")
    else:
        print("val.rar not found — will split train for validation")

    # ── Split if no val data ─────────────────────────────────────────
    if not val_samples:
        random.Random(args.seed).shuffle(train_samples)
        split = int(len(train_samples) * args.train_ratio)
        val_samples = train_samples[split:]
        train_samples = train_samples[:split]
        print(f"  split: {len(train_samples)} train / {len(val_samples)} val")

    # ── Write manifests ──────────────────────────────────────────────
    print("=" * 60)
    print("Writing manifests ...")
    for tag, samples in [("train", train_samples), ("val", val_samples)]:
        manifest_path = out_dir / f"{tag}.txt"
        lines = [f"{img_path.as_posix()}\t{plate}\n" for img_path, plate in samples]
        manifest_path.write_text("".join(lines), encoding="utf-8")
        print(f"  {manifest_path.name}: {len(lines)} samples")

    # ── Statistics ───────────────────────────────────────────────────
    print("=" * 60)
    print("Statistics:")
    all_plates = [p for _, p in train_samples] + [p for _, p in val_samples]
    province_counter: Counter[str] = Counter()
    plate_len_counter: Counter[int] = Counter()
    for plate in all_plates:
        for ch in plate:
            if "一" <= ch <= "鿿":
                province_counter[ch] += 1
                break
        else:
            province_counter["(other)"] += 1
        plate_len_counter[len(plate)] += 1

    print(f"  total plates: {len(all_plates)}")
    print(f"  province distribution (top 15):")
    for prov, cnt in province_counter.most_common(15):
        print(f"    {prov}: {cnt} ({cnt / len(all_plates) * 100:.1f}%)")
    print(f"  plate lengths: {dict(sorted(plate_len_counter.items()))}")

    # ── Save alphabet ────────────────────────────────────────────────
    alpha_path = out_dir / "alphabet.txt"
    alpha_path.write_text("\n".join(DEFAULT_ALPHABET) + "\n", encoding="utf-8")
    print(f"\nAlphabet saved to {alpha_path}")
    print("Done.")


if __name__ == "__main__":
    main()
