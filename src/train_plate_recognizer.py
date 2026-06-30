from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from plate_pipeline import DEFAULT_ALPHABET, PlateCRNN


ROOT_DIR = Path(__file__).resolve().parents[1]


class PlateTextDataset(Dataset):
    def __init__(self, manifest: Path, alphabet: list[str]) -> None:
        self.samples: list[tuple[Path, str]] = []
        self.alphabet = alphabet
        valid_chars = set(alphabet[:-1])
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path_text = line.split("\t")
            if len(path_text) != 2:
                continue
            path, text = Path(path_text[0]), path_text[1]
            if path.exists() and all(ch in valid_chars for ch in text):
                self.samples.append((path, text))
        self.char_to_idx = {ch: i for i, ch in enumerate(alphabet)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, text = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((48, 160, 3), dtype=np.uint8)
        image = cv2.resize(image, (160, 48), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        target = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        return tensor, target, text


def collate_batch(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return images, flat_targets, target_lengths, texts


def decode_batch(logits: torch.Tensor, alphabet: list[str]) -> list[str]:
    blank = len(alphabet) - 1
    pred = logits.softmax(-1).argmax(-1).detach().cpu().numpy()
    texts: list[str] = []
    for row in pred:
        chars = []
        last = blank
        for idx in row.tolist():
            if idx != blank and idx != last:
                chars.append(alphabet[idx])
            last = idx
        texts.append("".join(chars))
    return texts


def evaluate(model: PlateCRNN, loader: DataLoader, alphabet: list[str], device: torch.device) -> tuple[float, float]:
    model.eval()
    total_chars = 0
    correct_chars = 0
    total_plates = 0
    correct_plates = 0
    with torch.no_grad():
        for images, _targets, _lengths, texts in loader:
            images = images.to(device)
            logits = model(images)
            preds = decode_batch(logits, alphabet)
            for pred, gt in zip(preds, texts):
                total_plates += 1
                correct_plates += int(pred == gt)
                for a, b in zip(pred, gt):
                    correct_chars += int(a == b)
                total_chars += len(gt)
    char_acc = correct_chars / max(total_chars, 1)
    plate_acc = correct_plates / max(total_plates, 1)
    return char_acc, plate_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRNN plate text recognizer on CCPD plate crops.")
    parser.add_argument("--train", type=Path, default=Path("data/processed/ccpd_plate/recognition/train.txt"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/ccpd_plate/recognition/val.txt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("models/plate_recognizer.pt"))
    args = parser.parse_args()

    alphabet = DEFAULT_ALPHABET
    train_set = PlateTextDataset(args.train, alphabet)
    val_set = PlateTextDataset(args.val, alphabet)
    if len(train_set) == 0:
        raise RuntimeError(f"No training samples found in {args.train}")
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device(args.device)
    model = PlateCRNN(len(alphabet)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=len(alphabet) - 1, zero_infinity=True)
    best_plate_acc = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets, target_lengths, _texts in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        char_acc, plate_acc = evaluate(model, val_loader, alphabet, device) if len(val_set) else (0.0, 0.0)
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_char_acc={char_acc:.4f} val_plate_acc={plate_acc:.4f}"
        )
        if plate_acc > best_plate_acc:
            best_plate_acc = plate_acc
            torch.save({"alphabet": alphabet, "model_state": model.state_dict()}, args.out)
            print(f"saved {args.out}")


if __name__ == "__main__":
    main()
