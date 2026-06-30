from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOCAL_ULTRALYTICS_CONFIG = Path(__file__).resolve().parents[1] / ".ultralytics"
LOCAL_ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_ULTRALYTICS_CONFIG))

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",
]
SPECIAL_TOKENS = ["使", "挂", "民", "港", "澳", "航", "领"]
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
DIGITS = list("0123456789")
PLATE_ALPHABET = PROVINCES + SPECIAL_TOKENS + LETTERS + DIGITS
BLANK_TOKEN = "<blank>"
DEFAULT_ALPHABET = PLATE_ALPHABET + [BLANK_TOKEN]


@dataclass
class PlateDetection:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    corners: np.ndarray | None
    crop: np.ndarray | None
    text: str
    text_confidence: float


class PlateCRNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, None)).squeeze(2)
        feat = feat.permute(0, 2, 1)
        seq, _ = self.rnn(feat)
        return self.classifier(seq)


class PlateRecognizer:
    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model_path = model_path
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.alphabet = DEFAULT_ALPHABET
        self.model: PlateCRNN | torch.jit.ScriptModule | None = None
        self.available = False
        if torch is None:
            return
        if model_path.exists():
            self._load(model_path)

    def _load(self, model_path: Path) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "alphabet" in checkpoint:
            self.alphabet = list(checkpoint["alphabet"])
        num_classes = len(self.alphabet)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model = PlateCRNN(num_classes)
            model.load_state_dict(checkpoint["model_state"])
            self.model = model.to(self.device).eval()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model = PlateCRNN(num_classes)
            model.load_state_dict(checkpoint["state_dict"])
            self.model = model.to(self.device).eval()
        else:
            try:
                self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
            except Exception:
                model = PlateCRNN(num_classes)
                model.load_state_dict(checkpoint)
                self.model = model.to(self.device).eval()
        self.available = True

    def recognize(self, crop_bgr: np.ndarray | None) -> tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0
        if not self.available or self.model is None:
            return "NO_RECOGNIZER", 0.0
        image = cv2.resize(crop_bgr, (160, 48), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if logits.ndim == 3 and logits.shape[1] == len(self.alphabet):
                logits = logits.permute(0, 2, 1)
            probs = torch.softmax(logits[0], dim=-1)
            indices = probs.argmax(dim=-1).detach().cpu().numpy().tolist()
            confs = probs.max(dim=-1).values.detach().cpu().numpy().tolist()
        text, used_conf = ctc_greedy_decode(indices, confs, self.alphabet)
        return text, float(np.mean(used_conf)) if used_conf else 0.0


def ctc_greedy_decode(indices: list[int], confs: list[float], alphabet: list[str]) -> tuple[str, list[float]]:
    blank = len(alphabet) - 1
    chars: list[str] = []
    used_conf: list[float] = []
    last = blank
    for idx, conf in zip(indices, confs):
        if idx != blank and idx != last and 0 <= idx < len(alphabet):
            chars.append(alphabet[idx])
            used_conf.append(float(conf))
        last = idx
    return "".join(chars), used_conf


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def crop_plate(frame: np.ndarray, bbox: tuple[float, float, float, float], corners: np.ndarray | None) -> np.ndarray | None:
    h, w = frame.shape[:2]
    if corners is not None and corners.shape[0] >= 4:
        pts = order_corners(corners[:4])
        width = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
        height = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
        width = max(80, min(240, width))
        height = max(24, min(90, height))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(frame, matrix, (width, height))

    x1, y1, x2, y2 = bbox
    pad_x = max(4.0, (x2 - x1) * 0.08)
    pad_y = max(3.0, (y2 - y1) * 0.18)
    xa = max(0, int(math.floor(x1 - pad_x)))
    ya = max(0, int(math.floor(y1 - pad_y)))
    xb = min(w, int(math.ceil(x2 + pad_x)))
    yb = min(h, int(math.ceil(y2 + pad_y)))
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb].copy()


class LicensePlatePipeline:
    def __init__(self, detector_path: Path, recognizer_path: Path, device: str | None = None) -> None:
        if cv2 is None:
            raise RuntimeError(f"OpenCV import failed: {CV2_IMPORT_ERROR}")
        if YOLO is None:
            raise RuntimeError(f"Ultralytics import failed: {YOLO_IMPORT_ERROR}")
        if torch is None:
            raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")
        if not detector_path.exists():
            raise FileNotFoundError(f"License plate detector not found: {detector_path}")
        self.detector = YOLO(str(detector_path))
        self.recognizer = PlateRecognizer(recognizer_path, device=device)

    def infer(
        self,
        frame: np.ndarray,
        frame_id: int,
        conf: float,
        imgsz: int,
        device: str | None,
        recognize: bool = True,
    ) -> list[PlateDetection]:
        result = self.detector.track(
            source=frame,
            persist=frame_id > 0,
            tracker="bytetrack.yaml",
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy)) + 100000
        keypoints = None
        if getattr(result, "keypoints", None) is not None and result.keypoints is not None:
            if result.keypoints.xy is not None:
                keypoints = result.keypoints.xy.cpu().numpy()

        detections: list[PlateDetection] = []
        for i, (box, score, track_id) in enumerate(zip(xyxy, confs, ids)):
            corners = keypoints[i, :4, :] if keypoints is not None and len(keypoints) > i else None
            crop = crop_plate(frame, tuple(float(v) for v in box.tolist()), corners)
            text, text_conf = self.recognizer.recognize(crop) if recognize else ("", 0.0)
            detections.append(
                PlateDetection(
                    track_id=int(track_id),
                    bbox=tuple(float(v) for v in box.tolist()),
                    confidence=float(score),
                    corners=corners,
                    crop=crop,
                    text=text,
                    text_confidence=text_conf,
                )
            )
        return detections


def get_chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_text_box(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: int = 24) -> None:
    if not text:
        return
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = get_chinese_font(scale)
    bbox = draw.textbbox((x, y), text, font=font)
    x2 = min(frame.shape[1] - 1, bbox[2] + 10)
    y2 = min(frame.shape[0] - 1, bbox[3] + 8)
    draw.rectangle((x, y, x2, y2), fill=(20, 20, 20))
    draw.text((x + 5, y + 2), text, font=font, fill=(color[2], color[1], color[0]))
    frame[:, :] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_plate_detection(frame: np.ndarray, det: PlateDetection, stable_text: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
    color = (60, 220, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if det.corners is not None:
        pts = det.corners.astype(np.int32)
        for p in pts[:4]:
            cv2.circle(frame, tuple(p.tolist()), 3, (0, 255, 255), -1)
        cv2.polylines(frame, [pts[:4]], True, (0, 255, 255), 1)
    raw = det.text if det.text else "--"
    label = f"#{det.track_id} {raw} {det.text_confidence:.2f}"
    draw_text_box(frame, label, x1, max(0, y1 - 32), color, scale=18)
    if stable_text:
        text_x = max(0, min(frame.shape[1] - 160, x2 + 8))
        draw_text_box(frame, stable_text, text_x, max(0, y1), (80, 255, 120), scale=30)
