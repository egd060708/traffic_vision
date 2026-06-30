from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import BOTH, BOTTOM, DISABLED, LEFT, NORMAL, TOP, Button, Checkbutton, Frame, Label, StringVar, Tk, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

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
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "videos" / "predict_line"
MODEL_DIR = ROOT_DIR / "models"
CONFIG_PATH = ROOT_DIR / "configs" / "calibration.json"
OUTPUT_VIDEO_DIR = ROOT_DIR / "outputs" / "videos"
OUTPUT_CSV_DIR = ROOT_DIR / "outputs" / "csv"

YOLO_CLASS_FILTER = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck in COCO


@dataclass
class Calibration:
    reference_size: tuple[int, int]
    target_point: tuple[float, float]
    far_point: tuple[float, float]
    axis_length_m: float
    target_line: tuple[tuple[float, float], tuple[float, float]]
    roi_polygon: tuple[tuple[float, float], ...]
    speed_window: int = 12
    min_approach_speed_mps: float = 0.15

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            reference_size=tuple(data["reference_size"]),
            target_point=tuple(data["target_point"]),
            far_point=tuple(data["far_point"]),
            axis_length_m=float(data["axis_length_m"]),
            target_line=(tuple(data["target_line"][0]), tuple(data["target_line"][1])),
            roi_polygon=tuple(tuple(p) for p in data["roi_polygon"]),
            speed_window=int(data.get("speed_window", 12)),
            min_approach_speed_mps=float(data.get("min_approach_speed_mps", 0.15)),
        )

    def to_reference(self, point: tuple[float, float], frame_shape: tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        ref_w, ref_h = self.reference_size
        return np.array([point[0] * ref_w / w, point[1] * ref_h / h], dtype=np.float32)

    def from_reference(self, point: tuple[float, float], frame_shape: tuple[int, int, int]) -> tuple[int, int]:
        h, w = frame_shape[:2]
        ref_w, ref_h = self.reference_size
        return int(round(point[0] * w / ref_w)), int(round(point[1] * h / ref_h))

    def roi_mask(self, frame_shape: tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        pts = np.array([self.from_reference(p, frame_shape) for p in self.roi_polygon], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def signed_distance_from_target(
        self,
        point: tuple[float, float],
        frame_shape: tuple[int, int, int],
        target_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> float:
        p = self.to_reference(point, frame_shape)
        line = target_line if target_line is not None else self.target_line
        line_a = np.array(line[0], dtype=np.float32)
        line_b = np.array(line[1], dtype=np.float32)
        line_vec = line_b - line_a
        start = np.array(self.target_point, dtype=np.float32)
        end = np.array(self.far_point, dtype=np.float32)
        axis = end - start
        axis_len_px = float(np.linalg.norm(axis))
        if axis_len_px <= 1e-6:
            return 0.0
        unit = axis / axis_len_px
        denom = float(line_vec[0] * unit[1] - line_vec[1] * unit[0])
        if abs(denom) <= 1e-6:
            projected_px = float(np.dot(p - line_a, unit))
        else:
            delta = p - line_a
            projected_px = float((delta[0] * line_vec[1] - delta[1] * line_vec[0]) / denom)
        return -projected_px / axis_len_px * self.axis_length_m

    def distance_from_target(self, point: tuple[float, float], frame_shape: tuple[int, int, int]) -> float:
        return max(0.0, self.signed_distance_from_target(point, frame_shape))

    def in_roi(self, point: tuple[float, float], frame_shape: tuple[int, int, int]) -> bool:
        mask = self.roi_mask(frame_shape)
        x, y = int(point[0]), int(point[1])
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            return False
        return mask[y, x] > 0

    def draw(
        self,
        frame: np.ndarray,
        target_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
        line_source: str = "fallback",
        line_confidence: float = 0.0,
    ) -> None:
        active_line = target_line if target_line is not None else self.target_line
        line_a = self.from_reference(active_line[0], frame.shape)
        line_b = self.from_reference(active_line[1], frame.shape)
        fallback_a = self.from_reference(self.target_line[0], frame.shape)
        fallback_b = self.from_reference(self.target_line[1], frame.shape)
        axis_a = self.from_reference(self.target_point, frame.shape)
        axis_b = self.from_reference(self.far_point, frame.shape)
        roi = np.array([self.from_reference(p, frame.shape) for p in self.roi_polygon], dtype=np.int32)

        overlay = frame.copy()
        cv2.polylines(overlay, [roi], True, (80, 80, 80), 1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        if line_source != "fallback":
            cv2.line(frame, fallback_a, fallback_b, (120, 120, 120), 1)
        draw_a, draw_b = line_a, line_b
        cv2.line(frame, draw_a, draw_b, (0, 0, 255), 3)
        cv2.circle(frame, line_a, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"target {line_source} {line_confidence:.2f}", (draw_a[0] + 8, max(24, draw_a[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)


@dataclass
class TargetLineEstimate:
    line: tuple[tuple[float, float], tuple[float, float]]
    confidence: float
    source: str


def line_angle_deg(line: tuple[tuple[float, float], tuple[float, float]]) -> float:
    (x1, y1), (x2, y2) = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def angle_delta_deg(a: float, b: float) -> float:
    diff = (a - b + 90.0) % 180.0 - 90.0
    return abs(diff)


def line_from_fit_to_rect(vx: float, vy: float, x0: float, y0: float, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
    points: list[tuple[float, float]] = []
    if abs(vx) > 1e-6:
        for x in (0.0, float(width - 1)):
            t = (x - x0) / vx
            y = y0 + t * vy
            if 0.0 <= y <= height - 1:
                points.append((x, y))
    if abs(vy) > 1e-6:
        for y in (0.0, float(height - 1)):
            t = (y - y0) / vy
            x = x0 + t * vx
            if 0.0 <= x <= width - 1:
                points.append((x, y))
    if len(points) < 2:
        return None
    best = max(
        ((points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    return best


class TargetLineEstimator:
    def __init__(self, calibration: Calibration) -> None:
        self.calibration = calibration
        self.line_ref = calibration.target_line
        self.confidence = 0.0
        self.source = "fallback"
        self.fail_count = 0
        self.offset_history: deque[float] = deque(maxlen=11)
        self.offset_px = 0.0

    def update(self, frame: np.ndarray, enabled: bool, locked: bool) -> TargetLineEstimate:
        if not enabled:
            self.line_ref = self.calibration.target_line
            self.confidence = 0.0
            self.source = "fallback"
            return TargetLineEstimate(self.line_ref, self.confidence, self.source)
        if locked:
            source = "locked" if self.source != "fallback" else "fallback"
            return TargetLineEstimate(self.line_ref, self.confidence, source)

        detected = self._detect_parallel_offset(frame)
        if detected is None:
            detected = self._detect(frame)
        if detected is None:
            self.fail_count += 1
            if self.fail_count > 20 and self.source == "fallback":
                self.line_ref = self.calibration.target_line
            return TargetLineEstimate(self.line_ref, self.confidence, self.source)

        line, confidence = detected
        self.fail_count = 0
        if confidence < 0.25:
            return TargetLineEstimate(self.line_ref, self.confidence, self.source)

        alpha = 0.35 if self.source != "fallback" else 0.75
        self.line_ref = tuple(
            tuple(float((1.0 - alpha) * old + alpha * new) for old, new in zip(old_pt, new_pt))
            for old_pt, new_pt in zip(self.line_ref, line)
        )
        self.confidence = confidence
        self.source = "dynamic"
        return TargetLineEstimate(self.line_ref, self.confidence, self.source)

    def _detect_parallel_offset(self, frame: np.ndarray) -> tuple[tuple[tuple[float, float], tuple[float, float]], float] | None:
        h, w = frame.shape[:2]
        roi = self.calibration.roi_mask(frame.shape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 140], dtype=np.uint8), np.array([179, 105, 255], dtype=np.uint8))
        white = cv2.bitwise_and(white, roi)
        white = cv2.morphologyEx(
            white,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
            iterations=1,
        )
        edges = cv2.Canny(white, 40, 130)
        ys, xs = np.where(edges > 0)
        if len(xs) < 24:
            return None

        line_a = np.array(self.calibration.target_line[0], dtype=np.float32)
        line_b = np.array(self.calibration.target_line[1], dtype=np.float32)
        frame_a = np.array(self.calibration.from_reference(tuple(line_a), frame.shape), dtype=np.float32)
        frame_b = np.array(self.calibration.from_reference(tuple(line_b), frame.shape), dtype=np.float32)
        direction = frame_b - frame_a
        line_len = float(np.linalg.norm(direction))
        if line_len <= 1e-6:
            return None
        direction /= line_len
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        offsets = (pts[:, 0] - frame_a[0]) * normal[0] + (pts[:, 1] - frame_a[1]) * normal[1]
        along = (pts[:, 0] - frame_a[0]) * direction[0] + (pts[:, 1] - frame_a[1]) * direction[1]
        band = max(50.0, min(115.0, w * 0.085))
        along_pad = max(float(w), float(h))
        keep = (np.abs(offsets) <= band) & (along >= -along_pad) & (along <= line_len + along_pad)
        offsets = offsets[keep]
        if len(offsets) < 24:
            return None

        q1, q3 = np.percentile(offsets, [25, 75])
        iqr = max(8.0, float(q3 - q1))
        robust = offsets[(offsets >= q1 - 1.5 * iqr) & (offsets <= q3 + 1.5 * iqr)]
        if len(robust) < 18:
            return None

        offset_px = float(np.median(robust))
        self.offset_history.append(offset_px)
        median_offset = float(np.median(np.array(self.offset_history, dtype=np.float32)))

        max_step = max(3.0, w * 0.006)
        delta = float(np.clip(median_offset - self.offset_px, -max_step, max_step))
        if self.source == "fallback" and len(self.offset_history) >= 3:
            self.offset_px = median_offset
        else:
            self.offset_px += delta

        shifted_a = frame_a + normal * self.offset_px
        shifted_b = frame_b + normal * self.offset_px
        line_ref = (
            tuple(float(v) for v in self.calibration.to_reference(tuple(shifted_a), frame.shape)),
            tuple(float(v) for v in self.calibration.to_reference(tuple(shifted_b), frame.shape)),
        )

        count_score = min(1.0, len(robust) / 650.0)
        spread_score = max(0.0, 1.0 - min(1.0, float(np.std(robust)) / 42.0))
        history_score = min(1.0, len(self.offset_history) / self.offset_history.maxlen)
        confidence = max(0.0, min(0.9, 0.45 * count_score + 0.35 * spread_score + 0.20 * history_score))
        return line_ref, confidence

    def _detect(self, frame: np.ndarray) -> tuple[tuple[tuple[float, float], tuple[float, float]], float] | None:
        h, w = frame.shape[:2]
        roi = self.calibration.roi_mask(frame.shape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        white = cv2.inRange(hsv, np.array([0, 0, 145], dtype=np.uint8), np.array([179, 95, 255], dtype=np.uint8))
        white = cv2.bitwise_and(white, roi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)
        white = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
        edges = cv2.Canny(white, 50, 150)

        min_len = max(24, int(w * 0.025))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=25, minLineLength=min_len, maxLineGap=35)
        if lines is None:
            return None

        fallback_frame = tuple(self.calibration.from_reference(p, frame.shape) for p in self.calibration.target_line)
        expected_angle = line_angle_deg(fallback_frame)
        points: list[tuple[float, float]] = []
        total_len = 0.0
        for item in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in item]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len < min_len:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if angle_delta_deg(angle, expected_angle) > 28.0:
                continue
            mid = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            if roi[int(np.clip(mid[1], 0, h - 1)), int(np.clip(mid[0], 0, w - 1))] == 0:
                continue
            points.extend([(x1, y1), (x2, y2)])
            total_len += seg_len

        if len(points) < 6:
            return self._detect_from_fallback_band(frame, white)

        pts = np.array(points, dtype=np.float32)
        fit = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        vx, vy, x0, y0 = [float(v) for v in fit]
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            return None
        vx, vy = vx / norm, vy / norm
        distances = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        inliers = pts[distances <= 14.0]
        if len(inliers) >= 6:
            fit = cv2.fitLine(inliers, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
            vx, vy, x0, y0 = [float(v) for v in fit]
            norm = math.hypot(vx, vy)
            vx, vy = vx / norm, vy / norm

        line_frame = line_from_fit_to_rect(vx, vy, x0, y0, w, h)
        if line_frame is None:
            return None

        fitted_angle = line_angle_deg(line_frame)
        if angle_delta_deg(fitted_angle, expected_angle) > 30.0:
            return None

        line_ref = tuple(self.calibration.to_reference(p, frame.shape).tolist() for p in line_frame)
        inlier_ratio = float(len(inliers) / max(len(pts), 1)) if len(points) >= 6 else 0.0
        length_score = min(1.0, total_len / max(220.0, w * 0.18))
        count_score = min(1.0, len(points) / 18.0)
        confidence = max(0.0, min(1.0, 0.45 * length_score + 0.35 * inlier_ratio + 0.20 * count_score))
        return (tuple(tuple(float(v) for v in p) for p in line_ref), confidence)

    def _detect_from_fallback_band(
        self,
        frame: np.ndarray,
        white_mask: np.ndarray,
    ) -> tuple[tuple[tuple[float, float], tuple[float, float]], float] | None:
        h, w = frame.shape[:2]
        edges = cv2.Canny(white_mask, 40, 130)
        ys, xs = np.where(edges > 0)
        if len(xs) < 12:
            return None

        fallback_frame = tuple(self.calibration.from_reference(p, frame.shape) for p in self.calibration.target_line)
        line_a = np.array(fallback_frame[0], dtype=np.float32)
        line_b = np.array(fallback_frame[1], dtype=np.float32)
        line_vec = line_b - line_a
        line_len = float(np.linalg.norm(line_vec))
        if line_len <= 1e-6:
            return None
        unit = line_vec / line_len

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        distances = np.abs((pts[:, 0] - line_a[0]) * unit[1] - (pts[:, 1] - line_a[1]) * unit[0])
        band = max(45.0, min(95.0, w * 0.07))
        pts = pts[distances <= band]
        if len(pts) < 24:
            return None

        fit = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        vx, vy, x0, y0 = [float(v) for v in fit]
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            return None
        vx, vy = vx / norm, vy / norm
        residuals = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        inliers = pts[residuals <= 16.0]
        if len(inliers) >= 12:
            fit = cv2.fitLine(inliers, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
            vx, vy, x0, y0 = [float(v) for v in fit]
            norm = math.hypot(vx, vy)
            vx, vy = vx / norm, vy / norm

        line_frame = line_from_fit_to_rect(vx, vy, x0, y0, w, h)
        if line_frame is None:
            return None

        expected_angle = line_angle_deg(fallback_frame)
        fitted_angle = line_angle_deg(line_frame)
        angle_score = max(0.0, 1.0 - angle_delta_deg(fitted_angle, expected_angle) / 35.0)
        if angle_score <= 0.05:
            return None

        line_ref = tuple(self.calibration.to_reference(p, frame.shape).tolist() for p in line_frame)
        inlier_ratio = float(len(inliers) / max(len(pts), 1))
        count_score = min(1.0, len(pts) / 900.0)
        confidence = max(0.0, min(0.82, 0.35 * angle_score + 0.35 * inlier_ratio + 0.30 * count_score))
        return (tuple(tuple(float(v) for v in p) for p in line_ref), confidence)


@dataclass
class TrackState:
    track_id: int
    label: str
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    speed_kmh: float = 0.0
    ttc_s: float | None = None
    predicted_hit_time_s: float | None = None
    distance_m: float = 0.0
    signed_distance_m: float = 0.0
    front_point: tuple[float, float] = (0.0, 0.0)
    last_seen_frame: int = 0

    def update(
        self,
        timestamp_s: float,
        signed_distance_m: float,
        front_point: tuple[float, float],
        frame_id: int,
        window: int,
        min_approach_speed_mps: float,
    ) -> None:
        self.signed_distance_m = signed_distance_m
        self.distance_m = max(0.0, signed_distance_m)
        self.front_point = front_point
        self.last_seen_frame = frame_id
        self.samples.append((timestamp_s, signed_distance_m))
        while len(self.samples) > window:
            self.samples.popleft()
        if len(self.samples) < 5:
            self.speed_kmh = 0.0
            self.ttc_s = None
            self.predicted_hit_time_s = None
            return

        times = np.array([s[0] for s in self.samples], dtype=np.float32)
        distances = np.array([s[1] for s in self.samples], dtype=np.float32)
        if float(times[-1] - times[0]) < 0.12:
            return

        slope, _ = np.polyfit(times - times[0], distances, 1)
        speed_mps = abs(float(slope))
        approach_speed_mps = -float(slope)
        self.speed_kmh = speed_mps * 3.6
        if signed_distance_m <= 0.02:
            self.ttc_s = 0.0
            self.predicted_hit_time_s = timestamp_s
        elif approach_speed_mps >= min_approach_speed_mps:
            self.ttc_s = signed_distance_m / approach_speed_mps
            self.predicted_hit_time_s = timestamp_s + self.ttc_s
        else:
            self.ttc_s = None
            self.predicted_hit_time_s = None


def bottom_center_xyxy(box: np.ndarray) -> tuple[float, float]:
    x1, _y1, x2, y2 = box.tolist()
    return (x1 + x2) / 2.0, y2


def estimate_front_contact_point(
    box: np.ndarray,
    calibration: Calibration,
    frame_shape: tuple[int, int, int],
    target_line: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    x1, _y1, x2, y2 = box.tolist()
    samples = [(x1 + (x2 - x1) * i / 8.0, y2) for i in range(9)]
    return min(
        samples,
        key=lambda p: calibration.signed_distance_from_target(p, frame_shape, target_line),
    )


def choose_model_path() -> str:
    local = MODEL_DIR / "yolo11n.pt"
    if local.exists():
        return str(local)
    return "yolo11n.pt"


def yolo_label_to_project_label(yolo_name: str, override: str, video_path: str) -> str:
    if override != "Auto":
        return override
    name = yolo_name.lower()
    parts = {p.lower() for p in Path(video_path).parts}
    if name in {"car", "bus", "truck"}:
        return "car"
    if name == "motorcycle":
        return "e-bike"
    if name == "bicycle":
        if "bike" in parts:
            return "bicycle/e-bike"
        return "bicycle"
    return "vehicle"


def color_for_label(label: str) -> tuple[int, int, int]:
    if label == "car":
        return 40, 210, 255
    if label == "e-bike":
        return 255, 120, 40
    if label in {"bicycle", "bicycle/e-bike"}:
        return 80, 255, 120
    return 230, 230, 230


def draw_text_box(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 10, y)
    x2 = min(frame.shape[1] - 1, x + tw + 8)
    cv2.rectangle(frame, (x, y - th - baseline - 8), (x2, y + baseline + 4), (20, 20, 20), -1)
    cv2.putText(frame, text, (x + 4, y - 3), font, scale, color, thickness, cv2.LINE_AA)


class YoloSpeedTtcApp:
    def __init__(self, root: Tk) -> None:
        if cv2 is None:
            messagebox.showerror("OpenCV 导入失败", f"无法导入 cv2：\n{CV2_IMPORT_ERROR}")
            raise SystemExit(1)
        if YOLO is None:
            messagebox.showerror("Ultralytics 导入失败", f"无法导入 ultralytics：\n{YOLO_IMPORT_ERROR}")
            raise SystemExit(1)

        self.root = root
        self.root.title("YOLO Vehicle Speed & TTC")
        self.root.geometry("1180x780")

        self.calibration = Calibration.load(CONFIG_PATH)
        self.line_estimator = TargetLineEstimator(self.calibration)
        self.model: YOLO | None = None
        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.csv_file = None
        self.csv_writer: csv.writer | None = None

        self.video_path = ""
        self.fps = 25.0
        self.frame_id = 0
        self.playing = False
        self.photo: ImageTk.PhotoImage | None = None
        self.track_states: dict[int, TrackState] = {}

        self.status_var = StringVar(value="正在加载 YOLO 模型...")
        self.class_var = StringVar(value="Auto")
        self.save_var = StringVar(value="0")
        self.dynamic_line_var = StringVar(value="1")
        self.lock_line_var = StringVar(value="1")
        self.conf_var = StringVar(value="0.25")
        self.imgsz_var = StringVar(value="640")
        self.device_var = StringVar(value="auto")

        self._build_ui()
        self.root.after(100, self.load_model)

    def _build_ui(self) -> None:
        toolbar = Frame(self.root)
        toolbar.pack(side=TOP, fill="x", padx=8, pady=8)

        self.open_button = Button(toolbar, text="Open Video", command=self.open_video, state=DISABLED)
        self.open_button.pack(side=LEFT, padx=(0, 6))
        self.play_button = Button(toolbar, text="Play", command=self.toggle_play, state=DISABLED)
        self.play_button.pack(side=LEFT, padx=6)
        self.reset_button = Button(toolbar, text="Reset", command=self.reset_video, state=DISABLED)
        self.reset_button.pack(side=LEFT, padx=6)

        Label(toolbar, text="Class").pack(side=LEFT, padx=(18, 4))
        class_box = ttk.Combobox(toolbar, textvariable=self.class_var, width=15, state="readonly")
        class_box["values"] = ("Auto", "car", "bicycle", "e-bike", "bicycle/e-bike", "vehicle")
        class_box.pack(side=LEFT)

        Label(toolbar, text="conf").pack(side=LEFT, padx=(14, 4))
        conf_box = ttk.Combobox(toolbar, textvariable=self.conf_var, width=6, state="readonly")
        conf_box["values"] = ("0.15", "0.20", "0.25", "0.30", "0.40")
        conf_box.pack(side=LEFT)

        Label(toolbar, text="imgsz").pack(side=LEFT, padx=(14, 4))
        imgsz_box = ttk.Combobox(toolbar, textvariable=self.imgsz_var, width=6, state="readonly")
        imgsz_box["values"] = ("480", "640", "800")
        imgsz_box.pack(side=LEFT)

        Label(toolbar, text="device").pack(side=LEFT, padx=(14, 4))
        device_box = ttk.Combobox(toolbar, textvariable=self.device_var, width=8, state="readonly")
        device_box["values"] = ("auto", "cpu", "0")
        device_box.pack(side=LEFT)

        Checkbutton(toolbar, text="Save video/csv", variable=self.save_var, onvalue="1", offvalue="0").pack(side=LEFT, padx=(16, 0))
        Checkbutton(toolbar, text="Dynamic line", variable=self.dynamic_line_var, onvalue="1", offvalue="0").pack(side=LEFT, padx=(16, 0))
        Checkbutton(toolbar, text="Lock line", variable=self.lock_line_var, onvalue="1", offvalue="0").pack(side=LEFT, padx=(8, 0))

        self.video_label = Label(self.root, bg="#111111")
        self.video_label.pack(side=TOP, fill=BOTH, expand=True, padx=8)

        status = Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(side=BOTTOM, fill="x", padx=8, pady=8)

    def load_model(self) -> None:
        model_path = choose_model_path()
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            messagebox.showerror(
                "模型加载失败",
                "YOLO 模型加载失败。\n\n"
                "如果是网络下载失败，请手动下载 yolo11n.pt 放到 models/ 目录。\n"
                "下载链接：https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt\n\n"
                f"错误信息：{exc}",
            )
            self.status_var.set("YOLO model load failed.")
            return
        self.open_button.config(state=NORMAL)
        self.status_var.set(f"YOLO 模型已加载：{model_path}。点击 Open Video 导入视频。")

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(VIDEO_DIR if VIDEO_DIR.exists() else ROOT_DIR),
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")),
        )
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> None:
        self.close_outputs()
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("打开失败", f"无法打开视频：\n{path}")
            return

        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_id = 0
        self.track_states.clear()
        self.line_estimator = TargetLineEstimator(self.calibration)
        self.playing = True
        self.play_button.config(state=NORMAL, text="Pause")
        self.reset_button.config(state=NORMAL)

        if self.save_var.get() == "1":
            self.open_outputs(path)

        self.status_var.set(f"已载入：{os.path.relpath(path, ROOT_DIR)} | fps={self.fps:.2f}")
        self.root.after(1, self.process_next_frame)

    def open_outputs(self, video_path: str) -> None:
        OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        csv_path = OUTPUT_CSV_DIR / f"{stem}_yolo_speed_ttc.csv"
        self.csv_file = csv_path.open("w", newline="", encoding="utf-8-sig")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "video", "frame", "time_s", "track_id", "label", "yolo_class",
            "x1", "y1", "x2", "y2", "front_x", "front_y",
            "distance_m", "signed_distance_m", "speed_kmh", "ttc_s",
            "predicted_hit_time_s", "target_line_source", "target_line_conf"
        ])

    def ensure_writer(self, frame: np.ndarray) -> None:
        if self.save_var.get() != "1" or self.writer is not None:
            return
        stem = Path(self.video_path).stem
        out_path = OUTPUT_VIDEO_DIR / f"{stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))

    def close_outputs(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def reset_video(self) -> None:
        if self.video_path:
            self.load_video(self.video_path)

    def toggle_play(self) -> None:
        if self.cap is None:
            return
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.root.after(1, self.process_next_frame)

    def process_next_frame(self) -> None:
        if not self.playing or self.cap is None or self.model is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.play_button.config(text="Play")
            self.close_outputs()
            self.status_var.set("视频播放完成。")
            return

        start = time.perf_counter()
        annotated = self.run_yolo_and_draw(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.ensure_writer(annotated)
        if self.writer is not None:
            self.writer.write(annotated)

        self.show_frame(annotated)
        active = sum(1 for state in self.track_states.values() if self.frame_id - state.last_seen_frame <= 2)
        self.status_var.set(
            f"{Path(self.video_path).name} | frame={self.frame_id} | targets={active} | infer+draw={elapsed_ms:.0f} ms"
        )
        self.frame_id += 1
        self.root.after(1, self.process_next_frame)

    def run_yolo_and_draw(self, frame: np.ndarray) -> np.ndarray:
        timestamp_s = self.frame_id / max(self.fps, 1e-6)
        conf = float(self.conf_var.get())
        imgsz = int(self.imgsz_var.get())
        device = None if self.device_var.get() == "auto" else self.device_var.get()

        result = self.model.track(
            source=frame,
            persist=self.frame_id > 0,
            tracker="bytetrack.yaml",
            classes=YOLO_CLASS_FILTER,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        output = frame.copy()
        boxes = result.boxes
        line_estimate = self.line_estimator.update(
            frame,
            enabled=self.dynamic_line_var.get() == "1",
            locked=self.lock_line_var.get() == "1",
        )
        self.calibration.draw(output, line_estimate.line, line_estimate.source, line_estimate.confidence)
        if boxes is None or len(boxes) == 0:
            return output

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy)) + 100000
        names = result.names

        for box, class_id, track_id in zip(xyxy, cls, ids):
            center = bottom_center_xyxy(box)
            if not self.calibration.in_roi(center, output.shape):
                continue
            yolo_name = names.get(int(class_id), str(class_id)) if hasattr(names, "get") else str(names[int(class_id)])
            label = yolo_label_to_project_label(yolo_name, self.class_var.get(), self.video_path)
            front_point = estimate_front_contact_point(box, self.calibration, output.shape, line_estimate.line)
            signed_distance_m = self.calibration.signed_distance_from_target(front_point, output.shape, line_estimate.line)

            if int(track_id) not in self.track_states:
                self.track_states[int(track_id)] = TrackState(int(track_id), label)
            state = self.track_states[int(track_id)]
            state.label = label
            state.update(
                timestamp_s,
                signed_distance_m,
                front_point,
                self.frame_id,
                self.calibration.speed_window,
                self.calibration.min_approach_speed_mps,
            )

            self.draw_detection(output, box, yolo_name, state)
            self.write_csv_row(box, yolo_name, state, timestamp_s, line_estimate)

        self.cleanup_tracks()
        return output

    def draw_detection(self, frame: np.ndarray, box: np.ndarray, yolo_name: str, state: TrackState) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        color = color_for_label(state.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cx, cy = bottom_center_xyxy(box)
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)
        fx, fy = state.front_point
        cv2.circle(frame, (int(fx), int(fy)), 5, (0, 255, 255), -1)

        speed = f"{state.speed_kmh:.1f} km/h" if state.speed_kmh >= 0.5 else "-- km/h"
        if state.ttc_s is None or not math.isfinite(state.ttc_s):
            ttc = "TTC --"
        elif state.ttc_s <= 0.05:
            ttc = "crossed"
        else:
            ttc = f"TTC {state.ttc_s:.1f}s"
        text = f"#{state.track_id} {state.label} {speed} {ttc}"
        draw_text_box(frame, text, x1, y1 - 8, color)
        cv2.putText(frame, yolo_name, (x1, min(frame.shape[0] - 8, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def write_csv_row(
        self,
        box: np.ndarray,
        yolo_name: str,
        state: TrackState,
        timestamp_s: float,
        line_estimate: TargetLineEstimate,
    ) -> None:
        if self.csv_writer is None:
            return
        x1, y1, x2, y2 = box.tolist()
        fx, fy = state.front_point
        self.csv_writer.writerow([
            Path(self.video_path).name,
            self.frame_id,
            f"{timestamp_s:.3f}",
            state.track_id,
            state.label,
            yolo_name,
            f"{x1:.1f}",
            f"{y1:.1f}",
            f"{x2:.1f}",
            f"{y2:.1f}",
            f"{fx:.1f}",
            f"{fy:.1f}",
            f"{state.distance_m:.3f}",
            f"{state.signed_distance_m:.3f}",
            f"{state.speed_kmh:.3f}",
            "" if state.ttc_s is None else f"{state.ttc_s:.3f}",
            "" if state.predicted_hit_time_s is None else f"{state.predicted_hit_time_s:.3f}",
            line_estimate.source,
            f"{line_estimate.confidence:.3f}",
        ])

    def cleanup_tracks(self) -> None:
        stale = [tid for tid, state in self.track_states.items() if self.frame_id - state.last_seen_frame > 90]
        for tid in stale:
            del self.track_states[tid]

    def show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(320, self.video_label.winfo_width())
        max_h = max(240, self.video_label.winfo_height())
        image.thumbnail((max_w, max_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo)

    def on_close(self) -> None:
        self.playing = False
        if self.cap is not None:
            self.cap.release()
        self.close_outputs()
        self.root.destroy()


def main() -> None:
    root = Tk()
    app = YoloSpeedTtcApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
