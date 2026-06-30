from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import BOTH, BOTTOM, DISABLED, END, LEFT, NORMAL, RIGHT, TOP, Button, Frame, Label, StringVar, Tk, filedialog, messagebox, ttk

try:
    import cv2
except Exception as exc:  # pragma: no cover - shown in GUI at runtime
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

import numpy as np
from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = APP_DIR / "videos" / "predict_line"


@dataclass
class Calibration:
    """Map frame bottom-center points to distance from the crosswalk line."""

    reference_size: tuple[int, int] = (1200, 683)
    target_point: tuple[float, float] = (92.0, 516.0)
    far_point: tuple[float, float] = (1046.0, 434.0)
    axis_length_m: float = 12.10
    roi_polygon: tuple[tuple[float, float], ...] = (
        (0.0, 405.0),
        (1200.0, 382.0),
        (1200.0, 642.0),
        (0.0, 660.0),
    )

    def to_reference(self, point: tuple[float, float], frame_shape: tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        ref_w, ref_h = self.reference_size
        return np.array([point[0] * ref_w / w, point[1] * ref_h / h], dtype=np.float32)

    def from_reference(self, point: tuple[float, float], frame_shape: tuple[int, int, int]) -> tuple[int, int]:
        h, w = frame_shape[:2]
        ref_w, ref_h = self.reference_size
        return int(point[0] * w / ref_w), int(point[1] * h / ref_h)

    def roi_mask(self, frame_shape: tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        pts = np.array([self.from_reference(p, frame_shape) for p in self.roi_polygon], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def distance_from_target(self, point: tuple[float, float], frame_shape: tuple[int, int, int]) -> float:
        p = self.to_reference(point, frame_shape)
        a = np.array(self.target_point, dtype=np.float32)
        b = np.array(self.far_point, dtype=np.float32)
        axis = b - a
        axis_len_px = float(np.linalg.norm(axis))
        if axis_len_px < 1e-6:
            return 0.0
        unit = axis / axis_len_px
        projected_px = float(np.dot(p - a, unit))
        return max(0.0, projected_px / axis_len_px * self.axis_length_m)

    def draw_guides(self, frame: np.ndarray) -> None:
        target = self.from_reference(self.target_point, frame.shape)
        far = self.from_reference(self.far_point, frame.shape)
        cv2.line(frame, target, far, (255, 180, 30), 2)
        cv2.circle(frame, target, 7, (0, 0, 255), -1)
        cv2.putText(frame, "crosswalk / target", (target[0] + 8, max(target[1] - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def bottom_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, float(y2)


def merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        result: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i, box in enumerate(merged):
            if used[i]:
                continue
            x1, y1, x2, y2 = box
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                close = abs(((x1 + x2) / 2) - ((bx1 + bx2) / 2)) < max(x2 - x1, bx2 - bx1) * 0.7
                if iou((x1, y1, x2, y2), merged[j]) > 0.02 or close and abs(y2 - by2) < 60:
                    x1, y1, x2, y2 = min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)
                    used[j] = True
                    changed = True
            result.append((x1, y1, x2, y2))
        merged = result
    return merged


class MotionDetector:
    def __init__(self, calibration: Calibration) -> None:
        self.calibration = calibration
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=450, varThreshold=28, detectShadows=True)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
        self.roi: np.ndarray | None = None

    def reset(self) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=450, varThreshold=28, detectShadows=True)
        self.roi = None

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.roi is None or self.roi.shape != frame.shape[:2]:
            self.roi = self.calibration.roi_mask(frame.shape)
        fg = self.subtractor.apply(frame)
        fg = cv2.bitwise_and(fg, fg, mask=self.roi)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)
        fg = cv2.dilate(fg, self.kernel_close, iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        min_area = max(250, int(w * h * 0.00045))
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < min_area or bw < 18 or bh < 18:
                continue
            if bw > w * 0.75 or bh > h * 0.65:
                continue
            boxes.append((x, y, x + bw, y + bh))
        return merge_boxes(boxes)


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    label: str
    last_frame: int
    missed: int = 0
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    speed_kmh: float = 0.0
    ttc_s: float | None = None
    distance_m: float = 0.0

    def update_metrics(self) -> None:
        if len(self.history) < 5:
            self.speed_kmh = 0.0
            self.ttc_s = None
            return
        samples = list(self.history)
        t = np.array([s[0] for s in samples], dtype=np.float32)
        d = np.array([s[1] for s in samples], dtype=np.float32)
        if float(t[-1] - t[0]) < 0.15:
            return
        slope, _ = np.polyfit(t - t[0], d, 1)
        speed_mps = abs(float(slope))
        self.speed_kmh = speed_mps * 3.6
        approach_mps = -float(slope)
        if approach_mps > 0.15 and self.distance_m > 0.05:
            self.ttc_s = self.distance_m / approach_mps
        else:
            self.ttc_s = None


class SimpleTracker:
    def __init__(self, calibration: Calibration) -> None:
        self.calibration = calibration
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.max_missed = 12

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(
        self,
        boxes: list[tuple[int, int, int, int]],
        frame_id: int,
        timestamp_s: float,
        frame_shape: tuple[int, int, int],
        label: str,
    ) -> list[Track]:
        unmatched_tracks = set(self.tracks)
        unmatched_boxes = set(range(len(boxes)))
        pairs: list[tuple[float, int, int]] = []
        for tid, track in self.tracks.items():
            for idx, box in enumerate(boxes):
                score = iou(track.box, box)
                if score > 0:
                    pairs.append((score, tid, idx))
        pairs.sort(reverse=True)

        for score, tid, idx in pairs:
            if score < 0.08 or tid not in unmatched_tracks or idx not in unmatched_boxes:
                continue
            self._assign(self.tracks[tid], boxes[idx], frame_id, timestamp_s, frame_shape, label)
            unmatched_tracks.remove(tid)
            unmatched_boxes.remove(idx)

        for tid in list(unmatched_tracks):
            track = self.tracks[tid]
            track.missed += 1
            if track.missed > self.max_missed:
                del self.tracks[tid]

        for idx in unmatched_boxes:
            box = boxes[idx]
            track = Track(self.next_id, box, label, frame_id)
            self.next_id += 1
            self._assign(track, box, frame_id, timestamp_s, frame_shape, label)
            self.tracks[track.track_id] = track

        return sorted(self.tracks.values(), key=lambda item: item.track_id)

    def _assign(
        self,
        track: Track,
        box: tuple[int, int, int, int],
        frame_id: int,
        timestamp_s: float,
        frame_shape: tuple[int, int, int],
        label: str,
    ) -> None:
        point = bottom_center(box)
        distance_m = self.calibration.distance_from_target(point, frame_shape)
        track.box = box
        track.label = label
        track.last_frame = frame_id
        track.missed = 0
        track.distance_m = distance_m
        track.history.append((timestamp_s, distance_m))
        track.update_metrics()


def infer_label(video_path: str, override: str) -> str:
    if override != "Auto":
        return override
    parts = [p.lower() for p in Path(video_path).parts]
    if "car" in parts:
        return "car"
    if "bike" in parts:
        return "bicycle/e-bike"
    return "vehicle"


def draw_track(frame: np.ndarray, track: Track) -> None:
    x1, y1, x2, y2 = track.box
    color = (0, 210, 255) if track.ttc_s is None or track.ttc_s > 2.0 else (0, 60, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    speed = f"{track.speed_kmh:.1f} km/h" if track.speed_kmh > 0.5 else "-- km/h"
    ttc = f"TTC {track.ttc_s:.1f}s" if track.ttc_s is not None and math.isfinite(track.ttc_s) else "TTC --"
    text = f"#{track.track_id} {track.label} {speed} {ttc}"
    baseline = max(24, y1 - 8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, baseline - th - 8), (min(frame.shape[1] - 1, x1 + tw + 8), baseline + 5), (20, 20, 20), -1)
    cv2.putText(frame, text, (x1 + 4, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cx, cy = bottom_center(track.box)
    cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)


class SpeedTtcApp:
    def __init__(self, root: Tk) -> None:
        if cv2 is None:
            messagebox.showerror("OpenCV import failed", f"cv2 cannot be imported:\n{CV2_IMPORT_ERROR}")
            raise SystemExit(1)

        self.root = root
        self.root.title("Vehicle Speed and TTC Estimator")
        self.root.geometry("1120x760")
        self.calibration = Calibration()
        self.detector = MotionDetector(self.calibration)
        self.tracker = SimpleTracker(self.calibration)
        self.cap: cv2.VideoCapture | None = None
        self.video_path = ""
        self.playing = False
        self.frame_id = 0
        self.fps = 25.0
        self.last_frame_time = 0.0
        self.photo: ImageTk.PhotoImage | None = None

        self.status_var = StringVar(value="Open a video to start.")
        self.class_var = StringVar(value="Auto")
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = Frame(self.root)
        toolbar.pack(side=TOP, fill="x", padx=8, pady=8)
        Button(toolbar, text="Open Video", command=self.open_video).pack(side=LEFT, padx=(0, 6))
        self.play_button = Button(toolbar, text="Play", command=self.toggle_play, state=DISABLED)
        self.play_button.pack(side=LEFT, padx=6)
        Button(toolbar, text="Reset", command=self.reset_current).pack(side=LEFT, padx=6)
        Label(toolbar, text="Class:").pack(side=LEFT, padx=(18, 4))
        combo = ttk.Combobox(toolbar, textvariable=self.class_var, width=16, state="readonly")
        combo["values"] = ("Auto", "car", "bicycle", "e-bike", "bicycle/e-bike", "vehicle")
        combo.pack(side=LEFT)

        self.video_label = Label(self.root, bg="#111111")
        self.video_label.pack(side=TOP, fill=BOTH, expand=True, padx=8)
        status = Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(side=BOTTOM, fill="x", padx=8, pady=8)

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(DEFAULT_VIDEO_DIR if DEFAULT_VIDEO_DIR.exists() else APP_DIR),
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")),
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str) -> None:
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Open failed", f"Cannot open video:\n{path}")
            return
        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_id = 0
        self.playing = True
        self.detector.reset()
        self.tracker.reset()
        self.play_button.config(state=NORMAL, text="Pause")
        self.status_var.set(f"Loaded: {os.path.relpath(path, APP_DIR)} | fps={self.fps:.2f}")
        self._schedule_next(1)

    def reset_current(self) -> None:
        if not self.video_path:
            return
        self.load_video(self.video_path)

    def toggle_play(self) -> None:
        if self.cap is None:
            return
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._schedule_next(1)

    def _schedule_next(self, delay_ms: int) -> None:
        self.root.after(max(1, delay_ms), self.update_frame)

    def update_frame(self) -> None:
        if not self.playing or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.play_button.config(text="Play")
            self.status_var.set("Video finished.")
            return

        timestamp_s = self.frame_id / max(self.fps, 1e-6)
        label = infer_label(self.video_path, self.class_var.get())
        boxes = self.detector.detect(frame)
        tracks = self.tracker.update(boxes, self.frame_id, timestamp_s, frame.shape, label)

        self.calibration.draw_guides(frame)
        for track in tracks:
            if track.missed == 0:
                draw_track(frame, track)
        cv2.putText(frame, f"frame {self.frame_id}  fps {self.fps:.1f}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        self.show_frame(frame)
        active = sum(1 for t in tracks if t.missed == 0)
        self.status_var.set(f"{Path(self.video_path).name} | frame={self.frame_id} | active targets={active}")
        self.frame_id += 1

        delay = int(1000 / max(self.fps, 1.0))
        elapsed_ms = int((time.time() - self.last_frame_time) * 1000) if self.last_frame_time else 0
        self.last_frame_time = time.time()
        self._schedule_next(max(1, delay - elapsed_ms))

    def show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(320, self.video_label.winfo_width())
        max_h = max(240, self.video_label.winfo_height())
        image.thumbnail((max_w, max_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo)


def main() -> None:
    root = Tk()
    SpeedTtcApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
