from __future__ import annotations

import csv
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import BOTH, BOTTOM, DISABLED, LEFT, NORMAL, TOP, Button, Checkbutton, Frame, Label, StringVar, Tk, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

from plate_pipeline import LicensePlatePipeline, PlateDetection, draw_plate_detection


ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "videos" / "check_number"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_VIDEO_DIR = ROOT_DIR / "outputs" / "videos"
OUTPUT_CSV_DIR = ROOT_DIR / "outputs" / "csv"
DETECTOR_PATH = MODEL_DIR / "license_plate_det.pt"
RECOGNIZER_PATH = MODEL_DIR / "plate_recognizer_other.pt"


@dataclass
class PlateTrackState:
    track_id: int
    observations: deque[str] = field(default_factory=deque)
    stable_text: str = ""
    last_seen_frame: int = 0
    last_confidence: float = 0.0

    def update(self, text: str, confidence: float, frame_id: int, window: int) -> None:
        self.last_seen_frame = frame_id
        self.last_confidence = confidence
        if text and text != "NO_RECOGNIZER":
            self.observations.append(text)
        while len(self.observations) > window:
            self.observations.popleft()
        self.stable_text = vote_plate_text(list(self.observations))


def vote_plate_text(values: list[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    common_len = Counter(len(v) for v in values).most_common(1)[0][0]
    candidates = [v for v in values if len(v) == common_len]
    if not candidates:
        return Counter(values).most_common(1)[0][0]
    chars: list[str] = []
    for i in range(common_len):
        chars.append(Counter(v[i] for v in candidates).most_common(1)[0][0])
    return "".join(chars)


class LicensePlateApp:
    def __init__(self, root: Tk) -> None:
        print("[GUI] Initializing LicensePlateApp...", flush=True)
        if cv2 is None:
            messagebox.showerror("OpenCV import failed", f"Cannot import cv2:\n{CV2_IMPORT_ERROR}")
            raise SystemExit(1)

        self.root = root
        self.root.title("License Plate Recognition")
        self.root.geometry("1180x780")
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))

        self.pipeline: LicensePlatePipeline | None = None
        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.csv_file = None
        self.csv_writer: csv.writer | None = None

        self.video_path = ""
        self.fps = 25.0
        self.frame_id = 0
        self.playing = False
        self.photo: ImageTk.PhotoImage | None = None
        self.track_states: dict[int, PlateTrackState] = {}

        self.status_var = StringVar(value="Loading license plate models...")
        self.save_var = StringVar(value="0")
        self.conf_var = StringVar(value="0.25")
        self.imgsz_var = StringVar(value="640")
        self.device_var = StringVar(value="auto")
        self.interval_var = StringVar(value="2")
        self.vote_window_var = StringVar(value="15")

        print("[GUI] Building UI...", flush=True)
        self._build_ui()
        print("[GUI] UI built. Scheduling model load in 100ms...", flush=True)
        self.root.after(100, self.load_models)

    def _build_ui(self) -> None:
        toolbar = Frame(self.root)
        toolbar.pack(side=TOP, fill="x", padx=8, pady=8)

        self.open_button = Button(toolbar, text="Open Video", command=self.open_video, state=DISABLED)
        self.open_button.pack(side=LEFT, padx=(0, 6))
        self.play_button = Button(toolbar, text="Play", command=self.toggle_play, state=DISABLED)
        self.play_button.pack(side=LEFT, padx=6)
        self.reset_button = Button(toolbar, text="Reset", command=self.reset_video, state=DISABLED)
        self.reset_button.pack(side=LEFT, padx=6)

        Label(toolbar, text="conf").pack(side=LEFT, padx=(18, 4))
        conf_box = ttk.Combobox(toolbar, textvariable=self.conf_var, width=6, state="readonly")
        conf_box["values"] = ("0.15", "0.20", "0.25", "0.30", "0.40", "0.50")
        conf_box.pack(side=LEFT)

        Label(toolbar, text="imgsz").pack(side=LEFT, padx=(14, 4))
        imgsz_box = ttk.Combobox(toolbar, textvariable=self.imgsz_var, width=6, state="readonly")
        imgsz_box["values"] = ("480", "640", "800")
        imgsz_box.pack(side=LEFT)

        Label(toolbar, text="device").pack(side=LEFT, padx=(14, 4))
        device_box = ttk.Combobox(toolbar, textvariable=self.device_var, width=8, state="readonly")
        device_box["values"] = ("auto", "cpu", "0")
        device_box.pack(side=LEFT)

        Label(toolbar, text="interval").pack(side=LEFT, padx=(14, 4))
        interval_box = ttk.Combobox(toolbar, textvariable=self.interval_var, width=5, state="readonly")
        interval_box["values"] = ("1", "2", "3", "5")
        interval_box.pack(side=LEFT)

        Label(toolbar, text="vote").pack(side=LEFT, padx=(14, 4))
        vote_box = ttk.Combobox(toolbar, textvariable=self.vote_window_var, width=5, state="readonly")
        vote_box["values"] = ("5", "10", "15", "20", "30")
        vote_box.pack(side=LEFT)

        Checkbutton(toolbar, text="Save video/csv", variable=self.save_var, onvalue="1", offvalue="0").pack(side=LEFT, padx=(16, 0))

        self.video_label = Label(self.root, bg="#111111")
        self.video_label.pack(side=TOP, fill=BOTH, expand=True, padx=8)

        status = Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(side=BOTTOM, fill="x", padx=8, pady=8)

    def load_models(self) -> None:
        print("[GUI] load_models() starting...", flush=True)
        print(f"[GUI]   detector: {DETECTOR_PATH}  exists={DETECTOR_PATH.exists()}", flush=True)
        print(f"[GUI]   recognizer: {RECOGNIZER_PATH}  exists={RECOGNIZER_PATH.exists()}", flush=True)
        try:
            device = None if self.device_var.get() == "auto" else self.device_var.get()
            print(f"[GUI]   device={device!r}, creating pipeline...", flush=True)
            t0 = time.time()
            self.pipeline = LicensePlatePipeline(DETECTOR_PATH, RECOGNIZER_PATH, device=device)
            print(f"[GUI]   pipeline created in {time.time()-t0:.1f}s", flush=True)
        except FileNotFoundError as exc:
            print(f"[GUI] ERROR: FileNotFoundError: {exc}", flush=True)
            self.status_var.set("Missing license plate detector.")
            messagebox.showerror(
                "Missing model",
                "\n".join([
                    str(exc),
                    "",
                    "Train the detector first:",
                    f"{sys.executable} src/train_plate_detector.py "
                    "--data data/processed/ccpd_plate/yolo_pose/data.yaml",
                    "",
                    "The GUI expects the trained detector at:",
                    str(DETECTOR_PATH),
                ]),
            )
            return
        except Exception as exc:
            print(f"[GUI] ERROR: {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            self.status_var.set("License plate model load failed.")
            messagebox.showerror("Model load failed", str(exc))
            return

        rec_status = "recognizer loaded" if self.pipeline.recognizer.available else "recognizer missing: text will show NO_RECOGNIZER"
        self.open_button.config(state=NORMAL)
        self.status_var.set(f"Detector loaded: {DETECTOR_PATH.name}; {rec_status}.")
        print(f"[GUI] Models ready! {rec_status}", flush=True)

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
            messagebox.showerror("Open failed", f"Cannot open video:\n{path}")
            return

        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_id = 0
        self.track_states.clear()
        self.playing = True
        self.play_button.config(state=NORMAL, text="Pause")
        self.reset_button.config(state=NORMAL)

        if self.save_var.get() == "1":
            self.open_outputs(path)

        self.status_var.set(f"Loaded: {os.path.relpath(path, ROOT_DIR)} | fps={self.fps:.2f}")
        self.root.after(1, self.process_next_frame)

    def open_outputs(self, video_path: str) -> None:
        OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        csv_path = OUTPUT_CSV_DIR / f"{stem}_plate_recognition.csv"
        self.csv_file = csv_path.open("w", newline="", encoding="utf-8-sig")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "video", "frame", "time_s", "track_id", "x1", "y1", "x2", "y2",
            "plate_conf", "raw_text", "raw_text_conf", "stable_text",
        ])

    def ensure_writer(self, frame: np.ndarray) -> None:
        if self.save_var.get() != "1" or self.writer is not None:
            return
        OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(self.video_path).stem
        out_path = OUTPUT_VIDEO_DIR / f"{stem}_plate_annotated.mp4"
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
        if not self.playing or self.cap is None or self.pipeline is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.play_button.config(text="Play")
            self.close_outputs()
            self.status_var.set("Video finished.")
            return

        start = time.perf_counter()
        annotated = self.run_plate_and_draw(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.ensure_writer(annotated)
        if self.writer is not None:
            self.writer.write(annotated)
        self.show_frame(annotated)
        active = sum(1 for s in self.track_states.values() if self.frame_id - s.last_seen_frame <= 8)
        self.status_var.set(
            f"{Path(self.video_path).name} | frame={self.frame_id} | plates={active} | infer+draw={elapsed_ms:.0f} ms"
        )
        self.frame_id += 1
        self.root.after(1, self.process_next_frame)

    def run_plate_and_draw(self, frame: np.ndarray) -> np.ndarray:
        conf = float(self.conf_var.get())
        imgsz = int(self.imgsz_var.get())
        device = None if self.device_var.get() == "auto" else self.device_var.get()
        interval = max(1, int(self.interval_var.get()))
        vote_window = max(1, int(self.vote_window_var.get()))
        recognize = self.frame_id % interval == 0

        detections = self.pipeline.infer(frame, self.frame_id, conf, imgsz, device, recognize=recognize)
        output = frame.copy()
        timestamp_s = self.frame_id / max(self.fps, 1e-6)
        for det in detections:
            state = self.track_states.setdefault(det.track_id, PlateTrackState(det.track_id))
            if recognize:
                state.update(det.text, det.text_confidence, self.frame_id, vote_window)
            else:
                state.last_seen_frame = self.frame_id
            draw_plate_detection(output, det, state.stable_text)
            self.write_csv_row(det, state.stable_text, timestamp_s)
        self.cleanup_tracks()
        return output

    def write_csv_row(self, det: PlateDetection, stable_text: str, timestamp_s: float) -> None:
        if self.csv_writer is None:
            return
        x1, y1, x2, y2 = det.bbox
        self.csv_writer.writerow([
            Path(self.video_path).name,
            self.frame_id,
            f"{timestamp_s:.3f}",
            det.track_id,
            f"{x1:.1f}",
            f"{y1:.1f}",
            f"{x2:.1f}",
            f"{y2:.1f}",
            f"{det.confidence:.3f}",
            det.text,
            f"{det.text_confidence:.3f}",
            stable_text,
        ])

    def cleanup_tracks(self) -> None:
        stale = [tid for tid, state in self.track_states.items() if self.frame_id - state.last_seen_frame > 120]
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
    print("[GUI] main() starting...", flush=True)
    print(f"[GUI] Python: {sys.executable}", flush=True)
    print(f"[GUI] Working dir: {os.getcwd()}", flush=True)
    print(f"[GUI] ROOT_DIR: {ROOT_DIR}", flush=True)
    root = Tk()
    print("[GUI] Tk root created", flush=True)
    app = LicensePlateApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    print("[GUI] Entering mainloop...", flush=True)
    root.mainloop()


if __name__ == "__main__":
    main()
