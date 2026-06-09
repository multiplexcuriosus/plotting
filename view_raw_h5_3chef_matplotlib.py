#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


STEP_S_DEFAULT = 0.033


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path", help="Path to raw_events.h5")
    ap.add_argument(
        "--shifts",
        type=float,
        nargs=3,
        required=True,
        metavar=("A", "B", "C"),
        help="Three cumulative history windows in seconds, e.g. --shifts 0.05 0.25 1.5",
    )
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--step", type=float, default=STEP_S_DEFAULT, help="Time navigation step in seconds")
    ap.add_argument("--contrast", type=float, default=4.0)
    ap.add_argument("--event-step", type=float, default=1.0, help="Per-event accumulation magnitude")
    ap.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional initial display time in seconds relative to first event timestamp",
    )
    ap.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Optional prefix used when pressing 's' to save the current figure",
    )
    return ap.parse_args()


class EventHistoryViewer:
    def __init__(self, h5_path, shifts_s, width, height, dt_s, contrast, event_step, start_time_s=None, save_prefix=None):
        self.h5_path = Path(h5_path)
        self.width = int(width)
        self.height = int(height)
        self.shifts_s = [float(v) for v in shifts_s]
        self.shifts_us = np.array([int(round(v * 1e6)) for v in self.shifts_s], dtype=np.int64)
        self.dt_us = int(round(float(dt_s) * 1e6))
        self.contrast = float(contrast)
        self.event_step = float(event_step)
        self.save_prefix = save_prefix
        self.save_counter = 0

        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        if any(v <= 0 for v in self.shifts_s):
            raise ValueError("All shift values must be positive")
        if self.dt_us <= 0:
            raise ValueError("Navigation step must be positive")

        self._load_data()
        self._setup_time_range(start_time_s)
        self._setup_figure()
        self._update()

    def _load_data(self):
        with h5py.File(self.h5_path, "r") as f:
            required = ["/events/x", "/events/y", "/events/t_us"]
            missing = [k for k in required if k not in f]
            if missing:
                raise KeyError(f"Missing required datasets: {missing}")

            self.x = f["/events/x"][:].astype(np.int32)
            self.y = f["/events/y"][:].astype(np.int32)
            self.t_us = f["/events/t_us"][:].astype(np.int64)
            self.polarity = f["/events/type"][:].astype(np.int8) if "/events/type" in f else np.ones_like(self.x, dtype=np.int8)

        if len(self.x) == 0:
            raise ValueError("No events found in H5 file")
        if not (len(self.x) == len(self.y) == len(self.t_us) == len(self.polarity)):
            raise ValueError("Event datasets do not all have the same length")

        order = np.argsort(self.t_us, kind="stable")
        self.x = self.x[order]
        self.y = self.y[order]
        self.t_us = self.t_us[order]
        self.polarity = self.polarity[order]

        valid = (
            (self.x >= 0) & (self.x < self.width) &
            (self.y >= 0) & (self.y < self.height)
        )
        self.x = self.x[valid]
        self.y = self.y[valid]
        self.t_us = self.t_us[valid]
        self.polarity = self.polarity[valid]

        if len(self.x) == 0:
            raise ValueError("No valid in-bounds events found")

        self.t0_us = int(self.t_us[0])
        self.t_last_us = int(self.t_us[-1])

    def _setup_time_range(self, start_time_s):
        self.min_now_us = self.t0_us + int(np.max(self.shifts_us))
        self.max_now_us = self.t_last_us
        if self.min_now_us > self.max_now_us:
            raise ValueError(
                "Largest history window exceeds total duration of the event recording"
            )

        if start_time_s is None:
            self.now_us = self.min_now_us
        else:
            requested = self.t0_us + int(round(start_time_s * 1e6))
            self.now_us = int(np.clip(requested, self.min_now_us, self.max_now_us))

    @staticmethod
    def _render_gray_baseline_frame(x, y, polarity, width, height, contrast=4.0, step=1.0):
        frame = np.full((height, width), 128.0, dtype=np.float32)
        if x.size == 0:
            return frame.astype(np.uint8)

        acc = np.zeros((height, width), dtype=np.float32)
        pos = polarity == 1
        neg = ~pos
        if np.any(pos):
            np.add.at(acc, (y[pos], x[pos]), +step)
        if np.any(neg):
            np.add.at(acc, (y[neg], x[neg]), -step)

        m = float(np.max(np.abs(acc)))
        if m > 0:
            acc /= m

        frame = 128.0 + acc * (contrast * 127.0)
        np.clip(frame, 0, 255, out=frame)
        return frame.astype(np.uint8)

    def _window_slice(self, start_us, end_us):
        left = np.searchsorted(self.t_us, start_us, side="left")
        right = np.searchsorted(self.t_us, end_us, side="right")
        return slice(left, right)

    def _build_frames(self):
        gray_frames = []
        for shift_us in self.shifts_us:
            start_us = self.now_us - int(shift_us)
            sl = self._window_slice(start_us, self.now_us)
            gray = self._render_gray_baseline_frame(
                self.x[sl],
                self.y[sl],
                self.polarity[sl],
                self.width,
                self.height,
                contrast=self.contrast,
                step=self.event_step,
            )
            gray_frames.append(gray)

        history_frame = np.stack(gray_frames, axis=-1)
        return history_frame, gray_frames

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(10.5, 8.8), constrained_layout=False)
        gs = GridSpec(
            2,
            3,
            figure=self.fig,
            height_ratios=[1.2, 1.0],
            hspace=0.12,
            wspace=0.035,
            left=0.045,
            right=0.985,
            top=0.95,
            bottom=0.07,
        )

        self.ax_main = self.fig.add_subplot(gs[0, :])
        self.ax_ch = [
            self.fig.add_subplot(gs[1, 0]),
            self.fig.add_subplot(gs[1, 1]),
            self.fig.add_subplot(gs[1, 2]),
        ]

        for ax in [self.ax_main] + self.ax_ch:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        self.ax_main.set_title("History-Aware-Event-Frame", fontsize=16, pad=8)

        dummy_main = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        dummy_gray = np.full((self.height, self.width), 128, dtype=np.uint8)

        self.im_main = self.ax_main.imshow(dummy_main, interpolation="nearest")
        self.im_ch = [ax.imshow(dummy_gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest") for ax in self.ax_ch]

        self.bottom_labels = []
        for ax, shift_s in zip(self.ax_ch, self.shifts_s):
            label = ax.set_xlabel(f"$\\Delta t={int(round(shift_s * 1000))}$ ms", fontsize=13, labelpad=8)
            self.bottom_labels.append(label)

        self.time_text = self.fig.text(0.5, 0.016, "", ha="center", va="bottom", fontsize=12)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update(self):
        main_img, gray_frames = self._build_frames()
        self.im_main.set_data(main_img)
        for im, frame in zip(self.im_ch, gray_frames):
            im.set_data(frame)

        rel_t_s = (self.now_us - self.t0_us) / 1e6
        self.time_text.set_text(f"t = {rel_t_s:.3f} s")
        self.fig.canvas.draw_idle()

    def _save_current_figure(self):
        prefix = self.save_prefix or self.h5_path.stem
        rel_ms = int(round((self.now_us - self.t0_us) / 1000.0))
        out = self.h5_path.parent / f"{prefix}_{rel_ms:07d}ms.png"
        self.fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[INFO] Saved {out}")

    def _on_key(self, event):
        key = event.key
        if key in ("q", "escape"):
            plt.close(self.fig)
            return
        if key == "s":
            self._save_current_figure()
            return
        if key == "home":
            self.now_us = self.min_now_us
            self._update()
            return
        if key == "end":
            self.now_us = self.max_now_us
            self._update()
            return

        increment = self.dt_us
        if key in ("shift+right", "shift+left"):
            increment *= 10

        if key in ("right", "shift+right"):
            self.now_us = min(self.max_now_us, self.now_us + increment)
            self._update()
        elif key in ("left", "shift+left"):
            self.now_us = max(self.min_now_us, self.now_us - increment)
            self._update()

    def show(self):
        plt.show()


def main():
    args = parse_args()
    viewer = EventHistoryViewer(
        h5_path=args.h5_path,
        shifts_s=args.shifts,
        width=args.width,
        height=args.height,
        dt_s=args.step,
        contrast=args.contrast,
        event_step=args.event_step,
        start_time_s=args.start_time,
        save_prefix=args.save_prefix,
    )
    viewer.show()


if __name__ == "__main__":
    main()
