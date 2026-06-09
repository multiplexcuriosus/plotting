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
        help="Three cumulative event-history windows in seconds, e.g. --shifts 0.05 0.25 1.5",
    )
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--step", type=float, default=STEP_S_DEFAULT, help="Time navigation step in seconds")
    ap.add_argument("--contrast", type=float, default=4.0, help="Color contrast multiplier")
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
    ap.add_argument(
        "--save-format",
        type=str,
        default="png",
        choices=("png", "pdf", "svg"),
        help="Format used when pressing 's'",
    )
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


class EventHistoryViewer:
    def __init__(
        self,
        h5_path,
        shifts_s,
        width,
        height,
        dt_s,
        contrast,
        event_step,
        start_time_s=None,
        save_prefix=None,
        save_format="png",
        dpi=300,
    ):
        self.h5_path = Path(h5_path)
        self.width = int(width)
        self.height = int(height)
        self.shifts_s = [float(v) for v in shifts_s]
        self.shifts_us = np.array([int(round(v * 1e6)) for v in self.shifts_s], dtype=np.int64)
        self.dt_us = int(round(float(dt_s) * 1e6))
        self.contrast = float(contrast)
        self.event_step = float(event_step)
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.dpi = int(dpi)

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
            raise ValueError("Largest event-history window exceeds recording duration")

        if start_time_s is None:
            self.now_us = self.min_now_us
        else:
            requested = self.t0_us + int(round(start_time_s * 1e6))
            self.now_us = int(np.clip(requested, self.min_now_us, self.max_now_us))

    @staticmethod
    def _render_redblue_baseline_frame(x, y, polarity, width, height, contrast=4.0, step=1.0):
        """
        Neutral baseline: gray (128,128,128).
        Positive events: pixels become bluer.
        Negative events: pixels become redder.
        """
        acc = np.zeros((height, width), dtype=np.float32)
        if x.size > 0:
            pos = polarity == 1
            neg = ~pos
            if np.any(pos):
                np.add.at(acc, (y[pos], x[pos]), +step)
            if np.any(neg):
                np.add.at(acc, (y[neg], x[neg]), -step)

        m = float(np.max(np.abs(acc)))
        if m > 0.0:
            acc /= m

        amp = np.clip(np.abs(acc) * (contrast * 127.0), 0.0, 127.0)
        pos_mag = np.where(acc > 0, amp, 0.0)
        neg_mag = np.where(acc < 0, amp, 0.0)

        r = np.full((height, width), 128.0, dtype=np.float32)
        g = np.full((height, width), 128.0, dtype=np.float32)
        b = np.full((height, width), 128.0, dtype=np.float32)

        # Positive -> bluer, Negative -> redder.
        r = r + neg_mag - pos_mag
        g = g - (pos_mag + neg_mag)
        b = b + pos_mag - neg_mag

        rgb = np.stack([r, g, b], axis=-1)
        np.clip(rgb, 0, 255, out=rgb)
        return rgb.astype(np.uint8)

    def _window_slice(self, start_us, end_us):
        left = np.searchsorted(self.t_us, start_us, side="left")
        right = np.searchsorted(self.t_us, end_us, side="right")
        return slice(left, right)

    def _build_frames(self):
        frames = []
        for shift_us in self.shifts_us:
            start_us = self.now_us - int(shift_us)
            sl = self._window_slice(start_us, self.now_us)
            frame = self._render_redblue_baseline_frame(
                self.x[sl],
                self.y[sl],
                self.polarity[sl],
                self.width,
                self.height,
                contrast=self.contrast,
                step=self.event_step,
            )
            frames.append(frame)
        return frames

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(12.0, 4.6), constrained_layout=False)
        gs = GridSpec(
            1,
            3,
            figure=self.fig,
            wspace=0.03,
            left=0.02,
            right=0.995,
            top=0.965,
            bottom=0.12,
        )

        self.axes = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        dummy = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        self.images = [ax.imshow(dummy, interpolation="nearest") for ax in self.axes]

        labels = [
            f"(a) {int(round(self.shifts_s[0] * 1000))} ms event history",
            f"(b) {int(round(self.shifts_s[1] * 1000))} ms event history",
            f"(c) {int(round(self.shifts_s[2] * 1000))} ms event history",
        ]
        for ax, label in zip(self.axes, labels):
            ax.set_xlabel(label, fontsize=11, labelpad=8)

        self.time_text = self.fig.text(0.5, 0.03, "", ha="center", va="bottom", fontsize=10)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update(self):
        frames = self._build_frames()
        for im, frame in zip(self.images, frames):
            im.set_data(frame)

        rel_t_s = (self.now_us - self.t0_us) / 1e6
        self.time_text.set_text(f"t = {rel_t_s:.3f} s")
        self.fig.canvas.draw_idle()

    def _save_current_figure(self):
        prefix = self.save_prefix or self.h5_path.stem
        rel_ms = int(round((self.now_us - self.t0_us) / 1000.0))
        out = self.h5_path.parent / f"{prefix}_{rel_ms:07d}ms.{self.save_format}"
        self.fig.savefig(out, dpi=self.dpi, bbox_inches="tight", facecolor="white")
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
        save_format=args.save_format,
        dpi=args.dpi,
    )
    viewer.show()


if __name__ == "__main__":
    main()
