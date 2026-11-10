"""Colour-blob tracking of the two tape markers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .colors import ColorRange

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


@dataclass
class MarkerSample:
    x: float
    y: float
    area: float
    ok: bool


class MarkerTracker:
    """Tracks one coloured marker across frames.

    The blob centroid is the mean of every pixel in the component, so it lands well
    inside a pixel even though the mask itself is binary — with a marker a few dozen
    pixels across that is worth roughly a tenth of a pixel of resolution.

    Candidates are gated to a radius around the previous hit so that same-coloured
    clutter elsewhere in the frame cannot steal the track. The gate grows while the
    marker is missing, so it recovers on its own after an occlusion instead of
    latching onto nothing forever.
    """

    def __init__(
        self,
        color: ColorRange,
        min_area: float = 40.0,
        max_jump: float = 60.0,
    ) -> None:
        self.color = color
        self.min_area = float(min_area)
        self.max_jump = float(max_jump)
        self._last: tuple[float, float] | None = None
        self._missed = 0

    def update(self, hsv: np.ndarray) -> MarkerSample:
        mask = self.color.mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        last = self._last
        gate = None
        if last is not None:
            # Expanding gate: one frame's worth of travel per frame since the last hit,
            # capped at the frame diagonal so it degrades to "search everywhere".
            diag = float(np.hypot(*hsv.shape[:2]))
            gate = min(self.max_jump * (1 + self._missed), diag)

        best_label = -1
        best_area = 0.0
        for label in range(1, count):  # label 0 is the background
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            if gate is not None and last is not None:
                cx, cy = centroids[label]
                if np.hypot(cx - last[0], cy - last[1]) > gate:
                    continue
            if area > best_area:
                best_area, best_label = area, label

        if best_label < 0:
            self._missed += 1
            x, y = self._last if self._last is not None else (float("nan"), float("nan"))
            return MarkerSample(x=x, y=y, area=0.0, ok=False)

        moments = cv2.moments((labels == best_label).astype(np.uint8), binaryImage=True)
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        self._last = (x, y)
        self._missed = 0
        return MarkerSample(x=x, y=y, area=best_area, ok=True)


@dataclass
class TrackResult:
    """Per-frame tracking output for both markers."""

    frames: np.ndarray  # int, frame index in the source file
    t_s: np.ndarray
    upper_x: np.ndarray
    upper_y: np.ndarray
    lower_x: np.ndarray
    lower_y: np.ndarray
    upper_ok: np.ndarray  # bool
    lower_ok: np.ndarray  # bool

    @property
    def separation_px(self) -> np.ndarray:
        """Marker-to-marker distance.

        This scalar is what the whole measurement rests on: because both markers sit
        on the fork, the distance between them is unchanged by the camera translating
        or rolling, so handheld shake cancels out without any stabilisation.
        """
        return np.hypot(self.upper_x - self.lower_x, self.upper_y - self.lower_y)

    @property
    def ok(self) -> np.ndarray:
        return self.upper_ok & self.lower_ok


def track_video(
    frames,
    upper: ColorRange,
    lower: ColorRange,
    min_area: float = 40.0,
    max_jump: float = 60.0,
) -> TrackResult:
    """Run both trackers over an iterable of :class:`~sag_tracker.video.Frame`."""
    up = MarkerTracker(upper, min_area=min_area, max_jump=max_jump)
    lo = MarkerTracker(lower, min_area=min_area, max_jump=max_jump)

    idx: list[int] = []
    ts: list[float] = []
    ux: list[float] = []
    uy: list[float] = []
    lx: list[float] = []
    ly: list[float] = []
    uok: list[bool] = []
    lok: list[bool] = []

    for frame in frames:
        hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
        a = up.update(hsv)
        b = lo.update(hsv)
        idx.append(frame.index)
        ts.append(frame.t_s)
        ux.append(a.x)
        uy.append(a.y)
        lx.append(b.x)
        ly.append(b.y)
        uok.append(a.ok)
        lok.append(b.ok)

    return TrackResult(
        frames=np.asarray(idx, dtype=int),
        t_s=np.asarray(ts, dtype=float),
        upper_x=np.asarray(ux, dtype=float),
        upper_y=np.asarray(uy, dtype=float),
        lower_x=np.asarray(lx, dtype=float),
        lower_y=np.asarray(ly, dtype=float),
        upper_ok=np.asarray(uok, dtype=bool),
        lower_ok=np.asarray(lok, dtype=bool),
    )
