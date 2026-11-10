"""Video input: metadata, frame rate sanity, and frame iteration."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# A rebound stroke lasts roughly 150-400 ms. Below this frame rate there are too few
# samples across it for the velocity numbers to mean much.
MIN_USEFUL_FPS = 60.0


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    fps_source: str

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class Frame:
    index: int
    t_s: float
    image: np.ndarray


class VideoSource:
    """Reads a clip start-to-finish, optionally trimmed to a time window.

    Frames are read and discarded rather than seeked: keyframe seeking is unreliable
    across codecs and these clips are short, so accuracy beats speed here.
    """

    def __init__(
        self,
        path: str | Path,
        fps_override: float | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise VideoError(f"video not found: {self.path}")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise VideoError(f"could not open video: {self.path}")
        try:
            container_fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        if fps_override is not None:
            if fps_override <= 0:
                raise VideoError("--fps must be positive")
            fps, fps_source = float(fps_override), "override"
        elif container_fps and math.isfinite(container_fps) and container_fps > 0:
            fps, fps_source = container_fps, "container"
        else:
            raise VideoError(
                f"{self.path.name} declares no usable frame rate; pass --fps explicitly"
            )

        self.info = VideoInfo(
            path=self.path,
            fps=fps,
            frame_count=max(frame_count, 0),
            width=width,
            height=height,
            fps_source=fps_source,
        )
        self.start_s = start_s
        self.end_s = end_s
        if start_s is not None and end_s is not None and end_s <= start_s:
            raise VideoError("--end must be greater than --start")

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.info.fps < MIN_USEFUL_FPS:
            out.append(
                f"frame rate is {self.info.fps:g} fps — rebound velocity is only coarsely "
                f"resolved below {MIN_USEFUL_FPS:g} fps; re-shoot at 120 or 240 fps for "
                f"trustworthy rebound numbers (sag is unaffected)"
            )
        if self.info.fps_source == "override":
            out.append(f"frame rate taken from --fps {self.info.fps:g}, not the file")
        return out

    def frames(self) -> Iterator[Frame]:
        """Yield frames inside the trim window, timestamped from the start of the file."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise VideoError(f"could not open video: {self.path}")
        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                t_s = index / self.info.fps
                if self.end_s is not None and t_s > self.end_s:
                    break
                if self.start_s is None or t_s >= self.start_s:
                    yield Frame(index=index, t_s=t_s, image=image)
                index += 1
        finally:
            cap.release()
