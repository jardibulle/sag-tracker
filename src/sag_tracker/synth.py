"""Synthetic fork video with exact ground truth.

Renders a stand-in for a fork — a tilted stanchion with a tape marker on each side of
the dust seal — driven by a displacement programme whose sag, travel, damping ratio
and natural frequency are all known exactly. That makes the whole pipeline testable
end to end without waiting on real footage, and gives a fixture that can reproduce
awkward conditions (camera shake, sensor noise, an occluded marker) on demand.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Chosen in HSV so the rendered markers land squarely inside the presets in colors.py.
UPPER_HSV = (165, 255, 255)  # pink
LOWER_HSV = (60, 255, 230)  # green


@dataclass
class SynthConfig:
    fps: float = 120.0
    width: int = 480
    height: int = 800
    px_per_mm: float = 2.0
    tilt_deg: float = 22.0  # head angle, so the stroke is not conveniently vertical

    separation_top_mm: float = 220.0
    sag_mm: float = 32.0
    travel_mm: float = 160.0
    push_depth_mm: float = 60.0
    stanchion_mm: float = 35.0

    zeta: float = 0.35
    natural_freq_hz: float = 2.5
    cycles: int = 3

    marker_radius_px: int = 12
    shake: bool = False
    shake_px: float = 8.0
    shake_deg: float = 2.0
    noise_sigma: float = 4.0
    occlusions_s: list[tuple[float, float]] = field(default_factory=list)
    seed: int = 0


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class Programme:
    """The compression-vs-time schedule, in mm below the topped-out position."""

    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        self.segments: list[tuple[float, str, tuple]] = []
        self.release_times: list[float] = []
        self._build()

    def _add(self, duration: float, kind: str, args: tuple) -> None:
        self.segments.append((duration, kind, args))

    def _build(self) -> None:
        c = self.cfg
        sag, push, travel = c.sag_mm, c.push_depth_mm, c.travel_mm

        self._add(1.5, "hold", (0.0,))  # unweighted, topped out
        self._add(0.5, "ramp", (0.0, sag))  # rider gets on
        self._add(1.2, "hold", (sag,))  # settled at sag

        t = 3.2
        for _ in range(c.cycles):
            self._add(0.45, "ramp", (sag, sag + push))
            self._add(0.35, "hold", (sag + push,))
            self.release_times.append(t + 0.80)
            self._add(1.40, "free", (sag, push))
            self._add(0.50, "hold", (sag,))
            t += 2.70

        self._add(0.50, "ramp", (sag, travel))  # bottom it out for calibration
        self._add(0.70, "hold", (travel,))
        self._add(0.50, "ramp", (travel, sag))
        self._add(0.50, "hold", (sag,))

        self._add(0.50, "ramp", (sag, 0.0))  # rider steps off
        self._add(1.50, "hold", (0.0,))

    @property
    def duration_s(self) -> float:
        return sum(d for d, _, _ in self.segments)

    def compression_at(self, t: float) -> float:
        c = self.cfg
        for duration, kind, args in self.segments:
            if t > duration:
                t -= duration
                continue
            if kind == "hold":
                return args[0]
            if kind == "ramp":
                a, b = args
                return a + (b - a) * _smoothstep(t / duration)
            if kind == "free":
                base, amplitude = args
                return base + amplitude * _free_response(t, c.zeta, c.natural_freq_hz)
        return self.segments[-1][2][0]


def _free_response(tau: float, zeta: float, f_n: float) -> float:
    """Unit free vibration released from rest at unit displacement."""
    omega_n = 2.0 * math.pi * f_n
    omega_d = omega_n * math.sqrt(1.0 - zeta**2)
    return math.exp(-zeta * omega_n * tau) * (
        math.cos(omega_d * tau) + (zeta / math.sqrt(1.0 - zeta**2)) * math.sin(omega_d * tau)
    )


def _shake(cfg: SynthConfig, t: float) -> tuple[float, float, float]:
    """Smooth handheld wobble: (dx, dy, radians). Zero at t=0 so frame 0 stays clean."""
    if not cfg.shake:
        return 0.0, 0.0, 0.0
    dx = cfg.shake_px * (math.sin(2 * math.pi * 0.5 * t) + 0.4 * math.sin(2 * math.pi * 1.7 * t))
    dy = cfg.shake_px * (
        0.7 * math.sin(2 * math.pi * 0.9 * t) + 0.3 * math.sin(2 * math.pi * 2.3 * t)
    )
    theta = math.radians(cfg.shake_deg) * math.sin(2 * math.pi * 0.7 * t)
    return dx, dy, theta


# OpenCV draws at sub-pixel precision when coordinates are passed as fixed-point with
# this many fractional bits. Without it the markers snap to whole pixels and the
# fixture carries up to half a pixel of its own error into every assertion.
_SHIFT = 4


def _fixed(point: np.ndarray) -> tuple[int, int]:
    scale = 1 << _SHIFT
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def _hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    patch = np.array([[list(hsv)]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


class Renderer:
    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        tilt = math.radians(cfg.tilt_deg)
        # Unit vector pointing up the fork (image y grows downward).
        self.axis = np.array([math.sin(tilt), -math.cos(tilt)])
        self.across = np.array([math.cos(tilt), math.sin(tilt)])
        self.base = np.array([cfg.width * 0.34, cfg.height * 0.80])
        self.centre = np.array([cfg.width / 2.0, cfg.height / 2.0])
        self.upper_bgr = _hsv_to_bgr(UPPER_HSV)
        self.lower_bgr = _hsv_to_bgr(LOWER_HSV)
        self.rng = np.random.default_rng(cfg.seed)
        self._background = self._make_background()

    def _make_background(self) -> np.ndarray:
        cfg = self.cfg
        img = np.full((cfg.height, cfg.width, 3), 96, dtype=np.uint8)
        # Desaturated clutter, so the colour masks have to actually discriminate.
        rng = np.random.default_rng(cfg.seed + 1)
        for _ in range(24):
            centre = (int(rng.integers(0, cfg.width)), int(rng.integers(0, cfg.height)))
            shade = int(rng.integers(60, 150))
            cv2.circle(img, centre, int(rng.integers(10, 60)), (shade, shade, shade), -1)
        return img

    def _transform(self, point: np.ndarray, t: float) -> np.ndarray:
        dx, dy, theta = _shake(self.cfg, t)
        rel = point - self.centre
        rot = np.array(
            [
                rel[0] * math.cos(theta) - rel[1] * math.sin(theta),
                rel[0] * math.sin(theta) + rel[1] * math.cos(theta),
            ]
        )
        return rot + self.centre + np.array([dx, dy])

    def marker_positions(self, compression_mm: float) -> tuple[np.ndarray, np.ndarray]:
        """Lower marker is anchored to the ground; the crown drops toward it."""
        cfg = self.cfg
        separation_px = (cfg.separation_top_mm - compression_mm) * cfg.px_per_mm
        return self.base, self.base + self.axis * separation_px

    def stanchion_points(self, t: float = 0.0) -> tuple[tuple[int, int], tuple[int, int]]:
        """The two stanchion edges at mid-height — what a user would click to calibrate.

        Rounded to whole pixels on purpose: a person clicking a frame cannot do better.
        """
        cfg = self.cfg
        half = 0.5 * cfg.stanchion_mm * cfg.px_per_mm
        mid = self.base + self.axis * (cfg.separation_top_mm * cfg.px_per_mm * 0.6)
        a = self._transform(mid - self.across * half, t)
        b = self._transform(mid + self.across * half, t)
        return (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))

    def frame(self, t: float, compression_mm: float, occluded: bool) -> np.ndarray:
        cfg = self.cfg
        img = self._background.copy()
        lower, upper = self.marker_positions(compression_mm)

        def place(point: np.ndarray) -> tuple[int, int]:
            return _fixed(self._transform(point, t))

        width = int(round(cfg.stanchion_mm * cfg.px_per_mm))
        crown = upper + self.axis * 40
        cv2.line(img, place(lower), place(crown), (40, 40, 44), width, cv2.LINE_AA, _SHIFT)
        leg_bottom = lower - self.axis * 150
        cv2.line(
            img, place(leg_bottom), place(lower), (70, 72, 78), width + 8, cv2.LINE_AA, _SHIFT
        )

        radius = cfg.marker_radius_px << _SHIFT
        cv2.circle(img, place(lower), radius, self.lower_bgr, -1, cv2.LINE_AA, _SHIFT)
        if not occluded:
            cv2.circle(img, place(upper), radius, self.upper_bgr, -1, cv2.LINE_AA, _SHIFT)
        else:
            cv2.circle(
                img,
                place(upper),
                (cfg.marker_radius_px + 6) << _SHIFT,
                (90, 90, 95),
                -1,
                cv2.LINE_AA,
                _SHIFT,
            )

        if cfg.noise_sigma > 0:
            noise = self.rng.normal(0.0, cfg.noise_sigma, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img


def render(out_path: str | Path, cfg: SynthConfig | None = None) -> dict:
    """Write the video and its ground truth. Returns the ground-truth dict."""
    cfg = cfg or SynthConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    programme = Programme(cfg)
    renderer = Renderer(cfg)
    n_frames = int(round(programme.duration_s * cfg.fps))

    writer, out_path = _open_writer(out_path, cfg)
    compressions: list[float] = []
    try:
        for i in range(n_frames):
            t = i / cfg.fps
            compression = programme.compression_at(t)
            compressions.append(compression)
            occluded = any(a <= t < b for a, b in cfg.occlusions_s)
            writer.write(renderer.frame(t, compression, occluded))
    finally:
        writer.release()

    truth = _ground_truth(cfg, programme, renderer, compressions)
    truth_path = out_path.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    truth["video_path"] = str(out_path)
    truth["truth_path"] = str(truth_path)
    return truth


def _open_writer(out_path: Path, cfg: SynthConfig):
    size = (cfg.width, cfg.height)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, size)
    if writer.isOpened():
        return writer, out_path
    # Fall back to a container that every OpenCV build can write.
    fallback = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), cfg.fps, size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a video writer for mp4v or MJPG")
    return writer, fallback


def _ground_truth(
    cfg: SynthConfig, programme: Programme, renderer: Renderer, compressions: list[float]
) -> dict:
    arr = np.asarray(compressions, dtype=float)
    # Peak rebound speed as it actually exists in the rendered frames, sampled at the
    # same rate the analyser will see. Central differences, so no phase shift.
    separation = (cfg.separation_top_mm - arr)
    velocity = np.gradient(separation, 1.0 / cfg.fps)

    per_release: list[dict] = []
    for t_rel in programme.release_times:
        start = int(round(t_rel * cfg.fps))
        stop = min(len(velocity), start + int(round(1.2 * cfg.fps)))
        per_release.append(
            {
                "t_release_s": round(t_rel, 4),
                "peak_rebound_velocity_mm_s": round(float(np.max(velocity[start:stop])), 3),
            }
        )

    pt_a, pt_b = renderer.stanchion_points(0.0)
    omega_n = 2 * math.pi * cfg.natural_freq_hz
    return {
        "config": asdict(cfg),
        "mm_per_px": round(1.0 / cfg.px_per_mm, 6),
        "duration_s": round(programme.duration_s, 4),
        "frames": len(arr),
        "sag_mm": cfg.sag_mm,
        "travel_mm": cfg.travel_mm,
        "push_depth_mm": cfg.push_depth_mm,
        "sag_pct": round(cfg.sag_mm / cfg.travel_mm * 100.0, 3),
        "zeta": cfg.zeta,
        "natural_freq_hz": cfg.natural_freq_hz,
        "damped_freq_hz": round(
            omega_n * math.sqrt(1 - cfg.zeta**2) / (2 * math.pi), 4
        ),
        "separation_top_mm": cfg.separation_top_mm,
        "stanchion_mm": cfg.stanchion_mm,
        "stanchion_pt_a": list(pt_a),
        "stanchion_pt_b": list(pt_b),
        "releases": per_release,
        "compression_mm": [round(v, 4) for v in arr.tolist()],
    }
