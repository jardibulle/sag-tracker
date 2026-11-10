"""HSV colour ranges for the tape markers.

OpenCV packs HSV into uint8: H in [0, 179], S and V in [0, 255]. Hue is circular,
so pink and red straddle the 179/0 seam and need two sub-ranges instead of one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

HSVBound = tuple[int, int, int]


@dataclass(frozen=True)
class ColorRange:
    """One or more HSV boxes that together select a marker colour."""

    name: str
    bounds: tuple[tuple[HSVBound, HSVBound], ...] = field(default=())

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Binary mask (uint8, 0/255) of pixels falling inside any sub-range."""
        out = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.bounds:
            out |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return out


def _wrap(h_lo: int, h_hi: int, s_min: int, v_min: int) -> tuple[tuple[HSVBound, HSVBound], ...]:
    """Build sub-ranges for a hue window, splitting it if it crosses the 179/0 seam."""
    if h_lo <= h_hi:
        return (((h_lo, s_min, v_min), (h_hi, 255, 255)),)
    return (
        ((h_lo, s_min, v_min), (179, 255, 255)),
        ((0, s_min, v_min), (h_hi, 255, 255)),
    )


PRESETS: dict[str, ColorRange] = {
    # Hue windows are deliberately wide: tape under real light drifts a lot, and the
    # saturation/value floors do most of the rejection work against a dull background.
    "pink": ColorRange("pink", _wrap(150, 4, 80, 70)),
    "red": ColorRange("red", _wrap(170, 8, 100, 60)),
    "orange": ColorRange("orange", _wrap(8, 22, 120, 90)),
    "yellow": ColorRange("yellow", _wrap(22, 35, 90, 90)),
    "green": ColorRange("green", _wrap(38, 85, 70, 50)),
    "blue": ColorRange("blue", _wrap(95, 130, 90, 50)),
}


def get_preset(name: str) -> ColorRange:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown colour {name!r}; choose from {', '.join(sorted(PRESETS))}"
        ) from None


def _circular_hue_mean(hues: np.ndarray) -> float:
    """Mean hue that respects the 179/0 wrap-around."""
    angles = hues.astype(np.float64) * (2.0 * np.pi / 180.0)
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float(np.mod(mean, 2.0 * np.pi) * (180.0 / (2.0 * np.pi)))


def from_patch(
    hsv_patch: np.ndarray,
    name: str = "sampled",
    h_tol: int = 12,
    s_min_margin: int = 60,
    v_min_margin: int = 60,
) -> ColorRange:
    """Derive a range from a sampled ROI — adapts to whatever the light is doing.

    Uses the patch's circular-mean hue widened by ``h_tol``, and puts the saturation
    and value floors a fixed margin below the patch median so shading on a curved
    stanchion does not fall out of the mask.
    """
    flat = hsv_patch.reshape(-1, 3)
    if flat.size == 0:
        raise ValueError("cannot sample a colour from an empty patch")
    h = _circular_hue_mean(flat[:, 0])
    s_med = float(np.median(flat[:, 1]))
    v_med = float(np.median(flat[:, 2]))
    h_lo = int(round(np.mod(h - h_tol, 180)))
    h_hi = int(round(np.mod(h + h_tol, 180)))
    s_min = int(np.clip(s_med - s_min_margin, 40, 255))
    v_min = int(np.clip(v_med - v_min_margin, 30, 255))
    return ColorRange(name, _wrap(h_lo, h_hi, s_min, v_min))
