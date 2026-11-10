"""Converting pixels to millimetres."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Scale:
    mm_per_px: float
    source: str
    detail: str = ""

    def to_mm(self, px: float) -> float:
        return px * self.mm_per_px


class CalibrationError(ValueError):
    pass


def from_stanchion(
    diameter_mm: float, pt_a: tuple[float, float], pt_b: tuple[float, float]
) -> Scale:
    """Scale from the stanchion's known diameter measured across it in the image.

    Sensitive to the camera not being square to the fork: a stanchion viewed
    off-axis is foreshortened differently from the fork's stroke, and the error does
    not cancel. Prefer :func:`from_travel` whenever the clip contains a bottom-out.
    """
    if diameter_mm <= 0:
        raise CalibrationError("stanchion diameter must be positive")
    width_px = math.dist(pt_a, pt_b)
    if width_px <= 0:
        raise CalibrationError("the two stanchion points are identical")
    return Scale(
        mm_per_px=diameter_mm / width_px,
        source="stanchion",
        detail=f"{diameter_mm:g} mm across {width_px:.1f} px",
    )


def from_travel(travel_mm: float, topped_px: float, bottomed_px: float) -> Scale:
    """Scale from full travel: the clip's full stroke spans a known number of mm.

    Robust to a camera that is not perfectly square, because the foreshortening
    applies to both the reference stroke and the measured stroke and cancels in the
    ratio.
    """
    if travel_mm <= 0:
        raise CalibrationError("travel must be positive")
    stroke_px = topped_px - bottomed_px
    if stroke_px <= 0:
        raise CalibrationError(
            "no compression stroke found between the topped-out and most-compressed "
            "frames — was the fork ever bottomed out in this clip?"
        )
    return Scale(
        mm_per_px=travel_mm / stroke_px,
        source="travel",
        detail=f"{travel_mm:g} mm across {stroke_px:.1f} px of stroke",
    )


def disagreement_pct(a: Scale, b: Scale) -> float:
    """Relative difference between two scales, as a percentage of their mean."""
    mean = 0.5 * (a.mm_per_px + b.mm_per_px)
    if mean == 0:
        return float("nan")
    return abs(a.mm_per_px - b.mm_per_px) / mean * 100.0
