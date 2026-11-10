from __future__ import annotations

import math

import pytest

from sag_tracker import calibration
from sag_tracker.synth import Renderer, SynthConfig


def test_stanchion_scale_is_diameter_over_measured_width():
    scale = calibration.from_stanchion(35.0, (100.0, 200.0), (170.0, 200.0))
    assert scale.mm_per_px == pytest.approx(0.5)
    assert scale.source == "stanchion"


def test_travel_scale_is_travel_over_the_stroke():
    scale = calibration.from_travel(160.0, topped_px=440.0, bottomed_px=120.0)
    assert scale.mm_per_px == pytest.approx(0.5)
    assert scale.source == "travel"


def test_travel_scale_refuses_a_clip_with_no_stroke():
    with pytest.raises(calibration.CalibrationError, match="bottomed out"):
        calibration.from_travel(160.0, topped_px=440.0, bottomed_px=440.0)


def test_negative_inputs_are_rejected():
    with pytest.raises(calibration.CalibrationError):
        calibration.from_stanchion(-35.0, (0.0, 0.0), (70.0, 0.0))
    with pytest.raises(calibration.CalibrationError, match="identical"):
        calibration.from_stanchion(35.0, (10.0, 10.0), (10.0, 10.0))


def test_disagreement_is_symmetric_and_relative():
    a = calibration.Scale(0.50, "travel")
    b = calibration.Scale(0.55, "stanchion")
    assert calibration.disagreement_pct(a, b) == pytest.approx(
        calibration.disagreement_pct(b, a)
    )
    assert calibration.disagreement_pct(a, b) == pytest.approx(100 * 0.05 / 0.525)


def test_both_routes_agree_on_the_synthetic_geometry():
    """Independent calibrations of the same scene must land in the same place."""
    cfg = SynthConfig()
    renderer = Renderer(cfg)
    pt_a, pt_b = renderer.stanchion_points()

    by_stanchion = calibration.from_stanchion(cfg.stanchion_mm, pt_a, pt_b)

    top = cfg.separation_top_mm * cfg.px_per_mm
    bottom = (cfg.separation_top_mm - cfg.travel_mm) * cfg.px_per_mm
    by_travel = calibration.from_travel(cfg.travel_mm, top, bottom)

    assert by_travel.mm_per_px == pytest.approx(1.0 / cfg.px_per_mm)
    # The stanchion points are whole pixels, as a person clicking them would be, so a
    # couple of percent is the best this route can do by construction.
    assert calibration.disagreement_pct(by_stanchion, by_travel) < 3.0


def test_to_mm_applies_the_scale():
    scale = calibration.from_travel(160.0, 440.0, 120.0)
    assert scale.to_mm(math.hypot(3, 4)) == pytest.approx(2.5)
