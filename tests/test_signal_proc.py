from __future__ import annotations

import numpy as np
import pytest

from sag_tracker import signal_proc


def test_find_runs_marks_contiguous_blocks():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert signal_proc.find_runs(mask) == [(1, 3), (5, 6), (7, 10)]


def test_find_runs_handles_edges():
    assert signal_proc.find_runs(np.ones(4, dtype=bool)) == [(0, 4)]
    assert signal_proc.find_runs(np.zeros(4, dtype=bool)) == []
    assert signal_proc.find_runs(np.array([], dtype=bool)) == []


def test_velocity_matches_the_analytic_derivative():
    fps = 240.0
    t = np.arange(0, 2, 1 / fps)
    y = 30.0 * np.sin(2 * np.pi * 2.5 * t)
    expected = 30.0 * 2 * np.pi * 2.5 * np.cos(2 * np.pi * 2.5 * t)

    got = signal_proc.velocity(y, fps, window_s=0.08)

    # Ignore the edges, where the polynomial fit has nothing to lean on.
    edge = int(0.1 * fps)
    assert np.allclose(got[edge:-edge], expected[edge:-edge], rtol=0.01)


def test_cubic_fit_preserves_a_sharp_peak_that_a_quadratic_would_clip():
    """The reason POLYORDER is 3 — a quadratic fit loses several percent of the peak."""
    fps = 120.0
    t = np.arange(0, 1.5, 1 / fps)
    y = 60.0 * (1 - np.exp(-8 * t) * np.cos(2 * np.pi * 2.5 * t))
    truth_peak = np.gradient(y, 1 / fps).max()

    cubic = signal_proc.velocity(y, fps, window_s=0.08, polyorder=3).max()
    quadratic = signal_proc.velocity(y, fps, window_s=0.08, polyorder=2).max()

    assert cubic == pytest.approx(truth_peak, rel=0.01)
    assert quadratic < truth_peak * 0.97


def test_short_gaps_are_bridged_and_long_ones_are_flagged():
    fps = 100.0
    t = np.arange(0, 1.0, 1 / fps)
    y = np.linspace(100.0, 200.0, len(t))
    ok = np.ones(len(t), dtype=bool)
    ok[20:23] = False  # 3 frames — bridgeable
    ok[50:70] = False  # 20 frames — too long to trust

    filled = signal_proc.interpolate_gaps(y, ok, t, max_gap_frames=5)

    assert np.allclose(filled.values, y)  # linear data interpolates exactly
    assert filled.valid[20:23].all()
    assert not filled.valid[50:70].any()
    assert len(filled.warnings) == 1
    assert "0.20 s" in filled.warnings[0]


def test_leading_dropout_is_held_and_excluded():
    t = np.arange(0, 0.5, 0.01)
    y = np.full(len(t), 150.0)
    ok = np.ones(len(t), dtype=bool)
    ok[:5] = False

    filled = signal_proc.interpolate_gaps(y, ok, t, max_gap_frames=5)

    assert not filled.valid[:5].any()
    assert filled.values[0] == 150.0
    assert "first" in filled.warnings[0]


def test_nothing_tracked_is_an_error_not_a_guess():
    t = np.arange(0, 0.5, 0.01)
    with pytest.raises(ValueError, match="neither marker"):
        signal_proc.interpolate_gaps(np.zeros(len(t)), np.zeros(len(t), bool), t)


def test_crossing_time_interpolates_between_samples():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.array([0.0, 10.0, 20.0, 30.0])

    assert signal_proc.crossing_time(t, y, 15.0, start=0) == pytest.approx(0.15)
    assert signal_proc.crossing_time(t, y, 99.0, start=0) is None


def test_crossing_time_ignores_crossings_before_the_start_index():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.array([0.0, 20.0, 5.0, 20.0])

    # The 0->20 rise happens before index 2 and must not be reported.
    assert signal_proc.crossing_time(t, y, 10.0, start=2) == pytest.approx(0.2 + 0.1 / 3)
