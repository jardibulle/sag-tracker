"""Rebound maths checked against closed-form responses, not against our own renderer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sag_tracker import metrics, signal_proc

SAG_LEVEL = 188.0
TOP_LEVEL = 220.0
FPS = 240.0


def free_response(t: np.ndarray, zeta: float, f_n: float) -> np.ndarray:
    """Unit displacement released from rest — the textbook underdamped solution."""
    w_n = 2 * math.pi * f_n
    w_d = w_n * math.sqrt(1 - zeta**2)
    return np.exp(-zeta * w_n * t) * (
        np.cos(w_d * t) + (zeta / math.sqrt(1 - zeta**2)) * np.sin(w_d * t)
    )


def make_release(zeta: float, f_n: float = 2.5, amplitude: float = 60.0, duration: float = 2.0):
    t = np.arange(0.0, duration, 1.0 / FPS)
    separation = SAG_LEVEL - amplitude * free_response(t, zeta, f_n)
    velocity = signal_proc.velocity(separation, FPS)
    return t, separation, velocity


def analyse(zeta: float, **kwargs) -> metrics.ReleaseMetrics:
    t, separation, velocity = make_release(zeta, **kwargs)
    return metrics.analyse_release(
        separation,
        velocity,
        t,
        release_idx=0,
        next_release_idx=None,
        sag_level=SAG_LEVEL,
        top_level=TOP_LEVEL,
        fps=FPS,
    )


@pytest.mark.parametrize("zeta", [0.15, 0.25, 0.35, 0.5])
def test_damping_ratio_recovered_from_a_known_response(zeta):
    result = analyse(zeta)
    assert result.damping_ratio == pytest.approx(zeta, abs=0.03)


@pytest.mark.parametrize("f_n", [1.8, 2.5, 3.4])
def test_natural_frequency_recovered(f_n):
    result = analyse(0.3, f_n=f_n)
    assert result.natural_freq_hz == pytest.approx(f_n, rel=0.05)
    assert result.damped_freq_hz == pytest.approx(f_n * math.sqrt(1 - 0.3**2), rel=0.05)


def test_depth_and_overshoot_match_the_analytic_amplitudes():
    zeta, amplitude = 0.35, 60.0
    result = analyse(zeta, amplitude=amplitude)

    assert result.depth_from_sag_mm == pytest.approx(amplitude, rel=0.01)
    assert result.depth_from_top_mm == pytest.approx(TOP_LEVEL - SAG_LEVEL + amplitude, rel=0.01)
    # First overshoot of an underdamped response: A * exp(-zeta*pi/sqrt(1-zeta^2)).
    expected = amplitude * math.exp(-zeta * math.pi / math.sqrt(1 - zeta**2))
    assert result.overshoot_mm == pytest.approx(expected, rel=0.05)


def test_lighter_damping_gives_more_overshoot_and_more_oscillations():
    light = analyse(0.15)
    heavy = analyse(0.5)

    assert light.overshoot_mm > heavy.overshoot_mm
    assert light.oscillation_count >= heavy.oscillation_count
    assert light.settling_time_s > heavy.settling_time_s


def test_an_overdamped_return_reports_no_damping_ratio_rather_than_a_number():
    t = np.arange(0.0, 2.0, 1.0 / FPS)
    separation = SAG_LEVEL - 60.0 * np.exp(-t / 0.25)  # monotone, never overshoots
    velocity = signal_proc.velocity(separation, FPS)

    result = metrics.analyse_release(
        separation, velocity, t, 0, None, SAG_LEVEL, TOP_LEVEL, FPS
    )

    assert result.damping_ratio is None
    assert result.damping_note == "overdamped or no measurable overshoot"
    assert result.overshoot_mm == pytest.approx(0.0, abs=0.5)
    assert result.complete


def test_topping_out_is_flagged():
    # Released from deep enough that the overshoot runs past full extension.
    result = analyse(0.05, amplitude=120.0)
    assert result.topped_out


def test_an_event_that_never_returns_to_sag_is_incomplete():
    t = np.arange(0.0, 0.3, 1.0 / FPS)
    separation = np.linspace(SAG_LEVEL - 60.0, SAG_LEVEL - 40.0, len(t))
    velocity = signal_proc.velocity(separation, FPS)

    result = metrics.analyse_release(
        separation, velocity, t, 0, None, SAG_LEVEL, TOP_LEVEL, FPS
    )

    assert not result.complete
    assert result.time_to_sag_s is None


def _stub(depth: float, peak: float = 500.0, complete: bool = True) -> metrics.ReleaseMetrics:
    return metrics.ReleaseMetrics(
        t_release_s=0.0,
        depth_from_sag_mm=depth,
        depth_from_top_mm=depth + 32,
        peak_rebound_velocity_mm_s=peak,
        time_to_sag_s=0.13,
        t90_s=0.12,
        overshoot_mm=18.0,
        topped_out=False,
        settling_time_s=0.5,
        oscillation_count=3,
        damping_ratio=0.35,
        damped_freq_hz=2.3,
        natural_freq_hz=2.5,
        damping_note=None,
        complete=complete,
    )


def test_a_deep_outlier_is_labelled_as_the_bottom_out():
    releases = [_stub(60), _stub(61), _stub(59), _stub(126, peak=380.0)]

    flagged = metrics.label_bottom_outs(releases)

    assert [r.role for r in releases] == ["rebound", "rebound", "rebound", "bottom-out"]
    assert len(flagged) == 1
    summary = metrics.aggregate(releases)
    assert summary["events_used"] == 3
    assert summary["events_found"] == 4
    assert summary["median_peak_rebound_velocity_mm_s"] == pytest.approx(500.0)


def test_bottom_out_labelling_needs_enough_events_to_judge():
    releases = [_stub(60), _stub(126)]
    assert metrics.label_bottom_outs(releases) == []
    assert all(r.role == "rebound" for r in releases)


def test_aggregate_ignores_incomplete_events():
    releases = [_stub(60, peak=500), _stub(60, peak=520), _stub(60, peak=9999, complete=False)]
    summary = metrics.aggregate(releases)
    assert summary["events_used"] == 2
    assert summary["median_peak_rebound_velocity_mm_s"] == pytest.approx(510.0)
