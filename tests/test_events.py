from __future__ import annotations

import numpy as np
import pytest

from sag_tracker import events, signal_proc

FPS = 120.0


def build_trace(levels: list[tuple[float, float]], ramp_s: float = 0.25):
    """Stitch a trace out of (hold_seconds, level) steps joined by smooth ramps."""
    values: list[float] = []
    for i, (duration, level) in enumerate(levels):
        if i:
            prev = levels[i - 1][1]
            n = int(ramp_s * FPS)
            u = np.linspace(0, 1, n, endpoint=False)
            values.extend(prev + (level - prev) * (3 * u**2 - 2 * u**3))
        values.extend([level] * int(duration * FPS))
    separation = np.asarray(values, dtype=float)
    t = np.arange(len(separation)) / FPS
    return t, separation, signal_proc.velocity(separation, FPS)


def test_plateaus_are_found_where_the_signal_holds_still():
    t, separation, velocity = build_trace([(1.5, 220.0), (1.2, 188.0), (1.0, 220.0)])
    valid = np.ones(len(t), dtype=bool)

    plateaus = events.find_plateaus(separation, velocity, t, valid)

    assert len(plateaus) == 3
    assert [round(p.level) for p in plateaus] == [220, 188, 220]
    assert all(p.duration_s >= events.T_STILL_S for p in plateaus)


def test_untrusted_frames_cannot_form_a_plateau():
    t, separation, velocity = build_trace([(1.5, 220.0), (1.2, 188.0)])
    valid = np.ones(len(t), dtype=bool)
    valid[: int(1.5 * FPS)] = False

    plateaus = events.find_plateaus(separation, velocity, t, valid)

    assert [round(p.level) for p in plateaus] == [188]


def test_clustering_merges_repeats_of_the_same_position():
    t, separation, velocity = build_trace(
        [(1.5, 220.0), (1.2, 188.0), (0.5, 128.0), (1.0, 188.5), (1.5, 220.0)]
    )
    plateaus = events.find_plateaus(separation, velocity, t, np.ones(len(t), bool))

    clusters = events.cluster_levels(plateaus)

    assert len(clusters) == 3
    assert sorted(round(c.level) for c in clusters) == [128, 188, 220]


def test_sag_is_the_level_the_rider_keeps_returning_to():
    """Sag wins on total time, which is why a single long bottom-out hold cannot steal it."""
    t, separation, velocity = build_trace(
        [
            (1.5, 220.0),  # unweighted
            (1.2, 188.0),  # sag
            (0.4, 128.0),  # pushed down
            (1.0, 188.0),  # back to sag
            (0.4, 128.0),
            (1.0, 188.0),  # sag again
            (1.4, 60.0),  # a long bottom-out hold
            (1.5, 220.0),
        ]
    )
    plateaus = events.find_plateaus(separation, velocity, t, np.ones(len(t), bool))

    top, sag = events.assign_reference_levels(events.cluster_levels(plateaus))

    assert top is not None and sag is not None
    assert round(top.level) == 220
    assert round(sag.level) == 188
    assert sag.total_duration_s > 2.0


def test_a_clip_with_only_one_level_yields_no_sag():
    t, separation, velocity = build_trace([(2.0, 220.0)])
    plateaus = events.find_plateaus(separation, velocity, t, np.ones(len(t), bool))

    top, sag = events.assign_reference_levels(events.cluster_levels(plateaus))

    assert top is not None
    assert sag is None


def test_assign_on_no_plateaus_returns_nothing():
    assert events.assign_reference_levels([]) == (None, None)


def test_releases_are_found_at_the_end_of_the_hold_not_its_middle():
    """The rider lets go when the hold ends — that is where the free response starts."""
    t, separation, _ = build_trace([(1.0, 188.0), (0.6, 128.0), (1.0, 188.0)])

    releases = events.find_release_events(separation, t, FPS, sag_level=188.0)

    assert len(releases) == 1
    hold_start = 1.0 + 0.25
    hold_end = hold_start + 0.6
    assert t[releases[0]] == pytest.approx(hold_end, abs=0.05)


def test_shallow_wobbles_are_not_releases():
    t, separation, _ = build_trace([(1.0, 188.0), (0.5, 185.0), (1.0, 188.0)])

    assert events.find_release_events(separation, t, FPS, sag_level=188.0) == []


def test_each_push_yields_exactly_one_release():
    t, separation, _ = build_trace(
        [(1.0, 188.0), (0.4, 128.0), (1.0, 188.0), (0.4, 128.0), (1.0, 188.0), (0.4, 128.0), (1.0, 188.0)]
    )

    assert len(events.find_release_events(separation, t, FPS, sag_level=188.0)) == 3
