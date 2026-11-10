from __future__ import annotations

import cv2
import numpy as np
import pytest

from sag_tracker.colors import get_preset
from sag_tracker.synth import Renderer, SynthConfig
from sag_tracker.tracking import MarkerTracker, TrackResult


@pytest.fixture
def renderer():
    return Renderer(SynthConfig(noise_sigma=0.0))


def _hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


@pytest.mark.parametrize("compression", [0.0, 32.0, 92.0, 160.0])
def test_centroids_land_on_the_rendered_markers(renderer, compression):
    # Fresh trackers per case: these are unrelated poses, not consecutive frames.
    hsv = _hsv(renderer.frame(0.0, compression, occluded=False))
    want_lower, want_upper = renderer.marker_positions(compression)

    got_upper = MarkerTracker(get_preset("pink")).update(hsv)
    got_lower = MarkerTracker(get_preset("green")).update(hsv)

    assert got_upper.ok and got_lower.ok
    assert got_upper.x == pytest.approx(want_upper[0], abs=0.3)
    assert got_upper.y == pytest.approx(want_upper[1], abs=0.3)
    assert got_lower.x == pytest.approx(want_lower[0], abs=0.3)
    assert got_lower.y == pytest.approx(want_lower[1], abs=0.3)


def test_proximity_gate_rejects_a_same_coloured_distractor(renderer):
    tracker = MarkerTracker(get_preset("pink"), max_jump=60.0)
    frame = renderer.frame(0.0, 0.0, occluded=False)
    _, want = renderer.marker_positions(0.0)
    tracker.update(_hsv(frame))  # lock on

    # A bigger pink blob elsewhere: without the gate its larger area would win.
    decoyed = frame.copy()
    cv2.circle(decoyed, (60, 700), 40, renderer.upper_bgr, -1)
    got = tracker.update(_hsv(decoyed))

    assert got.ok
    assert got.x == pytest.approx(want[0], abs=0.5)
    assert got.y == pytest.approx(want[1], abs=0.5)


def test_missing_marker_reports_not_ok_without_inventing_a_position(renderer):
    tracker = MarkerTracker(get_preset("pink"))
    tracker.update(_hsv(renderer.frame(0.0, 0.0, occluded=False)))
    _, last = renderer.marker_positions(0.0)

    got = tracker.update(_hsv(renderer.frame(0.0, 0.0, occluded=True)))

    assert not got.ok
    assert got.area == 0.0
    assert got.x == pytest.approx(last[0], abs=0.5)  # holds the last known spot


def test_gate_widens_so_the_track_recovers_after_an_occlusion(renderer):
    tracker = MarkerTracker(get_preset("pink"), max_jump=10.0)
    tracker.update(_hsv(renderer.frame(0.0, 0.0, occluded=False)))

    # 320 px of travel is far outside a single 10 px gate, but 40 missed frames widen
    # it to 410 px, which is the point.
    for _ in range(40):
        tracker.update(_hsv(renderer.frame(0.0, 0.0, occluded=True)))

    _, want = renderer.marker_positions(160.0)
    got = tracker.update(_hsv(renderer.frame(0.0, 160.0, occluded=False)))

    assert got.ok
    assert got.x == pytest.approx(want[0], abs=0.5)


def test_separation_is_invariant_to_rigid_camera_motion():
    """The core claim of the measurement model, checked on synthetic coordinates."""
    rng = np.random.default_rng(3)
    n = 200
    upper = np.stack([np.full(n, 300.0), np.linspace(200.0, 400.0, n)], axis=1)
    lower = np.stack([np.full(n, 160.0), np.full(n, 640.0)], axis=1)
    truth = np.hypot(*(upper - lower).T)

    theta = rng.uniform(-0.3, 0.3, n)
    shift = rng.uniform(-50, 50, (n, 2))
    rot = np.stack(
        [
            np.stack([np.cos(theta), -np.sin(theta)], axis=1),
            np.stack([np.sin(theta), np.cos(theta)], axis=1),
        ],
        axis=1,
    )
    moved_upper = np.einsum("nij,nj->ni", rot, upper) + shift
    moved_lower = np.einsum("nij,nj->ni", rot, lower) + shift

    track = TrackResult(
        frames=np.arange(n),
        t_s=np.arange(n) / 120.0,
        upper_x=moved_upper[:, 0],
        upper_y=moved_upper[:, 1],
        lower_x=moved_lower[:, 0],
        lower_y=moved_lower[:, 1],
        upper_ok=np.ones(n, bool),
        lower_ok=np.ones(n, bool),
    )

    assert np.allclose(track.separation_px, truth)
