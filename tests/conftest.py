"""Shared fixtures.

Rendering and analysing a clip takes a few seconds, so the session-scoped fixtures
below build each variant once and let the tests share it.
"""

from __future__ import annotations

import pytest

from sag_tracker.analysis import AnalysisConfig, run
from sag_tracker.synth import SynthConfig, render


def _make(tmp_path, name: str, **kwargs):
    cfg = SynthConfig(fps=120.0, **kwargs)
    truth = render(tmp_path / f"{name}.mp4", cfg)
    result = run(AnalysisConfig(video=truth["video_path"], travel_mm=cfg.travel_mm))
    return truth, result


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """A clean tripod-shot clip: the reference case."""
    return _make(tmp_path_factory.mktemp("clean"), "clean")


@pytest.fixture(scope="session")
def shaky_clip(tmp_path_factory):
    """The same programme filmed handheld — camera wobble must not change the answers."""
    return _make(tmp_path_factory.mktemp("shaky"), "shaky", shake=True)


@pytest.fixture(scope="session")
def occluded_clip(tmp_path_factory):
    """The upper marker is hidden for 0.3 s, far longer than the bridging window."""
    return _make(tmp_path_factory.mktemp("occ"), "occ", occlusions_s=[(5.0, 5.3)])
