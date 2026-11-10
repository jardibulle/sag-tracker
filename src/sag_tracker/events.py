"""Carving the trace into still plateaus and compress/release events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

# Defaults tuned for the documented filming protocol; all overridable from the CLI.
V_STILL_MM_S = 15.0  # below this the rider counts as holding still
T_STILL_S = 0.30  # a plateau has to last this long to be believed
LEVEL_TOL_MM = 4.0  # plateaus within this of each other are the same position
RELEASE_PROMINENCE_MM = 8.0
RELEASE_MIN_SEP_S = 0.40
RELEASE_MIN_DEPTH_MM = 8.0


@dataclass
class Plateau:
    start: int
    stop: int  # exclusive
    t_start: float
    t_end: float
    level: float
    label: str = "unassigned"

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "t_start_s": round(self.t_start, 3),
            "t_end_s": round(self.t_end, 3),
            "duration_s": round(self.duration_s, 3),
            "separation_mm": round(self.level, 2),
        }


@dataclass
class LevelCluster:
    level: float
    total_duration_s: float
    plateaus: list[Plateau] = field(default_factory=list)


def find_plateaus(
    separation: np.ndarray,
    velocity: np.ndarray,
    t_s: np.ndarray,
    valid: np.ndarray,
    v_still: float = V_STILL_MM_S,
    t_still: float = T_STILL_S,
) -> list[Plateau]:
    """Stretches where the suspension is not moving and the tracking is trustworthy.

    Unit-neutral: this runs before calibration, so ``separation`` and ``v_still`` are
    in pixels, and the resulting levels are converted to mm afterwards.
    """
    from .signal_proc import find_runs

    still = (np.abs(velocity) < v_still) & np.asarray(valid, dtype=bool)
    out: list[Plateau] = []
    for start, stop in find_runs(still):
        t_start, t_end = float(t_s[start]), float(t_s[stop - 1])
        if (t_end - t_start) < t_still:
            continue
        out.append(
            Plateau(
                start=start,
                stop=stop,
                t_start=t_start,
                t_end=t_end,
                level=float(np.median(separation[start:stop])),
            )
        )
    return out


def cluster_levels(plateaus: list[Plateau], tol: float = LEVEL_TOL_MM) -> list[LevelCluster]:
    """Group plateaus that sit at the same fork position.

    Single-linkage over the sorted levels: a gap wider than ``tol`` starts a new
    cluster. Cluster level is duration-weighted, so a long steady hold outvotes a
    brief one. Units follow whatever :func:`find_plateaus` was given.
    """
    if not plateaus:
        return []
    ordered = sorted(plateaus, key=lambda p: p.level)
    groups: list[list[Plateau]] = [[ordered[0]]]
    for plateau in ordered[1:]:
        if plateau.level - groups[-1][-1].level <= tol:
            groups[-1].append(plateau)
        else:
            groups.append([plateau])

    clusters: list[LevelCluster] = []
    for group in groups:
        total = sum(p.duration_s for p in group)
        level = (
            sum(p.level * p.duration_s for p in group) / total
            if total > 0
            else float(np.mean([p.level for p in group]))
        )
        clusters.append(LevelCluster(level=level, total_duration_s=total, plateaus=group))
    return clusters


def assign_reference_levels(
    clusters: list[LevelCluster],
) -> tuple[LevelCluster | None, LevelCluster | None]:
    """Pick the topped-out and sag references.

    The most extended cluster is the unweighted fork. Sag is then the remaining
    cluster with the most *total* time in it — the rider settles back to sag after
    every release, so sag accumulates several plateaus while the bottom-out hold only
    ever gets one. That makes the choice robust without asking anyone to nominate a
    window by hand.
    """
    if not clusters:
        return None, None
    top = max(clusters, key=lambda c: c.level)
    rest = [c for c in clusters if c is not top]
    if not rest:
        return top, None
    sag = max(rest, key=lambda c: c.total_duration_s)
    return top, sag


def find_release_events(
    separation_mm: np.ndarray,
    t_s: np.ndarray,
    fps: float,
    sag_level: float,
    prominence_mm: float = RELEASE_PROMINENCE_MM,
    min_sep_s: float = RELEASE_MIN_SEP_S,
    min_depth_mm: float = RELEASE_MIN_DEPTH_MM,
) -> list[int]:
    """Frame indices at which the fork starts extending after being pushed down.

    Peaks are found on the inverted trace, then nudged forward to the end of any hold
    at the bottom: the release proper starts when the rider lets go, which is the
    moment the response becomes a free vibration from rest.
    """
    separation_mm = np.asarray(separation_mm, dtype=float)
    minima, _ = find_peaks(
        -separation_mm,
        prominence=prominence_mm,
        distance=max(1, int(round(min_sep_s * fps))),
    )

    releases: list[int] = []
    for idx in minima:
        depth = sag_level - separation_mm[idx]
        if depth < min_depth_mm:
            continue
        releases.append(_release_start(separation_mm, int(idx), depth))
    return releases


def _release_start(separation_mm: np.ndarray, idx: int, depth: float) -> int:
    """Walk to the far end of the flat bottom, where the rider actually lets go."""
    tol = max(0.5, 0.02 * depth)
    floor = separation_mm[idx] + tol
    j = idx
    while j + 1 < len(separation_mm) and separation_mm[j + 1] <= floor:
        j += 1
    return j
