"""Turning the segmented trace into suspension numbers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import find_peaks

from .signal_proc import crossing_time

SETTLE_TOL_MM = 2.0
MAX_EVENT_S = 1.5
TOP_OUT_TOL_MM = 1.0


@dataclass
class ReleaseMetrics:
    """One compress-and-release cycle."""

    t_release_s: float
    depth_from_sag_mm: float
    depth_from_top_mm: float
    peak_rebound_velocity_mm_s: float
    time_to_sag_s: float | None
    t90_s: float | None
    overshoot_mm: float
    topped_out: bool
    settling_time_s: float | None
    oscillation_count: int
    damping_ratio: float | None
    damped_freq_hz: float | None
    natural_freq_hz: float | None
    damping_note: str | None
    complete: bool
    # "rebound" events feed the aggregate; a deliberate bottom-out is measured and
    # reported but kept out of it, because letting the fork back up under control is
    # not the same experiment as releasing it.
    role: str = "rebound"

    def as_dict(self) -> dict:
        return _round_floats(asdict(self))


def _round_floats(obj, places: int = 3):
    if isinstance(obj, dict):
        return {k: _round_floats(v, places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, places) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, places)
    return obj


def _alternating_extrema(
    residual: np.ndarray, start: int, stop: int, prominence: float
) -> list[int]:
    """Indices of the free-vibration turning points, starting at the release.

    The release itself is the first turning point: the rider holds the fork down and
    lets go, which is displacement at maximum with zero velocity — textbook initial
    conditions for a free response, so its amplitude is a legitimate first peak for
    the log-decrement fit.
    """
    segment = residual[start:stop]
    if len(segment) < 3:
        return [start]
    highs, _ = find_peaks(segment, prominence=prominence)
    lows, _ = find_peaks(-segment, prominence=prominence)
    idx = sorted(int(i) + start for i in np.concatenate([highs, lows]))

    out = [start]
    for i in idx:
        if i <= start:
            continue
        prev = out[-1]
        # Consecutive turning points must alternate in sign; if two land on the same
        # side, keep only the more extreme one. The release point is never replaced.
        if np.sign(residual[i]) == np.sign(residual[prev]):
            if prev != start and abs(residual[i]) > abs(residual[prev]):
                out[-1] = i
            continue
        out.append(i)
    return out


def _damping(
    amplitudes: list[float], times: list[float]
) -> tuple[float | None, float | None, float | None, str | None]:
    """Damping ratio, damped and natural frequency from the log decrement."""
    # Filter amplitudes and times together so the indices stay in step.
    pairs = [(a, t) for a, t in zip(amplitudes, times, strict=True) if a > 1e-9]
    amps = [a for a, _ in pairs]
    times = [t for _, t in pairs]
    if len(amps) < 2:
        return None, None, None, "overdamped or no measurable overshoot"

    if len(amps) >= 3:
        # Successive same-side turning points are one full cycle apart.
        deltas = [
            math.log(amps[i] / amps[i + 2])
            for i in range(len(amps) - 2)
            if amps[i] > amps[i + 2]
        ]
        periods = [times[i + 2] - times[i] for i in range(len(times) - 2)]
        note = None
    else:
        # Only one overshoot: the pair is half a cycle apart, hence the factor of two.
        if amps[0] <= amps[1]:
            return None, None, None, "amplitude did not decay — not a clean free response"
        deltas = [2.0 * math.log(amps[0] / amps[1])]
        periods = [2.0 * (times[1] - times[0])]
        note = "single overshoot — half-cycle estimate, treat as approximate"

    if not deltas:
        return None, None, None, "amplitude did not decay — not a clean free response"

    delta = float(np.median(deltas))
    zeta = delta / math.sqrt(4.0 * math.pi**2 + delta**2)
    period = float(np.median(periods)) if periods else 0.0
    f_d = 1.0 / period if period > 0 else None
    f_n = f_d / math.sqrt(1.0 - zeta**2) if f_d and zeta < 1.0 else None
    return zeta, f_d, f_n, note


def analyse_release(
    separation_mm: np.ndarray,
    velocity_mm_s: np.ndarray,
    t_s: np.ndarray,
    release_idx: int,
    next_release_idx: int | None,
    sag_level: float,
    top_level: float,
    fps: float,
    settle_tol_mm: float = SETTLE_TOL_MM,
    max_event_s: float = MAX_EVENT_S,
) -> ReleaseMetrics:
    end = len(separation_mm)
    if next_release_idx is not None:
        end = min(end, next_release_idx)
    end = min(end, release_idx + int(round(max_event_s * fps)) + 1)
    end = max(end, release_idx + 2)

    window = slice(release_idx, end)
    depth_sag = sag_level - separation_mm[release_idx]
    depth_top = top_level - separation_mm[release_idx]

    t0 = float(t_s[release_idx])
    t_sag = crossing_time(t_s, separation_mm, sag_level, release_idx, rising=True)
    t90 = crossing_time(
        t_s,
        separation_mm,
        separation_mm[release_idx] + 0.9 * depth_sag,
        release_idx,
        rising=True,
    )

    peak_v = float(np.max(velocity_mm_s[window])) if end > release_idx else float("nan")
    seg_max = float(np.max(separation_mm[window]))
    overshoot = max(0.0, seg_max - sag_level)
    topped_out = seg_max >= (top_level - TOP_OUT_TOL_MM)

    residual = separation_mm - sag_level
    unsettled = np.flatnonzero(np.abs(residual[window]) > settle_tol_mm)
    if unsettled.size == 0:
        settling = 0.0
    elif (release_idx + int(unsettled[-1])) < end - 1:
        settling = float(t_s[release_idx + int(unsettled[-1]) + 1] - t0)
    else:
        settling = None  # still moving when the window closed

    prominence = max(0.3, 0.02 * abs(depth_sag))
    extrema = _alternating_extrema(residual, release_idx, end, prominence)
    amps = [abs(float(residual[i])) for i in extrema]
    times = [float(t_s[i]) for i in extrema]
    zeta, f_d, f_n, note = _damping(amps, times)

    return ReleaseMetrics(
        t_release_s=t0,
        depth_from_sag_mm=float(depth_sag),
        depth_from_top_mm=float(depth_top),
        peak_rebound_velocity_mm_s=peak_v,
        time_to_sag_s=None if t_sag is None else t_sag - t0,
        t90_s=None if t90 is None else t90 - t0,
        overshoot_mm=overshoot,
        topped_out=bool(topped_out),
        settling_time_s=settling,
        oscillation_count=max(0, len(extrema) - 1),
        damping_ratio=zeta,
        damped_freq_hz=f_d,
        natural_freq_hz=f_n,
        damping_note=note,
        # Recovered, not "crossed": a heavily damped fork creeps back up to sag
        # asymptotically and may never strictly pass it, but the event is still a
        # complete, measurable release.
        complete=bool(seg_max >= sag_level - settle_tol_mm),
    )


BOTTOM_OUT_DEPTH_RATIO = 1.6


def label_bottom_outs(
    releases: list[ReleaseMetrics], ratio: float = BOTTOM_OUT_DEPTH_RATIO
) -> list[ReleaseMetrics]:
    """Flag the calibration bottom-out so it does not dilute the rebound medians.

    Needs at least three complete events before it will call anything an outlier —
    with fewer, there is no reliable sense of what a normal push looked like.
    """
    complete = [r for r in releases if r.complete]
    if len(complete) < 3:
        return []
    median_depth = float(np.median([r.depth_from_sag_mm for r in complete]))
    flagged = []
    for release in complete:
        if release.depth_from_sag_mm > ratio * median_depth:
            release.role = "bottom-out"
            flagged.append(release)
    return flagged


def aggregate(releases: list[ReleaseMetrics]) -> dict:
    """Median across the complete events, so one botched push cannot skew the answer."""
    usable = [r for r in releases if r.complete and r.role == "rebound"]
    fields = [
        "depth_from_sag_mm",
        "peak_rebound_velocity_mm_s",
        "time_to_sag_s",
        "t90_s",
        "overshoot_mm",
        "settling_time_s",
        "damping_ratio",
        "damped_freq_hz",
        "natural_freq_hz",
    ]
    out: dict = {"events_used": len(usable), "events_found": len(releases)}
    for name in fields:
        values = [
            getattr(r, name)
            for r in usable
            if getattr(r, name) is not None and not math.isnan(float(getattr(r, name)))
        ]
        out[f"median_{name}"] = round(float(np.median(values)), 3) if values else None
    out["topped_out_events"] = sum(1 for r in usable if r.topped_out)
    return out
