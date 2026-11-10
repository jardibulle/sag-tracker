"""Smoothing, differentiation and gap filling for the raw separation trace.

Named ``signal_proc`` rather than ``signal`` so it cannot shadow the stdlib module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) index ranges of contiguous ``True`` values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=True)]


@dataclass
class GapFill:
    values: np.ndarray
    valid: np.ndarray  # False where the value is a guess we do not trust
    interpolated: np.ndarray  # True where the value was filled in at all
    warnings: list[str] = field(default_factory=list)


def interpolate_gaps(
    values: np.ndarray,
    ok: np.ndarray,
    t_s: np.ndarray,
    max_gap_frames: int = 5,
) -> GapFill:
    """Bridge short tracking dropouts so the downstream maths sees a continuous trace.

    Interior gaps are always filled linearly — the arrays have to stay contiguous for
    filtering — but anything longer than ``max_gap_frames`` is marked invalid and
    reported, so a long occlusion shows up as a warning rather than as plausible data.
    """
    values = np.asarray(values, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    out = values.copy()
    valid = ok.copy()
    interpolated = np.zeros_like(ok)
    warnings: list[str] = []

    if not ok.any():
        raise ValueError(
            "neither marker was tracked in any frame — check the --upper-color / "
            "--lower-color presets against the tape actually used"
        )

    good = np.flatnonzero(ok)
    first, last = int(good[0]), int(good[-1])

    for start, stop in find_runs(~ok):
        span_s = (t_s[min(stop, len(t_s) - 1)] - t_s[start]) if len(t_s) else 0.0
        interpolated[start:stop] = True
        if stop <= first:  # leading dropout: hold the first good value
            out[start:stop] = values[first]
            valid[start:stop] = False
            warnings.append(
                f"markers not visible for the first {span_s:.2f} s — that stretch is excluded"
            )
        elif start > last:  # trailing dropout
            out[start:stop] = values[last]
            valid[start:stop] = False
            warnings.append(
                f"markers not visible for the last {span_s:.2f} s — that stretch is excluded"
            )
        else:
            lo, hi = start - 1, stop
            out[start:stop] = np.interp(
                np.arange(start, stop), [lo, hi], [values[lo], values[hi]]
            )
            if (stop - start) > max_gap_frames:
                valid[start:stop] = False
                warnings.append(
                    f"marker lost for {span_s:.2f} s at t={t_s[start]:.2f} s "
                    f"({stop - start} frames) — interpolated, treat that stretch with caution"
                )
            else:
                # A blink this short is bridged accurately enough to keep using.
                valid[start:stop] = True

    return GapFill(values=out, valid=valid, interpolated=interpolated, warnings=warnings)


def _window_length(n: int, fps: float, window_s: float, polyorder: int) -> int:
    """Odd Savitzky-Golay window, clamped to something the data can actually support."""
    win = max(round(window_s * fps), polyorder + 2)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win <= polyorder:
        raise ValueError(
            f"only {n} frames available — too few to filter; check the trim window"
        )
    return win


# Cubic rather than quadratic. Measured against a known damped response, a quadratic
# fit over the same window clips the peak of the rebound stroke by 5-9%, while a cubic
# tracks it to better than 0.5% — and peak rebound speed is one of the headline
# numbers, so a systematic shortfall there is not acceptable.
POLYORDER = 3


def smooth(
    y: np.ndarray, fps: float, window_s: float = 0.08, polyorder: int = POLYORDER
) -> np.ndarray:
    """Zero-phase smoothing that keeps the peaks where they are.

    Savitzky-Golay is used rather than a moving average because the peak of the
    rebound stroke is exactly the feature being measured, and a box filter would
    round it off and drag it in time.
    """
    y = np.asarray(y, dtype=float)
    win = _window_length(len(y), fps, window_s, polyorder)
    return np.asarray(savgol_filter(y, win, polyorder, mode="interp"), dtype=float)


def velocity(
    y: np.ndarray, fps: float, window_s: float = 0.08, polyorder: int = POLYORDER
) -> np.ndarray:
    """First derivative in units-per-second, fitted rather than differenced."""
    y = np.asarray(y, dtype=float)
    win = _window_length(len(y), fps, window_s, polyorder)
    return np.asarray(
        savgol_filter(y, win, polyorder, deriv=1, delta=1.0 / fps, mode="interp"),
        dtype=float,
    )


def crossing_time(
    t_s: np.ndarray, y: np.ndarray, level: float, start: int, rising: bool = True
) -> float | None:
    """Time at which ``y`` first crosses ``level`` at or after ``start``.

    Linearly interpolated between samples, which matters at low frame rates where a
    whole frame can be several percent of the rebound stroke.
    """
    y = np.asarray(y, dtype=float)
    for i in range(max(start + 1, 1), len(y)):
        prev, cur = y[i - 1], y[i]
        hit = (prev < level <= cur) if rising else (prev > level >= cur)
        if hit:
            span = cur - prev
            frac = 0.0 if span == 0 else (level - prev) / span
            return float(t_s[i - 1] + frac * (t_s[i] - t_s[i - 1]))
    return None
