"""End-to-end pipeline: video in, metrics and trace out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import calibration, events, metrics, signal_proc
from .colors import get_preset
from .tracking import track_video
from .video import VideoSource


class AnalysisError(RuntimeError):
    pass


@dataclass
class AnalysisConfig:
    video: Path
    upper_color: str = "pink"
    lower_color: str = "green"

    # Calibration — at least one route must be supplied.
    travel_mm: float | None = None
    stanchion_mm: float | None = None
    stanchion_pt_a: tuple[float, float] | None = None
    stanchion_pt_b: tuple[float, float] | None = None

    fps: float | None = None
    start_s: float | None = None
    end_s: float | None = None

    min_area: float = 40.0
    max_jump: float = 60.0
    max_gap_frames: int = 5
    smooth_window_s: float = 0.08

    v_still_mm_s: float = events.V_STILL_MM_S
    t_still_s: float = events.T_STILL_S
    level_tol_mm: float = events.LEVEL_TOL_MM
    release_prominence_mm: float = events.RELEASE_PROMINENCE_MM
    release_min_depth_mm: float = events.RELEASE_MIN_DEPTH_MM
    release_min_sep_s: float = events.RELEASE_MIN_SEP_S
    settle_tol_mm: float = metrics.SETTLE_TOL_MM


@dataclass
class AnalysisResult:
    payload: dict
    trace: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)


def run(config: AnalysisConfig) -> AnalysisResult:
    if config.travel_mm is None and config.stanchion_mm is None:
        raise AnalysisError(
            "no way to convert pixels to millimetres — pass --travel (preferred: "
            "bottom the fork out in the clip) and/or --stanchion with two points"
        )

    source = VideoSource(
        config.video, fps_override=config.fps, start_s=config.start_s, end_s=config.end_s
    )
    warnings = source.warnings()
    fps = source.info.fps

    track = track_video(
        source.frames(),
        upper=get_preset(config.upper_color),
        lower=get_preset(config.lower_color),
        min_area=config.min_area,
        max_jump=config.max_jump,
    )
    if len(track.frames) < 10:
        raise AnalysisError(
            f"only {len(track.frames)} frames were read — check the file and the "
            "--start/--end trim window"
        )

    sep_px_raw = track.separation_px
    gaps = signal_proc.interpolate_gaps(
        sep_px_raw, track.ok, track.t_s, max_gap_frames=config.max_gap_frames
    )
    warnings.extend(gaps.warnings)

    sep_px = signal_proc.smooth(gaps.values, fps, window_s=config.smooth_window_s)
    vel_px_s = signal_proc.velocity(gaps.values, fps, window_s=config.smooth_window_s)

    # Segmentation runs in pixels, using thresholds converted with a provisional
    # scale. Plateau boundaries barely move for small scale errors, so one pass is
    # enough — the final scale is applied to the resulting levels afterwards.
    provisional = _provisional_scale(config, sep_px)
    plateaus = events.find_plateaus(
        sep_px,
        vel_px_s,
        track.t_s,
        gaps.valid,
        v_still=config.v_still_mm_s / provisional,
        t_still=config.t_still_s,
    )
    clusters = events.cluster_levels(plateaus, tol=config.level_tol_mm / provisional)
    top_cluster, sag_cluster = events.assign_reference_levels(clusters)

    top_px = top_cluster.level if top_cluster else float(np.max(sep_px))
    if top_cluster is None:
        warnings.append(
            "no still, fully-extended stretch found — using the most-extended frame as "
            "the topped-out reference; film ~2 s of the unweighted bike first"
        )
    bottom_px = float(np.min(sep_px))

    scale, cal_payload, cal_warnings = _resolve_scale(config, top_px, bottom_px)
    warnings.extend(cal_warnings)
    mm = scale.mm_per_px

    separation_mm = sep_px * mm
    velocity_mm_s = vel_px_s * mm
    top_level = top_px * mm
    compression_mm = top_level - separation_mm
    max_compression = float(np.max(compression_mm))

    for plateau in plateaus:
        plateau.level *= mm
    if top_cluster is not None:
        for plateau in top_cluster.plateaus:
            plateau.label = "topped-out"

    sag_level = None
    if sag_cluster is not None:
        sag_level = sag_cluster.level * mm
        for plateau in sag_cluster.plateaus:
            plateau.label = "sag"
    else:
        warnings.append(
            "no settled rider position found — sag needs ~3 s of the rider on the bike "
            "holding still; reporting the trace only"
        )

    if config.travel_mm is not None:
        travel_mm, travel_source = float(config.travel_mm), "supplied"
    else:
        travel_mm, travel_source = max_compression, "measured (deepest compression seen)"
        warnings.append(
            "travel was not supplied — sag % is against the deepest compression in the "
            "clip, which only equals real travel if the fork was fully bottomed"
        )

    releases: list[metrics.ReleaseMetrics] = []
    if sag_level is not None:
        for plateau in plateaus:
            if plateau.label == "unassigned" and plateau.level < sag_level:
                plateau.label = "compressed"
        idxs = events.find_release_events(
            separation_mm,
            track.t_s,
            fps,
            sag_level=sag_level,
            prominence_mm=config.release_prominence_mm,
            min_sep_s=config.release_min_sep_s,
            min_depth_mm=config.release_min_depth_mm,
        )
        for n, idx in enumerate(idxs):
            releases.append(
                metrics.analyse_release(
                    separation_mm,
                    velocity_mm_s,
                    track.t_s,
                    release_idx=idx,
                    next_release_idx=idxs[n + 1] if n + 1 < len(idxs) else None,
                    sag_level=sag_level,
                    top_level=top_level,
                    fps=fps,
                    settle_tol_mm=config.settle_tol_mm,
                )
            )
        for flagged in metrics.label_bottom_outs(releases):
            warnings.append(
                f"the event at t={flagged.t_release_s:.2f} s went "
                f"{flagged.depth_from_sag_mm:.0f} mm below sag — treated as the "
                "bottom-out and excluded from the rebound medians"
            )
        if not releases:
            warnings.append(
                "no compress-and-release events detected — push the fork at least "
                f"{config.release_min_depth_mm:g} mm below sag and let go cleanly"
            )
        elif sum(1 for r in releases if r.complete and r.role == "rebound") < 3:
            warnings.append(
                "fewer than three clean release events — the medians rest on a small "
                "sample; the protocol asks for three pushes"
            )

    sag_mm = None if sag_level is None else top_level - sag_level
    sag_pct = None if (sag_mm is None or not travel_mm) else sag_mm / travel_mm * 100.0

    quality = _quality(track, gaps, sep_px_raw, mm)
    if quality["upper_tracked_pct"] < 90 or quality["lower_tracked_pct"] < 90:
        warnings.append(
            f"markers were only found in {quality['upper_tracked_pct']:.0f}% (upper) and "
            f"{quality['lower_tracked_pct']:.0f}% (lower) of frames — check the colour "
            "presets, lighting, and that nothing crosses in front of the tape"
        )

    payload = {
        "input": {
            "video": str(Path(config.video).resolve()),
            "fps": round(fps, 3),
            "fps_source": source.info.fps_source,
            "resolution": f"{source.info.width}x{source.info.height}",
            "frames_analysed": int(len(track.frames)),
            "analysed_span_s": [
                round(float(track.t_s[0]), 3),
                round(float(track.t_s[-1]), 3),
            ],
        },
        "calibration": cal_payload,
        "sag": {
            "sag_mm": _r(sag_mm),
            "sag_pct": _r(sag_pct),
            "travel_mm": _r(travel_mm),
            "travel_source": travel_source,
            "max_compression_mm": _r(max_compression),
            "topped_out_separation_mm": _r(top_level),
            "sag_separation_mm": _r(sag_level),
        },
        "rebound": {
            **metrics.aggregate(releases),
            "events": [r.as_dict() for r in releases],
        },
        "segments": [p.as_dict() for p in sorted(plateaus, key=lambda p: p.t_start)],
        "quality": quality,
        "warnings": warnings,
    }

    trace = {
        "frame": track.frames,
        "t_s": track.t_s,
        "separation_px": sep_px_raw,
        "separation_mm": separation_mm,
        "compression_mm": compression_mm,
        "velocity_mm_s": velocity_mm_s,
        "upper_ok": track.upper_ok,
        "lower_ok": track.lower_ok,
        "valid": gaps.valid,
    }
    return AnalysisResult(payload=payload, trace=trace, warnings=warnings)


def _provisional_scale(config: AnalysisConfig, sep_px: np.ndarray) -> float:
    """A rough mm/px, good enough to set the segmentation thresholds."""
    if config.stanchion_mm is not None and config.stanchion_pt_a and config.stanchion_pt_b:
        return calibration.from_stanchion(
            config.stanchion_mm, config.stanchion_pt_a, config.stanchion_pt_b
        ).mm_per_px
    stroke = float(np.max(sep_px) - np.min(sep_px))
    if stroke <= 0:
        raise AnalysisError(
            "the marker separation never changed — the fork did not move, or the "
            "tracker locked onto something static"
        )
    return float(config.travel_mm) / stroke


def _resolve_scale(
    config: AnalysisConfig, top_px: float, bottom_px: float
) -> tuple[calibration.Scale, dict, list[str]]:
    warnings: list[str] = []
    candidates: list[calibration.Scale] = []

    if config.travel_mm is not None:
        candidates.append(calibration.from_travel(config.travel_mm, top_px, bottom_px))
    if config.stanchion_mm is not None:
        if not (config.stanchion_pt_a and config.stanchion_pt_b):
            raise AnalysisError(
                "--stanchion also needs --stanchion-pt-a and --stanchion-pt-b; run "
                "`sag-tracker calibrate` to click them off a frame"
            )
        candidates.append(
            calibration.from_stanchion(
                config.stanchion_mm, config.stanchion_pt_a, config.stanchion_pt_b
            )
        )

    # Travel wins when both are available: its error cancels if the camera is not
    # perfectly square to the fork, whereas the stanchion measurement's does not.
    chosen = candidates[0]
    payload = {
        "mm_per_px": round(chosen.mm_per_px, 6),
        "source": chosen.source,
        "detail": chosen.detail,
        "alternatives": [
            {"source": c.source, "mm_per_px": round(c.mm_per_px, 6), "detail": c.detail}
            for c in candidates[1:]
        ],
        "disagreement_pct": None,
    }
    if len(candidates) == 2:
        disagreement = calibration.disagreement_pct(candidates[0], candidates[1])
        payload["disagreement_pct"] = round(disagreement, 2)
        if disagreement > 5.0:
            warnings.append(
                f"the two calibrations disagree by {disagreement:.1f}% — the camera is "
                "probably not square to the fork, or the fork was not fully bottomed; "
                "the travel-based scale was used"
            )
    return chosen, payload, warnings


def _quality(track, gaps, sep_px_raw: np.ndarray, mm_per_px: float) -> dict:
    n = len(track.frames)
    both_ok = track.ok
    jumps = np.abs(np.diff(sep_px_raw[both_ok])) if both_ok.sum() > 1 else np.array([0.0])
    return {
        "upper_tracked_pct": round(float(track.upper_ok.mean()) * 100.0, 1),
        "lower_tracked_pct": round(float(track.lower_ok.mean()) * 100.0, 1),
        "valid_pct": round(float(gaps.valid.mean()) * 100.0, 1),
        "interpolated_frames": int(gaps.interpolated.sum()),
        "frames": int(n),
        "max_interframe_jump_px": round(float(np.max(jumps)), 2),
        "max_interframe_jump_mm": round(float(np.max(jumps)) * mm_per_px, 2),
    }


def _r(value: float | None, places: int = 2) -> float | None:
    if value is None or np.isnan(value):
        return None
    return round(float(value), places)
