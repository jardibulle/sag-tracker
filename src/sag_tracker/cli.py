"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import events, metrics, report
from .analysis import AnalysisConfig, AnalysisError, run
from .calibration import CalibrationError
from .colors import PRESETS
from .video import VideoError, VideoSource


def _point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y — got {text!r}") from None
    return x, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sag-tracker",
        description="Measure mountain bike suspension sag and rebound from a video.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    an = sub.add_parser("analyze", help="measure a clip and write metrics.json + trace.csv")
    an.add_argument("video", type=Path)
    an.add_argument(
        "--upper-color", default="pink", choices=sorted(PRESETS), help="tape on the stanchion"
    )
    an.add_argument(
        "--lower-color", default="green", choices=sorted(PRESETS), help="tape on the lower leg"
    )

    cal = an.add_argument_group(
        "calibration", "at least one route is required; --travel is the more reliable"
    )
    cal.add_argument(
        "--travel",
        type=float,
        metavar="MM",
        help="total fork travel; needs a bottom-out in the clip",
    )
    cal.add_argument("--stanchion", type=float, metavar="MM", help="stanchion diameter")
    cal.add_argument("--stanchion-pt-a", type=_point, metavar="X,Y")
    cal.add_argument("--stanchion-pt-b", type=_point, metavar="X,Y")

    an.add_argument("--fps", type=float, help="override the frame rate declared by the file")
    an.add_argument("--start", type=float, metavar="S", help="ignore everything before this time")
    an.add_argument("--end", type=float, metavar="S", help="ignore everything after this time")
    an.add_argument("--out-dir", type=Path, default=Path("results"))

    tune = an.add_argument_group("tuning", "defaults suit the documented filming protocol")
    tune.add_argument("--min-area", type=float, default=40.0, help="smallest believable blob, px")
    tune.add_argument("--max-jump", type=float, default=60.0, help="max marker travel per frame, px")
    tune.add_argument("--max-gap", type=int, default=5, help="dropout frames still worth bridging")
    tune.add_argument("--smooth-window", type=float, default=0.08, metavar="S")
    tune.add_argument("--v-still", type=float, default=events.V_STILL_MM_S, metavar="MM/S")
    tune.add_argument("--t-still", type=float, default=events.T_STILL_S, metavar="S")
    tune.add_argument("--level-tol", type=float, default=events.LEVEL_TOL_MM, metavar="MM")
    tune.add_argument(
        "--release-min-depth", type=float, default=events.RELEASE_MIN_DEPTH_MM, metavar="MM"
    )
    tune.add_argument("--settle-tol", type=float, default=metrics.SETTLE_TOL_MM, metavar="MM")

    cl = sub.add_parser(
        "calibrate", help="click across the stanchion to get the --stanchion-pt flags"
    )
    cl.add_argument("video", type=Path)
    cl.add_argument("--stanchion", type=float, required=True, metavar="MM")
    cl.add_argument("--at", type=float, default=0.0, metavar="S", help="frame to click on")

    sy = sub.add_parser("synth", help="render a synthetic clip with known ground truth")
    sy.add_argument("out", type=Path)
    sy.add_argument("--fps", type=float, default=120.0)
    sy.add_argument("--sag", type=float, default=32.0, metavar="MM")
    sy.add_argument("--travel", type=float, default=160.0, metavar="MM")
    sy.add_argument("--zeta", type=float, default=0.35)
    sy.add_argument("--freq", type=float, default=2.5, metavar="HZ")
    sy.add_argument("--cycles", type=int, default=3)
    sy.add_argument("--shake", action="store_true", help="add handheld camera wobble")
    sy.add_argument("--noise", type=float, default=4.0, metavar="SIGMA")
    sy.add_argument(
        "--occlude",
        type=str,
        default=None,
        metavar="T0,T1",
        help="hide the upper marker over this window, to exercise dropout handling",
    )

    return parser


def cmd_analyze(args: argparse.Namespace) -> int:
    config = AnalysisConfig(
        video=args.video,
        upper_color=args.upper_color,
        lower_color=args.lower_color,
        travel_mm=args.travel,
        stanchion_mm=args.stanchion,
        stanchion_pt_a=args.stanchion_pt_a,
        stanchion_pt_b=args.stanchion_pt_b,
        fps=args.fps,
        start_s=args.start,
        end_s=args.end,
        min_area=args.min_area,
        max_jump=args.max_jump,
        max_gap_frames=args.max_gap,
        smooth_window_s=args.smooth_window,
        v_still_mm_s=args.v_still,
        t_still_s=args.t_still,
        level_tol_mm=args.level_tol,
        release_min_depth_mm=args.release_min_depth,
        settle_tol_mm=args.settle_tol,
    )
    result = run(config)

    out_dir = args.out_dir
    metrics_path = report.write_metrics(out_dir / "metrics.json", result.payload)
    trace_path = report.write_trace(out_dir / "trace.csv", result.trace)

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"wrote {metrics_path}")
    print(f"wrote {trace_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    source = VideoSource(args.video)
    target = None
    for frame in source.frames():
        target = frame
        if frame.t_s >= args.at:
            break
    if target is None:
        raise VideoError(f"no frames read from {args.video}")

    picks: list[tuple[int, int]] = []
    window = "click the two stanchion edges  —  u undo, enter accept, esc cancel"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(picks) < 2:
            picks.append((x, y))

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            canvas = target.image.copy()
            for point in picks:
                cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            if len(picks) == 2:
                cv2.line(canvas, picks[0], picks[1], (0, 0, 255), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # esc
                print("cancelled", file=sys.stderr)
                return 1
            if key in (13, 10) and len(picks) == 2:
                break
            if key == ord("u") and picks:
                picks.pop()
    except cv2.error as exc:
        raise VideoError(
            f"could not open a window for clicking ({exc}); measure the stanchion width "
            "in any image viewer and pass --stanchion-pt-a / --stanchion-pt-b by hand"
        ) from None
    finally:
        # A build without GUI support raises here too; that must not bury the
        # explanation of why the window never opened.
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    (ax, ay), (bx, by) = picks
    print(
        f"--stanchion {args.stanchion:g} "
        f"--stanchion-pt-a {ax},{ay} --stanchion-pt-b {bx},{by}"
    )
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import SynthConfig, render

    occlusions = []
    if args.occlude:
        t0, t1 = (float(v) for v in args.occlude.split(","))
        occlusions = [(t0, t1)]

    truth = render(
        args.out,
        SynthConfig(
            fps=args.fps,
            sag_mm=args.sag,
            travel_mm=args.travel,
            zeta=args.zeta,
            natural_freq_hz=args.freq,
            cycles=args.cycles,
            shake=args.shake,
            noise_sigma=args.noise,
            occlusions_s=occlusions,
        ),
    )
    print(f"wrote {truth['video_path']}  ({truth['frames']} frames, {truth['duration_s']:.2f} s)")
    print(f"wrote {truth['truth_path']}")
    print(
        f"ground truth: sag {truth['sag_mm']:g} mm ({truth['sag_pct']:.1f} %), "
        f"travel {truth['travel_mm']:g} mm, zeta {truth['zeta']:g}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"analyze": cmd_analyze, "calibrate": cmd_calibrate, "synth": cmd_synth}
    try:
        return handlers[args.command](args)
    except (AnalysisError, VideoError, CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
