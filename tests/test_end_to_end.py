"""Video in, numbers out — measured against the fixture's exact ground truth."""

from __future__ import annotations

import csv
import json

import pytest

from sag_tracker import report
from sag_tracker.analysis import AnalysisConfig, AnalysisError, run
from sag_tracker.cli import main


def test_sag_is_recovered_to_within_a_millimetre(clip):
    truth, result = clip
    assert result.payload["sag"]["sag_mm"] == pytest.approx(truth["sag_mm"], abs=1.0)
    assert result.payload["sag"]["sag_pct"] == pytest.approx(truth["sag_pct"], abs=0.6)


def test_scale_and_travel_are_recovered(clip):
    truth, result = clip
    assert result.payload["calibration"]["mm_per_px"] == pytest.approx(
        truth["mm_per_px"], rel=0.005
    )
    assert result.payload["sag"]["max_compression_mm"] == pytest.approx(
        truth["travel_mm"], abs=1.0
    )
    assert result.payload["sag"]["topped_out_separation_mm"] == pytest.approx(
        truth["separation_top_mm"], abs=1.0
    )


def test_rebound_matches_the_prescribed_response(clip):
    truth, result = clip
    rebound = result.payload["rebound"]

    assert rebound["median_damping_ratio"] == pytest.approx(truth["zeta"], abs=0.05)
    assert rebound["median_natural_freq_hz"] == pytest.approx(
        truth["natural_freq_hz"], rel=0.05
    )
    expected_peak = truth["releases"][0]["peak_rebound_velocity_mm_s"]
    assert rebound["median_peak_rebound_velocity_mm_s"] == pytest.approx(
        expected_peak, rel=0.05
    )


def test_every_push_is_found_and_the_bottom_out_is_set_aside(clip):
    truth, result = clip
    rebound = result.payload["rebound"]

    assert rebound["events_used"] == truth["config"]["cycles"]
    assert rebound["events_found"] == truth["config"]["cycles"] + 1  # + the bottom-out
    assert [e["role"] for e in rebound["events"]][-1] == "bottom-out"
    assert any("bottom-out" in w for w in result.warnings)

    for event, expected in zip(rebound["events"], truth["releases"], strict=False):
        assert event["t_release_s"] == pytest.approx(expected["t_release_s"], abs=0.05)


def test_segments_are_labelled_in_order(clip):
    _, result = clip
    labels = [s["label"] for s in result.payload["segments"]]

    assert labels[0] == "topped-out"
    assert labels[-1] == "topped-out"
    assert labels.count("sag") >= 4  # settled once before and after every push
    assert "compressed" in labels


def test_clean_footage_tracks_every_frame(clip):
    _, result = clip
    quality = result.payload["quality"]
    assert quality["upper_tracked_pct"] == 100.0
    assert quality["lower_tracked_pct"] == 100.0
    assert quality["interpolated_frames"] == 0
    assert result.warnings == [w for w in result.warnings if "bottom-out" in w]


def test_camera_shake_does_not_move_the_answers(clip, shaky_clip):
    _, tripod = clip
    _, handheld = shaky_clip

    assert handheld.payload["sag"]["sag_mm"] == pytest.approx(
        tripod.payload["sag"]["sag_mm"], abs=0.5
    )
    assert handheld.payload["rebound"]["median_peak_rebound_velocity_mm_s"] == pytest.approx(
        tripod.payload["rebound"]["median_peak_rebound_velocity_mm_s"], rel=0.02
    )
    assert handheld.payload["rebound"]["median_damping_ratio"] == pytest.approx(
        tripod.payload["rebound"]["median_damping_ratio"], abs=0.03
    )
    assert handheld.payload["quality"]["upper_tracked_pct"] == 100.0


def test_a_long_occlusion_is_reported_and_survived(clip, occluded_clip):
    truth, result = occluded_clip
    _, clean = clip

    assert any("marker lost" in w for w in result.warnings)
    assert result.payload["quality"]["interpolated_frames"] > 30
    assert result.payload["quality"]["upper_tracked_pct"] < 100.0
    # The dropout sits between events, so the numbers should still stand up.
    assert result.payload["sag"]["sag_mm"] == pytest.approx(truth["sag_mm"], abs=1.0)
    assert result.payload["rebound"]["events_used"] == clean.payload["rebound"]["events_used"]


def test_metrics_json_is_serialisable_and_trace_csv_lines_up(clip, tmp_path):
    _, result = clip

    metrics_path = report.write_metrics(tmp_path / "metrics.json", result.payload)
    trace_path = report.write_trace(tmp_path / "trace.csv", result.trace)

    reloaded = json.loads(metrics_path.read_text())
    assert reloaded == result.payload

    with trace_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.payload["input"]["frames_analysed"]
    assert list(rows[0]) == report.TRACE_COLUMNS
    assert float(rows[0]["separation_mm"]) == pytest.approx(
        result.trace["separation_mm"][0], abs=1e-3
    )


def test_analysis_refuses_to_guess_a_scale(clip):
    truth, _ = clip
    with pytest.raises(AnalysisError, match="millimetres"):
        run(AnalysisConfig(video=truth["video_path"]))


def test_stanchion_alone_also_produces_a_result(clip):
    truth, _ = clip
    result = run(
        AnalysisConfig(
            video=truth["video_path"],
            stanchion_mm=truth["stanchion_mm"],
            stanchion_pt_a=tuple(truth["stanchion_pt_a"]),
            stanchion_pt_b=tuple(truth["stanchion_pt_b"]),
        )
    )
    assert result.payload["sag"]["sag_mm"] == pytest.approx(truth["sag_mm"], rel=0.05)
    assert result.payload["sag"]["travel_source"].startswith("measured")
    assert any("travel was not supplied" in w for w in result.warnings)


def test_both_calibrations_together_are_cross_checked(clip):
    truth, _ = clip
    result = run(
        AnalysisConfig(
            video=truth["video_path"],
            travel_mm=truth["travel_mm"],
            stanchion_mm=truth["stanchion_mm"],
            stanchion_pt_a=tuple(truth["stanchion_pt_a"]),
            stanchion_pt_b=tuple(truth["stanchion_pt_b"]),
        )
    )
    calibration = result.payload["calibration"]
    assert calibration["source"] == "travel"  # the more trustworthy route wins
    assert calibration["alternatives"][0]["source"] == "stanchion"
    assert calibration["disagreement_pct"] < 5.0


def test_trimming_to_a_window_analyses_only_that_window(clip):
    truth, _ = clip
    result = run(
        AnalysisConfig(video=truth["video_path"], travel_mm=truth["travel_mm"], start_s=3.0, end_s=6.0)
    )
    start, end = result.payload["input"]["analysed_span_s"]
    assert start >= 3.0
    assert end <= 6.0


def test_a_low_frame_rate_is_called_out(clip):
    truth, _ = clip
    result = run(
        AnalysisConfig(video=truth["video_path"], travel_mm=truth["travel_mm"], fps=30.0)
    )
    assert any("30 fps" in w for w in result.warnings)


def test_cli_writes_both_artefacts(clip, tmp_path, capsys):
    truth, _ = clip
    out_dir = tmp_path / "results"

    code = main(
        ["analyze", str(truth["video_path"]), "--travel", str(truth["travel_mm"]),
         "--out-dir", str(out_dir)]
    )

    assert code == 0
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "trace.csv").exists()
    assert "wrote" in capsys.readouterr().out


def test_cli_reports_a_missing_video_without_a_traceback(tmp_path, capsys):
    code = main(["analyze", str(tmp_path / "nope.mp4"), "--travel", "160"])
    assert code == 2
    assert "error:" in capsys.readouterr().err
