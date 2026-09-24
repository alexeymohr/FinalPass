"""Tests for the model-agnostic speech-quality evaluation harness.

Synthetic numeric fixtures only — no client audio, no model weights, no torch.
The harness keeps its decision logic torch-free precisely so these can run
inside FinalPass's own environment alongside the shipped suite.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from sqa_eval.channels import (  # noqa: E402
    DUAL_MONO_NULL_DB,
    ChannelPolicyRefused,
    describe_layout,
    null_depth_db,
    to_mono,
)
from sqa_eval.chunking import (  # noqa: E402
    SampleRateUnsupported,
    coverage,
    frame_center_source_sample,
    frame_source_samples,
    frame_to_source_span,
    plan_chunks,
    total_frames,
)
from sqa_eval.config import FROZEN  # noqa: E402
from sqa_eval.events import (  # noqa: E402
    binary_runs,
    build_candidates,
    rank_candidates,
    smooth_binary,
)
from sqa_eval.model_files import (  # noqa: E402
    AUDITED_CHECKPOINT_GLOBALS,
    ModelFileRejected,
    load_manifest,
    sha256_file,
    verify_manifest,
)
from sqa_eval.netguard import NetworkAccessDenied, NetworkGuard  # noqa: E402
from sqa_eval.speech_content import frame_rms_dbfs, speech_fraction  # noqa: E402

SR = 44100
STEP = 882  # source samples per 20 ms model frame at 44.1 kHz


# --- LAION retirement -----------------------------------------------------


def test_laion_harness_is_gone_from_the_tree() -> None:
    assert not (TOOLS / "laion_eval").exists()
    assert not (ROOT / "tests" / "test_laion_harness.py").exists()


def test_no_source_still_imports_or_calls_laion_code() -> None:
    """Prose may record that LAION was retired; nothing may still reach for it."""
    offenders = []
    for path in list((ROOT / "src").rglob("*.py")) + list(TOOLS.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            if any("laion" in n.lower() for n in names):
                offenders.append(f"{path.relative_to(ROOT)}: {names}")
    assert offenders == []


def test_finalpass_package_carries_no_laion_traces_at_all() -> None:
    offenders = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
                 if "laion" in p.read_text().lower()]
    assert offenders == []


def test_laion_import_is_unreachable() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("laion_eval.policy")


def test_retirement_note_records_the_lesson() -> None:
    note = (ROOT / "docs" / "LAION_RETIREMENT_NOTE.md").read_text().lower()
    assert "do not reintroduce" in note
    assert "speech-content precondition" in note


# --- frozen configuration -------------------------------------------------


def test_frozen_post_processing_matches_the_mission_numbers() -> None:
    d = FROZEN.as_dict()
    assert d["smoothing_ms"] == 200.0
    assert d["min_event_ms"] == 100.0
    assert d["calibration_percentile"] == 1.0
    assert d["frame_seconds"] == 0.02


# --- frame <-> sample mapping --------------------------------------------


def test_frame_grid_is_exact_whole_samples_at_supported_rates() -> None:
    assert frame_source_samples(16000) == 320
    assert frame_source_samples(44100) == 882
    assert frame_source_samples(48000) == 960
    assert frame_source_samples(24000) == 480


def test_unsupported_sample_rate_is_refused_not_rounded() -> None:
    with pytest.raises(SampleRateUnsupported):
        frame_source_samples(44101)


def test_frame_count_is_ceil_of_the_source_length() -> None:
    assert total_frames(0, SR) == 0
    assert total_frames(1, SR) == 1
    assert total_frames(STEP, SR) == 1
    assert total_frames(STEP + 1, SR) == 2
    assert total_frames(100 * STEP, SR) == 100


def test_frames_tile_the_source_with_no_hole_and_no_overlap() -> None:
    total = 100 * STEP + 17
    spans = [frame_to_source_span(f, SR, total) for f in range(total_frames(total, SR))]
    assert spans[0][0] == 0
    assert spans[-1][1] == total
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start


def test_frame_positions_are_exact_integers_not_float_round_trips() -> None:
    for f in (0, 1, 999, 123456):
        start, _ = frame_to_source_span(f, SR, 10**12)
        assert start == f * STEP
        assert frame_center_source_sample(f, SR) == f * STEP + STEP // 2


# --- chunk schedule -------------------------------------------------------


def test_chunk_coverage_has_zero_holes_across_awkward_lengths() -> None:
    for total in (1, STEP, 10 * STEP + 1, SR * 31, SR * 3600 + 12345):
        chunks = plan_chunks(total, SR)
        cov = coverage(chunks, total, SR)
        assert cov["uncovered_frames"] == 0, total
        assert cov["duplicated_frames"] == 0, total
        assert cov["kept_frames"] == cov["total_frames"], total


def test_kept_ranges_tile_back_to_back_in_order() -> None:
    chunks = plan_chunks(SR * 600, SR)
    assert chunks[0].keep_start == 0
    for a, b in zip(chunks, chunks[1:]):
        assert a.keep_end == b.keep_start
    assert chunks[-1].keep_end == total_frames(SR * 600, SR)


def test_interior_chunks_carry_the_frozen_context_margin_on_both_sides() -> None:
    chunks = plan_chunks(SR * 600, SR)
    middle = chunks[2]
    assert middle.keep_start - middle.start_frame == FROZEN.edge_margin_frames
    assert middle.end_frame - middle.keep_end == FROZEN.edge_margin_frames
    # The margin is analysed but discarded: it never reaches the results.
    assert middle.keep_count == FROZEN.hop_frames
    assert middle.frame_count == FROZEN.chunk_frames


def test_first_and_last_chunk_only_lose_the_margin_the_file_can_supply() -> None:
    chunks = plan_chunks(SR * 600, SR)
    assert chunks[0].start_frame == 0
    assert chunks[0].keep_offset == 0
    assert chunks[-1].end_frame == total_frames(SR * 600, SR)


def test_chunk_plan_is_deterministic() -> None:
    a = plan_chunks(SR * 1234 + 7, SR)
    b = plan_chunks(SR * 1234 + 7, SR)
    assert a == b


def test_chunk_sample_bounds_land_on_frame_boundaries() -> None:
    total = SR * 300 + 123
    for chunk in plan_chunks(total, SR):
        assert chunk.start_sample % STEP == 0
        assert chunk.end_sample in (chunk.end_frame * STEP, total)


# --- smoothing / minimum duration ----------------------------------------


def test_smoothing_matches_the_upstream_majority_vote_example() -> None:
    # From `conv_smoothing` in fgnt/local_sqa: window 3, threshold 2.
    got = smooth_binary([False, True, True, True, False, False, False, True], 3, 2)
    assert got == [False, True, True, True, False, False, False, False]


def test_smoothing_preserves_length_and_is_deterministic() -> None:
    flags = [bool((i * 7) % 5 < 2) for i in range(200)]
    once = smooth_binary(flags, FROZEN.smoothing_frames, FROZEN.smoothing_threshold)
    twice = smooth_binary(flags, FROZEN.smoothing_frames, FROZEN.smoothing_threshold)
    assert once == twice
    assert len(once) == len(flags)


def test_smoothing_fills_a_one_frame_gap_inside_a_long_detection() -> None:
    flags = [True] * 10 + [False] + [True] * 10
    out = smooth_binary(flags, FROZEN.smoothing_frames, FROZEN.smoothing_threshold)
    assert out[10] is True


def test_smoothing_removes_an_isolated_single_frame() -> None:
    flags = [False] * 20 + [True] + [False] * 20
    assert not any(smooth_binary(flags, FROZEN.smoothing_frames, FROZEN.smoothing_threshold))


def test_binary_runs_finds_contiguous_regions_including_the_tail() -> None:
    assert binary_runs([False, True, True, False, True]) == [(1, 3), (4, 5)]


def _scores(low_frames: int, *, lead: int = 40) -> list[float]:
    return [4.5] * lead + [1.5] * low_frames + [4.5] * lead


def test_detections_shorter_than_100ms_are_discarded() -> None:
    # 4 frames = 80 ms survives smoothing but must not survive the duration rule.
    short = build_candidates(_scores(4), threshold=2.79, source_id="a.wav",
                             sample_rate=SR, total_samples=84 * STEP)
    assert short == []


def test_detections_of_at_least_100ms_survive() -> None:
    kept = build_candidates(_scores(12), threshold=2.79, source_id="a.wav",
                            sample_rate=SR, total_samples=92 * STEP)
    assert len(kept) == 1
    assert kept[0].duration_seconds >= 0.1


def test_candidate_bounds_are_exact_original_samples() -> None:
    total = 92 * STEP
    (c,) = build_candidates(_scores(12), threshold=2.79, source_id="a.wav",
                            sample_rate=SR, total_samples=total)
    assert c.start_sample == c.start_frame * STEP
    assert c.end_sample == min(c.end_frame * STEP, total)
    assert c.start_seconds == c.start_sample / SR


def test_candidate_statistics_describe_the_region_only() -> None:
    scores = [4.5] * 40 + [1.5, 1.2, 1.9, 1.4, 1.6, 1.8, 1.3, 1.7, 1.1, 1.5, 1.2, 1.6] + [4.5] * 40
    (c,) = build_candidates(scores, threshold=2.79, source_id="a.wav",
                            sample_rate=SR, total_samples=len(scores) * STEP)
    region = scores[c.start_frame:c.end_frame]
    assert c.min_score == min(region)
    assert c.mean_score == pytest.approx(sum(region) / len(region))
    assert c.frames_below_threshold == sum(1 for s in region if s < 2.79)


def test_candidate_building_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    scores = (rng.random(4000) * 4 + 1).tolist()
    a = build_candidates(scores, threshold=2.79, source_id="a.wav",
                         sample_rate=SR, total_samples=4000 * STEP)
    b = build_candidates(scores, threshold=2.79, source_id="a.wav",
                         sample_rate=SR, total_samples=4000 * STEP)
    assert [x.as_dict() for x in a] == [x.as_dict() for x in b]


def test_adjacent_low_regions_merge_into_one_candidate() -> None:
    # Two 10-frame dips one frame apart: the 200 ms vote bridges them.
    scores = [4.5] * 40 + [1.5] * 10 + [4.5] + [1.5] * 10 + [4.5] * 40
    got = build_candidates(scores, threshold=2.79, source_id="a.wav",
                           sample_rate=SR, total_samples=len(scores) * STEP)
    assert len(got) == 1


def test_regions_far_apart_stay_separate() -> None:
    scores = [4.5] * 40 + [1.5] * 10 + [4.5] * 80 + [1.5] * 10 + [4.5] * 40
    got = build_candidates(scores, threshold=2.79, source_id="a.wav",
                           sample_rate=SR, total_samples=len(scores) * STEP)
    assert len(got) == 2


# --- boundary classification ---------------------------------------------


def test_region_overlapping_a_digital_black_gap_is_boundary_adjacent() -> None:
    scores = _scores(12)
    total = len(scores) * STEP
    (c,) = build_candidates(scores, threshold=2.79, source_id="a.wav", sample_rate=SR,
                            total_samples=total, gap_spans=[(44 * STEP, 46 * STEP)])
    assert c.overlaps_boundary_gap
    assert not c.interior


def test_region_far_from_every_boundary_is_interior() -> None:
    scores = _scores(12)
    total = len(scores) * STEP
    (c,) = build_candidates(scores, threshold=2.79, source_id="a.wav", sample_rate=SR,
                            total_samples=total, gap_spans=[(0, STEP)])
    assert not c.overlaps_boundary_gap
    assert c.interior
    assert c.distance_to_clip_boundary_samples is not None


# --- ranking --------------------------------------------------------------


def _cand(min_score, mean_score, frames, start, source="a.wav"):
    return build_candidates(
        [4.5] * 40 + [min_score] + [mean_score] * (frames - 1) + [4.5] * 40,
        threshold=2.79, source_id=source, sample_rate=SR,
        total_samples=(80 + frames) * STEP,
    )


def test_ranking_puts_the_lowest_minimum_score_first() -> None:
    a = _cand(1.1, 1.2, 12, 0)[0]
    b = _cand(2.1, 1.2, 12, 0)[0]
    assert rank_candidates([b, a])[0] is a


def test_ranking_is_fully_tie_broken_and_stable() -> None:
    rows = []
    for i, src in enumerate(("c.wav", "a.wav", "b.wav")):
        rows.extend(_cand(1.5, 1.5, 12, 0, source=src))
    ranked = rank_candidates(rows)
    assert [c.source_id for c in ranked] == ["a.wav", "b.wav", "c.wav"]
    assert rank_candidates(list(reversed(rows))) == ranked


def test_longer_duration_outranks_shorter_at_equal_scores() -> None:
    short = _cand(1.5, 1.5, 12, 0)[0]
    long = _cand(1.5, 1.5, 40, 0)[0]
    assert rank_candidates([short, long])[0] is long


# --- serialization --------------------------------------------------------


def test_serialized_candidate_carries_no_waveform_field() -> None:
    (c,) = build_candidates(_scores(12), threshold=2.79, source_id="a.wav",
                            sample_rate=SR, total_samples=92 * STEP)
    payload = json.dumps(c.as_dict())
    assert json.loads(payload) == c.as_dict()
    banned = ("audio", "waveform", "samples_array", "pcm", "spectrogram", "data")
    assert not any(b in c.as_dict() for b in banned)
    for value in c.as_dict().values():
        assert isinstance(value, (int, float, str, bool, type(None)))


def test_results_round_trip_as_pure_numbers() -> None:
    rows = [c.as_dict() for c in build_candidates(
        _scores(12), threshold=2.79, source_id="a.wav", sample_rate=SR,
        total_samples=92 * STEP)]
    assert json.loads(json.dumps(rows)) == rows


# --- model provisioning ---------------------------------------------------


def test_missing_model_directory_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ModelFileRejected, match="does not exist"):
        verify_manifest(tmp_path / "nope", {"a.pt": "0" * 64})


def test_missing_model_file_fails_closed_without_any_download(tmp_path: Path) -> None:
    with pytest.raises(ModelFileRejected, match="performs no downloads"):
        verify_manifest(tmp_path, {"weights.pt": "0" * 64})


def test_altered_model_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"real weights")
    good = sha256_file(path)
    assert verify_manifest(tmp_path, {"weights.pt": good}) == {"weights.pt": path}
    path.write_bytes(b"tampered weights")
    with pytest.raises(ModelFileRejected, match="pinned hashes"):
        verify_manifest(tmp_path, {"weights.pt": good})


def test_manifest_without_usable_hashes_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"files": {"a.pt": "short"}}))
    with pytest.raises(ModelFileRejected):
        load_manifest(bad)
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"files": {}}))
    with pytest.raises(ModelFileRejected):
        load_manifest(empty)


def test_no_harness_source_permits_an_unrestricted_checkpoint_load() -> None:
    """The forbidden spellings appear only inside the deny-list that names them."""
    from sqa_eval.model_files import FORBIDDEN_LOAD_KWARGS

    for path in TOOLS.rglob("*.py"):
        text = path.read_text()
        allowed = path.name == "model_files.py"
        for banned in FORBIDDEN_LOAD_KWARGS:
            assert banned not in text or allowed, f"{path}: {banned}"
        assert "pickle.loads" not in text, path
    assert FORBIDDEN_LOAD_KWARGS == ("weights_only=False", "trust_remote_code")


def test_every_torch_load_in_the_harness_is_weights_only() -> None:
    for path in TOOLS.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            if name != "load":
                continue
            owner = getattr(getattr(func, "value", None), "id", "")
            if owner != "torch":
                continue
            kwargs = {k.arg: k.value for k in node.keywords}
            assert "weights_only" in kwargs, f"{path}: torch.load without weights_only"
            assert kwargs["weights_only"].value is True, f"{path}: weights_only not True"


def test_audited_globals_are_documented_and_numeric_only() -> None:
    assert set(AUDITED_CHECKPOINT_GLOBALS) == {
        "numpy.core.multiarray.scalar", "numpy.dtype",
        "numpy.dtypes.Float64DType", "_codecs.encode",
    }


# --- network guard --------------------------------------------------------


def test_network_guard_raises_on_a_deliberate_outbound_connection() -> None:
    guard = NetworkGuard().install()
    try:
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("example.invalid", 80), timeout=1)
    finally:
        guard.uninstall()
    assert len(guard.attempts) == 1


def test_network_guard_blocks_name_resolution_too() -> None:
    guard = NetworkGuard().install()
    try:
        with pytest.raises(NetworkAccessDenied):
            socket.getaddrinfo("huggingface.co", 443)
    finally:
        guard.uninstall()


def test_network_guard_blocks_loopback() -> None:
    guard = NetworkGuard().install()
    try:
        with pytest.raises(NetworkAccessDenied):
            socket.socket().connect(("127.0.0.1", 9))
    finally:
        guard.uninstall()


def test_network_guard_sets_offline_environment_for_ml_libraries() -> None:
    import os
    guard = NetworkGuard().install()
    try:
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    finally:
        guard.uninstall()


def test_network_guard_restores_sockets_on_exit() -> None:
    original = socket.socket.connect
    with NetworkGuard():
        assert socket.socket.connect is not original
    assert socket.socket.connect is original


def test_offline_proof_covers_the_torch_hub_download_path() -> None:
    text = (TOOLS / "sqa_eval" / "offline_proof.py").read_text()
    assert "WAV2VEC2_BASE.get_model()" in text
    assert "torch_hub_download_refused" in text


# --- threshold provenance -------------------------------------------------


def test_scan_reads_the_threshold_and_never_recomputes_a_percentile() -> None:
    text = (TOOLS / "sqa_eval" / "scan.py").read_text()
    assert 'calibration["threshold"]["value"]' in text
    assert "percentile" not in text.replace("np.percentile(scores, 1)", "")
    tree = ast.parse(text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    percentile_calls = [c for c in calls if getattr(c.func, "attr", "") == "percentile"]
    # The only percentile the scan computes is a per-file descriptive statistic,
    # never the threshold.
    assert all(isinstance(c.args[1], ast.Constant) for c in percentile_calls)


def test_scan_refuses_to_run_if_the_frozen_config_drifted(tmp_path: Path) -> None:
    from sqa_eval.scan import FrozenConfigDrift, _check_frozen

    good = {"frozen_config": FROZEN.as_dict()}
    _check_frozen(good)  # must not raise
    drifted = {"frozen_config": {**FROZEN.as_dict(), "chunk_frames": 999}}
    with pytest.raises(FrozenConfigDrift, match="chunk_frames"):
        _check_frozen(drifted)


def test_calibration_percentile_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    values = rng.random(50_000) * 4 + 1
    a = float(np.percentile(values, FROZEN.calibration_percentile))
    b = float(np.percentile(values.copy(), FROZEN.calibration_percentile))
    assert a == b
    assert abs((values < a).mean() - 0.01) < 1e-3


# --- stereo policy --------------------------------------------------------


def test_mono_is_used_directly() -> None:
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    layout = describe_layout(x)
    assert layout["channels"] == 1
    assert to_mono(x, layout).shape == (1000,)


def test_dual_mono_stereo_collapses_to_one_leg() -> None:
    mono = np.random.default_rng(3).standard_normal((1000, 1)).astype(np.float32)
    stereo = np.repeat(mono, 2, axis=1)
    layout = describe_layout(stereo)
    assert layout["dual_mono"]
    assert null_depth_db(stereo[:, 0], stereo[:, 1]) >= DUAL_MONO_NULL_DB
    assert np.array_equal(to_mono(stereo, layout), stereo[:, 0])


def test_discrete_stereo_is_refused_not_half_analysed() -> None:
    rng = np.random.default_rng(5)
    stereo = rng.standard_normal((1000, 2)).astype(np.float32)
    layout = describe_layout(stereo)
    assert not layout["dual_mono"]
    with pytest.raises(ChannelPolicyRefused, match="no documented single-stream"):
        to_mono(stereo, layout, "x.wav")


def test_surround_is_out_of_scope() -> None:
    block = np.zeros((100, 6), dtype=np.float32)
    with pytest.raises(ChannelPolicyRefused, match="out of scope"):
        to_mono(block, describe_layout(block), "x.wav")


def test_near_dual_mono_below_the_threshold_is_still_refused() -> None:
    rng = np.random.default_rng(9)
    left = rng.standard_normal(20000)
    right = left + rng.standard_normal(20000) * 0.05  # ~26 dB null, under 40
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    layout = describe_layout(stereo)
    assert layout["null_depth_db"] < DUAL_MONO_NULL_DB
    with pytest.raises(ChannelPolicyRefused):
        to_mono(stereo, layout, "x.wav")


# --- speech content -------------------------------------------------------


def test_speech_content_separates_silence_from_speech() -> None:
    assert frame_rms_dbfs(np.zeros(882)) == float("-inf")
    loud = np.full(882, 0.1)
    assert -21.0 < frame_rms_dbfs(loud) < -19.0
    assert speech_fraction([-10.0, -50.0, -20.0, -90.0]) == 0.5


def test_speech_content_is_annotation_not_a_filter() -> None:
    scores = _scores(12)
    quiet = build_candidates(scores, threshold=2.79, source_id="a.wav", sample_rate=SR,
                             total_samples=92 * STEP,
                             speech_content=[0.0] * len(scores))
    assert len(quiet) == 1  # still reported
    assert quiet[0].speech_content == 0.0


# --- AAF markers ----------------------------------------------------------


def test_aaf_markers_land_on_the_same_samples_as_the_json(tmp_path: Path) -> None:
    from finalpass.timecode import samples_to_tc, timecode_mode
    from sqa_eval.markers import candidates_to_markers, write_candidate_markers

    mode = timecode_mode(23.976)
    rows = [c.as_dict() for c in build_candidates(
        _scores(12), threshold=2.79, source_id="a.wav", sample_rate=SR,
        total_samples=92 * STEP, speech_content=[1.0] * 92)]
    markers = candidates_to_markers(rows, mode=mode)
    assert len(markers) == len(rows) == 1
    for row, marker in zip(rows, markers):
        assert marker.start_sample == row["start_sample"]
        assert marker.end_sample == row["end_sample"]
        assert marker.sample_rate == row["sample_rate"]
        assert marker.start_tc == samples_to_tc(row["start_sample"], SR, mode)
        assert marker.length_edit_units >= 1

    out = tmp_path / "sqa_markers.aaf"
    assert write_candidate_markers(out, rows, mode=mode) == 1
    assert out.is_file() and out.stat().st_size > 0


def test_marker_labels_never_name_a_defect_class() -> None:
    from sqa_eval.markers import MARKER_CODE, marker_label

    row = {"min_score": 1.83, "interior": True, "duration_seconds": 0.42}
    label = marker_label(row).lower()
    for guess in ("warble", "buzz", "clipping", "bad_tts", "distortion", "garble"):
        assert guess not in label
        assert guess not in MARKER_CODE
    assert MARKER_CODE == "low_frame_quality_region"


# --- report shape ---------------------------------------------------------


def test_report_csv_fields_are_numeric_metadata_only() -> None:
    from sqa_eval.report import CSV_FIELDS

    banned = ("audio", "waveform", "pcm", "transcript", "text", "spectrogram")
    assert not any(b in field for field in CSV_FIELDS for b in banned)
    assert "start_sample" in CSV_FIELDS and "end_sample" in CSV_FIELDS


def test_csv_leads_with_file_time_and_problem() -> None:
    from sqa_eval.report import CSV_FIELDS

    assert CSV_FIELDS[:3] == ["file", "time", "problem"]


def test_clock_is_exact_time_from_file_start() -> None:
    from sqa_eval.report import clock

    assert clock(0, SR) == "0:00:00.000"
    assert clock(SR * 3723 + SR // 2, SR) == "1:02:03.500"
    assert clock(441, SR) == "0:00:00.010"


def test_problem_description_states_severity_length_and_place_only() -> None:
    from sqa_eval.report import describe

    row = {"min_score": 1.04, "duration_seconds": 0.66, "speech_content": 0.94,
           "interior": True}
    assert describe(row) == "Severe drop in voice quality during speech (660 ms)"
    edge = {**row, "min_score": 2.4, "interior": False, "speech_content": 0.1}
    assert describe(edge) == (
        "Slight drop in voice quality in a pause or near-silence, at a clip join (660 ms)")
    for guess in ("warble", "buzz", "clipping", "garble", "distort", "glitch"):
        assert guess not in describe(row).lower()
        assert guess not in describe(edge).lower()


def test_report_uses_the_frozen_speech_and_context_constants() -> None:
    """Guards a real bug: reading these off FROZEN silently yielded zero events."""
    from sqa_eval import config
    from sqa_eval import report as report_mod

    assert report_mod.SPEECH_RICH_FRACTION == config.SPEECH_RICH_FRACTION
    assert report_mod.AUDITION_CONTEXT_SECONDS == config.AUDITION_CONTEXT_SECONDS
    # These are module constants, not FrozenConfig fields; the frozen-config
    # dict the scan compares against must stay exactly as calibrated.
    assert "speech_rich_fraction" not in FROZEN.as_dict()
    assert "audition_context_seconds" not in FROZEN.as_dict()


def test_burden_metrics_are_computed_from_unique_seconds() -> None:
    from sqa_eval.report import burden

    rows = [{"duration_seconds": 0.5, "min_score": 1.0},
            {"duration_seconds": 1.5, "min_score": 2.0}]
    got = burden(rows, hours=2.0)
    assert got["events"] == 2
    assert got["unique_candidate_seconds"] == 2.0
    assert got["events_per_finished_hour"] == 1.0
    assert got["audition_minutes_per_finished_hour"] == round(2.0 / 60 / 2, 3)


# --- no ML dependency leaks into FinalPass --------------------------------


def test_finalpass_package_does_not_import_the_ml_stack() -> None:
    banned = ("torch", "torchaudio", "padertorch", "paderbox", "transformers", "pydub")
    offenders = []
    for path in (ROOT / "src" / "finalpass").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name.split(".")[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in banned:
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert offenders == []


def test_project_runtime_dependencies_stay_free_of_the_ml_stack() -> None:
    text = (ROOT / "pyproject.toml").read_text().lower()
    for banned in ("torch", "torchaudio", "padertorch", "paderbox", "transformers"):
        assert banned not in text


def test_torch_free_harness_modules_import_without_the_eval_venv() -> None:
    for module in ("config", "chunking", "events", "model_files",
                   "netguard", "speech_content", "channels"):
        source = (TOOLS / "sqa_eval" / f"{module}.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name.split(".")[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert "torch" not in names, f"{module} imports torch"
