import json

import pytest
import torch

from openpi.training import performance


def test_resolve_benchmark_assets_base_dir_requires_an_existing_directory(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x")

    assert performance.resolve_benchmark_assets_base_dir(assets_dir) == str(assets_dir.resolve())
    with pytest.raises(FileNotFoundError):
        performance.resolve_benchmark_assets_base_dir(tmp_path / "missing")
    with pytest.raises(NotADirectoryError):
        performance.resolve_benchmark_assets_base_dir(file_path)


def test_cuda_stage_timer_sums_repeated_accumulation_stages(monkeypatch):
    class FakeEvent:
        next_id = 0

        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.event_id = FakeEvent.next_id
            FakeEvent.next_id += 1
            self.synchronized = False

        def record(self):
            return None

        def synchronize(self):
            self.synchronized = True

        def elapsed_time(self, end):
            return float(end.event_id - self.event_id)

    monkeypatch.setattr(performance.torch.cuda, "Event", FakeEvent)
    timer = performance.CudaStageTimer()
    for stage in ("h2d", "forward", "backward"):
        timer.start(stage)
        timer.end(stage)
        timer.start(stage)
        timer.end(stage)
    timer.start("optimizer")
    timer.end("optimizer")

    assert timer.resolve_ms() == {
        "h2d_ms": 2.0,
        "forward_ms": 2.0,
        "backward_ms": 2.0,
        "optimizer_ms": 1.0,
    }


def test_timing_receipt_keeps_raw_warmup_and_summarizes_measured(tmp_path):
    path = tmp_path / "timings.jsonl"
    receipt = performance.TimingReceipt(path, warmup_steps=1, metadata={"batch_size": 16, "images_per_sample": 3})
    receipt.append({"step": 2001, "data_wait_ms": 100.0, "step_total_ms": 100.0})
    receipt.append({"step": 2002, "data_wait_ms": 10.0, "step_total_ms": 10.0})
    receipt.append({"step": 2003, "data_wait_ms": 30.0, "step_total_ms": 30.0})

    summary_path = receipt.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["receipt_index"] for row in rows] == [1, 2, 3]
    assert [row["phase"] for row in rows] == ["warmup", "measured", "measured"]
    summary = json.loads(summary_path.read_text())
    assert summary["measured_steps"] == 2
    assert summary["metadata"] == {"batch_size": 16, "images_per_sample": 3}
    assert summary["throughput"]["steps_per_second"] == 50.0
    assert summary["throughput"]["samples_per_second"] == 800.0
    assert summary["throughput"]["images_per_second"] == 2400.0
    assert summary["timings_ms"]["data_wait_ms"] == {
        "mean": 20.0,
        "p50": 20.0,
        "p90": 28.0,
        "p99": pytest.approx(29.8),
    }


def test_timing_receipt_reports_steady_throughput_without_checkpoint(tmp_path):
    path = tmp_path / "timings.jsonl"
    receipt = performance.TimingReceipt(path, warmup_steps=0, metadata={"batch_size": 16, "images_per_sample": 3})
    receipt.append({"step_total_ms": 100.0, "checkpoint_saved": False})
    receipt.append({"step_total_ms": 1000.0, "checkpoint_saved": True})

    summary = json.loads(receipt.close().read_text())

    assert summary["throughput"]["steps_per_second"] == pytest.approx(2 / 1.1)
    assert summary["steady_throughput_without_checkpoint"]["steps_per_second"] == 10.0
    assert summary["checkpoint_rows"] == 1


def test_timing_receipt_rejects_invalid_controls(tmp_path):
    with pytest.raises(ValueError, match="warmup"):
        performance.TimingReceipt(tmp_path / "timings.jsonl", warmup_steps=-1, metadata={})
    with pytest.raises(ValueError, match="flush"):
        performance.TimingReceipt(
            tmp_path / "timings.jsonl",
            warmup_steps=0,
            metadata={},
            flush_interval=0,
        )


def test_tree_sha256_is_order_stable_and_value_sensitive():
    left = {"b": torch.tensor([2, 3]), "a": torch.tensor([1])}
    right = {"a": torch.tensor([1]), "b": torch.tensor([2, 3])}
    changed = {"a": torch.tensor([1]), "b": torch.tensor([2, 4])}

    assert performance.tree_sha256(left) == performance.tree_sha256(right)
    assert performance.tree_sha256(left) != performance.tree_sha256(changed)


def test_summarize_values_handles_empty_and_percentiles():
    assert performance.summarize_values([]) == {}
    summary = performance.summarize_values([1.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["p50"] == 2.0
    assert summary["p90"] == pytest.approx(2.8)
    assert summary["p99"] == pytest.approx(2.98)
