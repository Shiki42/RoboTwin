import argparse
import sys

import pytest

from scripts import benchmark_training_pytorch


def test_parse_data_pipeline_options(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_training_pytorch.py",
            "--config-name",
            "config",
            "--assets-base-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "timing.jsonl"),
            "--run-index",
            "1",
            "--num-workers",
            "2",
            "--prefetch-factor",
            "4",
            "--persistent-workers",
            "--pin-memory",
            "--record-batch-sha256",
        ],
    )

    args = benchmark_training_pytorch._parse_args()  # noqa: SLF001

    assert args.num_workers == 2
    assert args.prefetch_factor == 4
    assert args.persistent_workers is True
    assert args.pin_memory is True
    assert args.record_batch_sha256 is True
    assert args.gradient_accumulation_steps == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"warmup_steps": -1}, "non-negative warmup"),
        ({"measured_steps": 0}, "positive measured steps"),
        ({"num_workers": -1}, "non-negative"),
        ({"prefetch_factor": 0}, "positive"),
        ({"gradient_accumulation_steps": 0}, "gradient accumulation"),
        ({"num_workers": 0, "persistent_workers": True}, "at least one worker"),
    ],
)
def test_validate_args_rejects_invalid_settings(updates, message):
    values = {
        "warmup_steps": 5,
        "measured_steps": 100,
        "num_workers": 2,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "gradient_accumulation_steps": 1,
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        benchmark_training_pytorch._validate_args(argparse.Namespace(**values))  # noqa: SLF001
