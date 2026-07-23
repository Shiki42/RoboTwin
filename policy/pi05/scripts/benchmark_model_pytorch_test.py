import sys

from scripts import benchmark_model_pytorch


def test_parse_attention_implementation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_model_pytorch.py",
            "--config-name",
            "config",
            "--assets-base-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "timing.jsonl"),
            "--run-index",
            "0",
            "--attention-implementation",
            "eager",
        ],
    )

    args = benchmark_model_pytorch._parse_args()  # noqa: SLF001

    assert args.attention_implementation == "eager"
