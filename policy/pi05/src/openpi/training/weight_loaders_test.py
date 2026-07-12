from __future__ import annotations

import numpy as np

from openpi.training.weight_loaders import _merge_params


def test_merge_params_initializes_explicit_new_casm_parameters():
    loaded = {"base": np.ones((2,), dtype=np.float32)}
    reference = {
        "base": np.zeros((2,), dtype=np.float32),
        "block": {
            "lora_kernel": np.full((2,), 2.0, dtype=np.float32),
            "cooperation_gate": np.full((2,), 3.0, dtype=np.float32),
            "cross_attention": np.full((2,), 4.0, dtype=np.float32),
            "unrequested": np.full((2,), 5.0, dtype=np.float32),
        },
    }
    merged = _merge_params(
        loaded,
        reference,
        missing_regex=".*(lora|cooperation_gate|cross_attention).*",
    )
    assert merged["base"].tolist() == [1.0, 1.0]
    assert merged["block"]["lora_kernel"].tolist() == [2.0, 2.0]
    assert merged["block"]["cooperation_gate"].tolist() == [3.0, 3.0]
    assert merged["block"]["cross_attention"].tolist() == [4.0, 4.0]
    assert "unrequested" not in merged["block"]
