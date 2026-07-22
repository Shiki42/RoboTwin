import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.audit_putcab_casm_dataset import audit_dataset


def _write_fixture(root, *, include_markers=True, readme=""):
    # Synthetic data is used only as a unit-test fixture.
    (root / "meta").mkdir()
    (root / "data/chunk-000").mkdir(parents=True)
    features = {
        "action": {"shape": [14]},
        "observation.state": {"shape": [14]},
        "observation.images.cam_high": {"shape": [3, 8, 8]},
        "observation.images.cam_left_wrist": {"shape": [3, 8, 8]},
        "observation.images.cam_right_wrist": {"shape": [3, 8, 8]},
    }
    if include_markers:
        features.update(
            {
                "observation.arm_active_mask": {"shape": [2]},
                "observation.phase_one_hot": {"shape": [2]},
            }
        )
    info = {"total_episodes": 1, "total_frames": 2, "features": features}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n')
    (root / "README.md").write_text(readme)
    arrays = {
        "action": pa.array([np.zeros(14).tolist(), np.ones(14).tolist()]),
        "observation.state": pa.array([np.zeros(14).tolist(), np.ones(14).tolist()]),
    }
    if include_markers:
        arrays.update(
            {
                "observation.arm_active_mask": pa.array([[1, 1], [1, 0]]),
                "observation.phase_one_hot": pa.array([[1, 0], [0, 1]]),
            }
        )
    pq.write_table(pa.table(arrays), root / "data/chunk-000/episode_000000.parquet")


def test_audit_passes_and_computes_phase_weight(tmp_path):
    _write_fixture(tmp_path)
    receipt = audit_dataset(
        tmp_path,
        expected_episodes=1,
        expected_repo="Shiki42/test",
        reject_fixed_roles=True,
    )
    assert receipt["status"] == "passed"
    assert receipt["phase_counts"] == {"sync": 1, "async": 1}
    assert receipt["gate_positive_weight"] == 1
    assert receipt["action_horizon_boundary_mask"]["cross_boundary_targets_masked"] == 49


def test_audit_rejects_missing_markers(tmp_path):
    _write_fixture(tmp_path, include_markers=False)
    with pytest.raises(ValueError, match="missing required dataset features"):
        audit_dataset(
            tmp_path,
            expected_episodes=1,
            expected_repo="Shiki42/test",
            reject_fixed_roles=False,
        )


def test_audit_can_reject_fixed_role_dataset(tmp_path):
    _write_fixture(tmp_path, readme="Fixed arm assignment: left arm carries the object; right arm opens the drawer.")
    with pytest.raises(ValueError, match="fixed arm roles"):
        audit_dataset(
            tmp_path,
            expected_episodes=1,
            expected_repo="Shiki42/test",
            reject_fixed_roles=True,
        )
