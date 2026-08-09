import hashlib
import json
from pathlib import Path

import numpy as np

from robotwin_image_transport import encode_images
from scripts.serve_robotwin_policy import RobotwinPolicyService
from scripts.serve_robotwin_policy import validate_checkpoint_inventory


class FakeModel:
    def __init__(self):
        self.observation_window = None
        self.calls = []

    def reset_obsrvationwindows(self):
        self.calls.append(("reset",))
        self.observation_window = None

    def rollout_metrics(self):
        return {"chunk_count": 1}

    def record_action(self, action, state):
        self.calls.append(("record_action", action, state))

    def update_observation_window(self, images, state, *, action_executed):
        self.calls.append(("update", action_executed))
        self.observation_window = {"images": images, "state": state}

    def set_language(self, instruction):
        self.calls.append(("language", instruction))

    def get_action(self):
        return np.ones((5, 14), dtype=np.float32)

    def execution_steps(self):
        return 3

    def record_chunk(self, length):
        self.calls.append(("chunk", length))

    def set_episode_seed(self, seed):
        self.calls.append(("seed", seed))


def test_service_dispatches_infer_observe_metrics_and_reset():
    model = FakeModel()
    service = RobotwinPolicyService(model)
    raw_images = [np.zeros((2, 2, 3), dtype=np.uint8)] * 3
    observation = {"images": encode_images(raw_images), "state": np.zeros(14)}

    assert service.infer({"command": "seed", "seed": 100008}) == {"ok": True}
    response = service.infer({"command": "infer", "instruction": "task", **observation})
    assert response["actions"].shape == (3, 14)
    assert ("seed", 100008) in model.calls
    assert ("language", "task") in model.calls
    assert ("chunk", 3) in model.calls
    assert all(
        np.array_equal(image, expected)
        for image, expected in zip(
            model.observation_window["images"],
            raw_images,
            strict=True,
        )
    )

    service.infer({"command": "observe", "action": np.ones(14), "previous_state": np.zeros(14), **observation})
    assert ("update", True) in model.calls
    metrics = service.infer({"command": "metrics"})["metrics"]
    assert metrics["chunk_count"] == 1
    assert metrics["server_policy_inference_requests"] == 1
    assert len(metrics["server_policy_action_sha256"]) == 64
    assert service.infer({"command": "reset"}) == {"ok": True}


def test_checkpoint_inventory_binds_real_files(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    asset = checkpoint / "assets/repo/norm_stats.json"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"norm")
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"model")
    files = [
        {
            "path": "assets/repo/norm_stats.json",
            "size_bytes": asset.stat().st_size,
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        },
        {
            "path": "model.safetensors",
            "size_bytes": model.stat().st_size,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        },
    ]
    canonical = "".join(f"{item['sha256']}  {item['size_bytes']}  {item['path']}\n" for item in files).encode()
    tree_sha256 = hashlib.sha256(canonical).hexdigest()
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_type": "inference_checkpoint_package",
                "config_name": "pi05_putcab_spline_field_pytorch",
                "producer_code_commit": "9" * 40,
                "checkpoint_step": 30000,
                "file_count": len(files),
                "total_bytes": sum(item["size_bytes"] for item in files),
                "tree_sha256": tree_sha256,
                "files": files,
            }
        )
    )

    metadata = validate_checkpoint_inventory(
        checkpoint,
        inventory,
        expected_tree_sha256=tree_sha256,
        expected_producer_code_commit="9" * 40,
        train_config_name="pi05_putcab_spline_field_pytorch",
    )

    assert metadata["checkpoint_tree_sha256"] == tree_sha256
    assert metadata["checkpoint_file_count"] == 2
