from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tyro

from openpi.serving import websocket_policy_server

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from pi_model import PI0  # noqa: E402
from robotwin_image_transport import ROBOTWIN_POLICY_PROTOCOL  # noqa: E402
from robotwin_image_transport import decode_images  # noqa: E402


class RobotwinPolicyService:
    """Stateful RPC adapter around PI0 for native RoboTwin rollouts."""

    def __init__(self, model: PI0):
        self._model = model
        self._reset_episode_trace()

    def _reset_episode_trace(self):
        self._action_hasher = hashlib.sha256()
        self._inference_requests = 0

    def _record_actions(self, actions):
        contiguous = np.ascontiguousarray(actions)
        self._action_hasher.update(contiguous.dtype.str.encode("ascii"))
        self._action_hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        self._action_hasher.update(contiguous.tobytes())
        self._inference_requests += 1

    def infer(self, request):
        command = request["command"]
        if command == "reset":
            self._model.reset_obsrvationwindows()
            self._reset_episode_trace()
            return {"ok": True}
        if command == "metrics":
            metrics = dict(self._model.rollout_metrics())
            metrics["server_policy_action_sha256"] = self._action_hasher.hexdigest()
            metrics["server_policy_inference_requests"] = self._inference_requests
            return {"metrics": metrics}
        if command == "seed":
            self._model.set_episode_seed(int(request["seed"]))
            self._reset_episode_trace()
            return {"ok": True}
        if command == "observe":
            self._model.record_action(request["action"], request["previous_state"])
            self._model.advance_after_action()
            return {"ok": True}
        if command == "infer":
            if self._model.observation_window is None:
                self._model.set_language(request["instruction"])
            self._update_observation(request, action_executed=False)
            actions = np.asarray(self._model.get_action()[: self._model.execution_steps()])
            self._model.record_chunk(len(actions))
            self._record_actions(actions)
            return {"actions": actions}
        raise ValueError(f"unknown command: {command}")

    def _update_observation(self, request, *, action_executed: bool):
        self._model.update_observation_window(
            decode_images(request["images"]),
            request["state"],
            action_executed=action_executed,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_inventory(
    checkpoint_path: Path,
    inventory_path: Path,
    *,
    expected_tree_sha256: str,
    expected_producer_code_commit: str,
    train_config_name: str,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    inventory_path = inventory_path.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_path}")
    if not inventory_path.is_file():
        raise FileNotFoundError(f"checkpoint inventory does not exist: {inventory_path}")

    document = json.loads(inventory_path.read_text())
    if not isinstance(document, Mapping):
        raise ValueError("checkpoint inventory must be a JSON object")
    if document.get("status") != "complete":
        raise ValueError("checkpoint inventory is not complete")
    if document.get("artifact_type") != "inference_checkpoint_package":
        raise ValueError("checkpoint inventory has the wrong artifact type")
    if document.get("config_name") != train_config_name:
        raise ValueError("checkpoint inventory config does not match the server config")
    if document.get("producer_code_commit") != expected_producer_code_commit:
        raise ValueError("checkpoint inventory producer commit mismatch")

    raw_files = document.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("checkpoint inventory has no files")
    files = []
    for index, item in enumerate(raw_files):
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid checkpoint inventory entry at index {index}")
        relative = item.get("path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"invalid checkpoint path at index {index}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe checkpoint path: {relative}")
        if relative_path.as_posix() != relative:
            raise ValueError(f"checkpoint path is not normalized: {relative}")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"invalid checkpoint size: {relative}")
        valid_sha = (
            isinstance(sha256, str)
            and len(sha256) == 64
            and sha256 == sha256.lower()
            and all(character in "0123456789abcdef" for character in sha256)
        )
        if not valid_sha:
            raise ValueError(f"invalid checkpoint SHA-256: {relative}")

        actual_path = (checkpoint_path / relative_path).resolve()
        if not actual_path.is_relative_to(checkpoint_path):
            raise ValueError(f"checkpoint path escapes the checkpoint directory: {relative}")
        if not actual_path.is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {actual_path}")
        if actual_path.stat().st_size != size_bytes:
            raise ValueError(f"checkpoint file size mismatch: {relative}")
        if _sha256_file(actual_path) != sha256:
            raise ValueError(f"checkpoint file SHA-256 mismatch: {relative}")
        files.append({"path": relative, "size_bytes": size_bytes, "sha256": sha256})

    paths = [item["path"] for item in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("checkpoint inventory files must be unique and sorted")
    file_count = document.get("file_count")
    total_bytes = document.get("total_bytes")
    if file_count != len(files):
        raise ValueError("checkpoint inventory file count mismatch")
    if total_bytes != sum(item["size_bytes"] for item in files):
        raise ValueError("checkpoint inventory byte count mismatch")
    canonical = "".join(f"{item['sha256']}  {item['size_bytes']}  {item['path']}\n" for item in files).encode()
    tree_sha256 = hashlib.sha256(canonical).hexdigest()
    if document.get("tree_sha256") != tree_sha256:
        raise ValueError("checkpoint inventory tree SHA-256 mismatch")
    if tree_sha256 != expected_tree_sha256:
        raise ValueError("checkpoint tree SHA-256 does not match the launch expectation")
    return {
        "checkpoint_tree_sha256": tree_sha256,
        "checkpoint_inventory_sha256": _sha256_file(inventory_path),
        "checkpoint_file_count": file_count,
        "checkpoint_total_bytes": total_bytes,
        "checkpoint_step": document.get("checkpoint_step"),
        "producer_code_commit": document.get("producer_code_commit"),
        "config_name": document.get("config_name"),
    }


@dataclasses.dataclass
class Args:
    train_config_name: str
    checkpoint_path: Path
    checkpoint_inventory: Path
    expected_checkpoint_tree_sha256: str
    expected_producer_code_commit: str
    async_scene_context_steps: int = 0
    port: int = 8000
    pi0_step: int = 50
    sync_action_chunk_steps: int = 10
    boundary_context_steps: int = 20
    gate_sync_threshold: float = 0.5
    gate_sync_confirmations: int = 2


def main(args: Args):
    checkpoint_path = args.checkpoint_path.resolve()
    checkpoint_metadata = validate_checkpoint_inventory(
        checkpoint_path,
        args.checkpoint_inventory,
        expected_tree_sha256=args.expected_checkpoint_tree_sha256,
        expected_producer_code_commit=args.expected_producer_code_commit,
        train_config_name=args.train_config_name,
    )
    model = PI0(
        args.train_config_name,
        checkpoint_path,
        args.pi0_step,
        async_scene_context_steps=args.async_scene_context_steps,
        sync_action_chunk_steps=args.sync_action_chunk_steps,
        boundary_context_steps=args.boundary_context_steps,
        gate_sync_threshold=args.gate_sync_threshold,
        gate_sync_confirmations=args.gate_sync_confirmations,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=RobotwinPolicyService(model),
        host="0.0.0.0",
        port=args.port,
        metadata={"protocol": ROBOTWIN_POLICY_PROTOCOL, **checkpoint_metadata},
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
