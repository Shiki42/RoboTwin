from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _action_digest(actions: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(actions)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _observation_digest(images: list[np.ndarray], state: np.ndarray, instruction: str) -> str:
    digest = hashlib.sha256()
    for image in images:
        contiguous = np.ascontiguousarray(image)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    contiguous_state = np.ascontiguousarray(state)
    digest.update(np.asarray(contiguous_state.shape, dtype=np.int64).tobytes())
    digest.update(contiguous_state.tobytes())
    digest.update(instruction.encode())
    return digest.hexdigest()


def _native_image(value) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected CHW RGB image, got {image.shape}")
    image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError("dataset image contains non-finite values")
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    return image.astype(np.uint8)


def _rpc(connection, packer, request: dict, *, timeout: float) -> dict:
    connection.send(packer.pack(request))
    response = connection.recv(timeout=timeout)
    if isinstance(response, str):
        raise RuntimeError(f"policy server returned an error: {response}")
    unpacked = msgpack_numpy.unpackb(response)
    if not isinstance(unpacked, dict):
        raise ValueError(f"policy server returned {type(unpacked).__name__}, not a dict")
    return unpacked


def _atomic_write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"probe output already exists: {path}")
    with temporary.open("x") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--expected-tree-sha256", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--server-code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(
        args.dataset_repo,
        root=args.dataset_root,
        video_backend="pyav",
    )
    sample = dataset[args.dataset_index]
    images = [
        _native_image(sample["observation.images.cam_high"]),
        _native_image(sample["observation.images.cam_right_wrist"]),
        _native_image(sample["observation.images.cam_left_wrist"]),
    ]
    state = np.asarray(sample["observation.state"], dtype=np.float32)
    instruction = str(sample["task"])
    if state.shape != (14,) or not np.all(np.isfinite(state)):
        raise ValueError(f"expected finite 14-DoF state, got {state.shape}")

    packer = msgpack_numpy.Packer()
    started = time.monotonic()
    with websockets.sync.client.connect(
        f"ws://{args.host}:{args.port}",
        compression=None,
        max_size=None,
    ) as connection:
        metadata = msgpack_numpy.unpackb(connection.recv(timeout=30))
        if metadata.get("protocol") != "robotwin_pi0_v2":
            raise ValueError(f"unexpected server protocol: {metadata}")
        if metadata.get("checkpoint_tree_sha256") != args.expected_tree_sha256:
            raise ValueError("server checkpoint tree SHA-256 mismatch")
        if metadata.get("checkpoint_inventory_sha256") != args.expected_inventory_sha256:
            raise ValueError("server checkpoint inventory SHA-256 mismatch")
        seed_response = _rpc(
            connection,
            packer,
            {"command": "seed", "seed": args.policy_seed},
            timeout=args.timeout_sec,
        )
        reset_response = _rpc(
            connection,
            packer,
            {"command": "reset"},
            timeout=args.timeout_sec,
        )
        response = _rpc(
            connection,
            packer,
            {
                "command": "infer",
                "images": images,
                "state": state,
                "instruction": instruction,
            },
            timeout=args.timeout_sec,
        )
        metrics_response = _rpc(
            connection,
            packer,
            {"command": "metrics"},
            timeout=args.timeout_sec,
        )
    elapsed = time.monotonic() - started

    if seed_response != {"ok": True} or reset_response != {"ok": True}:
        raise ValueError("policy server rejected seed or reset RPC")
    actions = np.asarray(response["actions"])
    if actions.shape != (10, 14) or not np.all(np.isfinite(actions)):
        raise ValueError(f"expected finite (10, 14) actions, got {actions.shape}")
    action_sha256 = _action_digest(actions)
    metrics = metrics_response["metrics"]
    if metrics.get("server_policy_action_sha256") != action_sha256:
        raise ValueError("server and client action digests differ")
    if metrics.get("server_policy_inference_requests") != 1:
        raise ValueError("server inference request count is not one")

    script_path = Path(__file__).resolve()
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "server": {
            "host": args.host,
            "port": args.port,
            "metadata": metadata,
            "code_commit": args.server_code_commit,
        },
        "dataset": {
            "repo": args.dataset_repo,
            "revision": args.dataset_revision,
            "root": str(args.dataset_root.resolve()),
            "index": args.dataset_index,
        },
        "observation": {
            "sha256": _observation_digest(images, state, instruction),
            "image_shapes": [list(image.shape) for image in images],
            "state_shape": list(state.shape),
            "instruction": instruction,
        },
        "inference": {
            "policy_seed": args.policy_seed,
            "action_shape": list(actions.shape),
            "action_dtype": str(actions.dtype),
            "action_min": float(actions.min()),
            "action_max": float(actions.max()),
            "action_sha256": action_sha256,
            "requests": 1,
            "elapsed_sec": elapsed,
            "server_metrics": metrics,
        },
        "probe_script": {"path": str(script_path), "sha256": _sha256_file(script_path)},
    }
    _atomic_write(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
