from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
np: Any = None
ARM_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
ARM_ACTION_LABELS = tuple(
    [f"left_joint_{index}" for index in range(6)]
    + [f"right_joint_{index}" for index in range(6)]
)


class ConsequenceError(RuntimeError):
    def __init__(self, message: str, *, gate: str = "execution") -> None:
        super().__init__(message)
        self.gate = gate


class DiagnosticComplete(Exception):
    pass


def require(condition: bool, message: str, *, gate: str = "execution") -> None:
    if not condition:
        raise ConsequenceError(message, gate=gate)


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def runtime_identity() -> dict[str, Any]:
    configured = os.environ.get("CTR_RUNTIME")
    runtime_root = (
        Path(configured).resolve()
        if configured
        else Path(sys.executable).resolve().parent.parent
    )
    manifest = runtime_root / "runtime_manifest.json"
    identity: dict[str, Any] = {
        "root": str(runtime_root),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "manifest": str(manifest) if manifest.is_file() else None,
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "version": None,
    }
    if manifest.is_file():
        try:
            payload = load_object(manifest)
            version = payload.get("version")
            identity["version"] = version if isinstance(version, str) else None
        except (OSError, ValueError, json.JSONDecodeError, ConsequenceError):
            identity["version"] = None
    return identity


def invalid_result(
    *, gate: str, reason: str, exception_type: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "acm.action_consequence_result.v1",
        "decision": "invalid",
        "failed_gate": gate,
        "reason": reason,
        "runtime": runtime_identity(),
    }
    if exception_type is not None:
        result["exception_type"] = exception_type
    return result


def pose_array(pose: Any) -> np.ndarray:
    return np.concatenate(
        (np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64))
    )


def quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    cosine = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return 2.0 * math.acos(cosine)


def consequence_feature(task: Any) -> dict[str, list[float]]:
    left = np.asarray(task.get_arm_pose("left"), dtype=np.float64)
    right = np.asarray(task.get_arm_pose("right"), dtype=np.float64)
    object_pose = pose_array(task.object.get_pose())
    cabinet_qpos = np.asarray(task.cabinet.get_qpos(), dtype=np.float64)
    return {
        "left_ee": left.tolist(),
        "right_ee": right.tolist(),
        "object": object_pose.tolist(),
        "cabinet_qpos": cabinet_qpos.tolist(),
    }


def feature_distance(
    baseline: dict[str, list[float]],
    alternative: dict[str, list[float]],
    *,
    position_scale_m: float,
    rotation_scale_rad: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    squared = 0.0
    for name in ("left_ee", "right_ee", "object"):
        left = np.asarray(baseline[name], dtype=np.float64)
        right = np.asarray(alternative[name], dtype=np.float64)
        translation = float(np.linalg.norm(right[:3] - left[:3]))
        rotation = quaternion_angle(left[3:], right[3:])
        result[f"{name}_translation_m"] = translation
        result[f"{name}_rotation_rad"] = rotation
        squared += (translation / position_scale_m) ** 2
        squared += (rotation / rotation_scale_rad) ** 2
    cabinet = float(
        np.linalg.norm(
            np.asarray(alternative["cabinet_qpos"], dtype=np.float64)
            - np.asarray(baseline["cabinet_qpos"], dtype=np.float64)
        )
    )
    result["cabinet_qpos_l2"] = cabinet
    squared += (cabinet / rotation_scale_rad) ** 2
    result["combined"] = math.sqrt(squared)
    return result


def current_joint_vector(task: Any) -> np.ndarray:
    return np.asarray(
        task.robot.get_left_arm_jointState()
        + task.robot.get_right_arm_jointState(),
        dtype=np.float64,
    )


def run_action(task: Any, action: np.ndarray) -> None:
    task.take_action(action.tolist(), action_type="qpos")


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 1.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def top_set(values: Sequence[float], count: int = 3) -> set[int]:
    return set(np.argsort(np.asarray(values, dtype=np.float64))[-count:].tolist())


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "state indices must be comma-separated integers"
        ) from error
    if not indices or indices != sorted(set(indices)) or indices[0] < 1:
        raise argparse.ArgumentTypeError(
            "state indices must be unique, increasing, and positive"
        )
    return indices


def load_actions(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as source:
        require("joint_action/vector" in source, "HDF5 has no joint_action/vector")
        actions = np.asarray(source["joint_action/vector"][:], dtype=np.float64)
    require(actions.ndim == 2 and actions.shape[1] == 14, "expected Nx14 actions")
    require(bool(np.isfinite(actions).all()), "actions contain non-finite values")
    return actions


def load_planner_trajectory(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    require(isinstance(payload, dict), "planner trajectory must be a mapping")
    for key in ("left_joint_path", "right_joint_path"):
        paths = payload.get(key)
        require(isinstance(paths, list) and paths, f"invalid planner trajectory: {key}")
        for item in paths:
            position = item.get("position") if isinstance(item, dict) else None
            velocity = item.get("velocity") if isinstance(item, dict) else None
            require(
                isinstance(item, dict)
                and item.get("status") == "Success"
                and isinstance(position, np.ndarray)
                and isinstance(velocity, np.ndarray)
                and position.ndim == 2
                and position.shape[1] == 6
                and velocity.shape == position.shape
                and bool(np.isfinite(position).all())
                and bool(np.isfinite(velocity).all()),
                f"invalid planner path entry: {key}",
            )
    return payload


def prepare_task(
    *,
    robotwin_root: Path,
    asset_root: Path,
    overlay_root: Path,
    task_name: str,
    task_config_path: Path,
    scene_info: dict[str, Any],
    seed: int,
    episode: int,
    planner_trajectory: Path | None,
) -> Any:
    import yaml

    sys.path.insert(0, str(robotwin_root))
    overlay_assets = overlay_root / "assets"
    overlay_assets.mkdir(parents=True, exist_ok=False)
    (overlay_assets / "objects").symlink_to(
        asset_root / "objects", target_is_directory=True
    )
    source_embodiment = asset_root / "embodiments" / "aloha-agilex"
    overlay_embodiment = overlay_assets / "embodiments" / "aloha-agilex"
    overlay_embodiment.mkdir(parents=True)
    for source in source_embodiment.iterdir():
        if source.name in {"curobo_left.yml", "curobo_right.yml"}:
            continue
        (overlay_embodiment / source.name).symlink_to(
            source, target_is_directory=source.is_dir()
        )
    for arm in ("left", "right"):
        template = source_embodiment / f"curobo_{arm}_tmp.yml"
        rendered = template.read_text(encoding="utf-8").replace(
            "${ASSETS_PATH}", str(overlay_root)
        )
        (overlay_embodiment / f"curobo_{arm}.yml").write_text(
            rendered, encoding="utf-8"
        )
    os.chdir(overlay_root)
    task_module = importlib.import_module(f"envs.{task_name}")
    task = getattr(task_module, task_name)()
    with task_config_path.open("r", encoding="utf-8") as stream:
        config = yaml.load(stream.read(), Loader=yaml.FullLoader)
    require(isinstance(config, dict), "task config must be a mapping")

    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    with Path(CONFIGS_PATH, "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as stream:
        embodiment_types = yaml.load(stream.read(), Loader=yaml.FullLoader)
    embodiment = config.get("embodiment")
    require(
        isinstance(embodiment, list) and len(embodiment) in (1, 3),
        "invalid embodiment config",
    )

    def robot_file(name: str) -> str:
        value = embodiment_types[name]["file_path"]
        require(isinstance(value, str) and value, f"missing robot file: {name}")
        path = Path(value)
        if not path.is_absolute():
            path = overlay_root / path
        return str(path.resolve())

    if len(embodiment) == 1:
        config["left_robot_file"] = robot_file(embodiment[0])
        config["right_robot_file"] = robot_file(embodiment[0])
        config["dual_arm_embodied"] = True
        embodiment_name = str(embodiment[0])
    else:
        config["left_robot_file"] = robot_file(embodiment[0])
        config["right_robot_file"] = robot_file(embodiment[1])
        config["embodiment_dis"] = embodiment[2]
        config["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment[0]}+{embodiment[1]}"

    def robot_config(path: str) -> dict[str, Any]:
        with Path(path, "config.yml").open("r", encoding="utf-8") as stream:
            payload = yaml.load(stream.read(), Loader=yaml.FullLoader)
        require(isinstance(payload, dict), f"invalid robot config: {path}")
        return payload

    recorded_paths = (
        load_planner_trajectory(planner_trajectory)
        if planner_trajectory is not None
        else None
    )
    config.update(
        {
            "task_name": task_name,
            "task_config": task_config_path.stem,
            "left_embodiment_config": robot_config(config["left_robot_file"]),
            "right_embodiment_config": robot_config(config["right_robot_file"]),
            "embodiment_name": embodiment_name,
            "need_plan": recorded_paths is None,
            "render_freq": 0,
            "save_data": False,
            "collect_data": False,
            "eval_mode": False,
            "replay_scene_info": scene_info,
        }
    )
    if recorded_paths is not None:
        config.update(recorded_paths)
    task.setup_demo(now_ep_num=episode, seed=seed, **config)
    return task


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    require(
        args.task == "put_object_cabinet",
        "pilot supports put_object_cabinet only",
        gate="input",
    )
    require(
        args.robotwin_root.is_dir(),
        "RoboTwin root is missing",
        gate="environment",
    )
    require(
        (args.asset_root / "objects").is_dir(),
        "object assets are missing",
        gate="environment",
    )
    require(
        (args.asset_root / "embodiments").is_dir(),
        "embodiment assets are missing",
        gate="environment",
    )
    for arm in ("left", "right"):
        require(
            (
                args.asset_root
                / "embodiments"
                / "aloha-agilex"
                / f"curobo_{arm}_tmp.yml"
            ).is_file(),
            f"CuRobo template is missing for {arm} arm",
            gate="environment",
        )
    require(
        bool(args.asset_revision.strip()),
        "asset revision is required",
        gate="input",
    )
    robotwin_commit = git_output(args.robotwin_root, "rev-parse", "HEAD")
    require(
        not git_output(args.robotwin_root, "status", "--porcelain"),
        "RoboTwin diagnostic worktree must be clean",
        gate="environment",
    )
    for path in (
        args.task_config,
        args.episode_hdf5,
        args.scene_info,
    ):
        require(path.is_file(), f"input is missing: {path}", gate="input")
    if args.planner_trajectory is not None:
        require(
            args.planner_trajectory.is_file(),
            f"input is missing: {args.planner_trajectory}",
            gate="input",
        )
    input_sha256 = {
        "task_config": sha256_file(args.task_config),
        "episode_hdf5": sha256_file(args.episode_hdf5),
        "scene_info": sha256_file(args.scene_info),
    }
    if args.planner_trajectory is not None:
        input_sha256["planner_trajectory"] = sha256_file(args.planner_trajectory)
    return {
        "robotwin_commit": robotwin_commit,
        "asset_revision": args.asset_revision,
        "seed": args.seed,
        "input_sha256": input_sha256,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    input_receipt = validate_inputs(args)
    actions = load_actions(args.episode_hdf5)
    require(args.state_indices[-1] < len(actions), "state index is outside episode")

    scene_database = load_object(args.scene_info)
    scene_key = f"episode_{args.episode}"
    require(scene_key in scene_database, f"scene metadata is missing: {scene_key}")

    lower = np.asarray([0.25] * 6 + [0.25] + [0.25] * 6 + [0.25])
    upper = np.asarray([1.0] * 6 + [1.0] + [1.0] * 6 + [1.0])
    empirical = np.quantile(actions, 0.99, axis=0) - np.quantile(
        actions, 0.01, axis=0
    )
    effective_scale = np.clip(empirical, lower, upper)
    perturbations = args.perturbation_fraction * effective_scale

    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    max_prefix_repeat = 0.0
    max_baseline_repeat = 0.0
    max_branch_tracking_error = 0.0
    max_replay_tracking_error = 0.0
    fresh_branch_count = 0
    native_joint_frames: list[np.ndarray] = []
    native_expert_success = False

    def new_task(label: str) -> Any:
        return prepare_task(
            robotwin_root=args.robotwin_root,
            asset_root=args.asset_root,
            overlay_root=args.output_dir / "robotwin-overlays" / label,
            task_name=args.task,
            task_config_path=args.task_config,
            scene_info=scene_database[scene_key],
            seed=input_receipt["seed"],
            episode=args.episode,
            planner_trajectory=args.planner_trajectory,
        )

    def run_fresh_branch(
        *,
        state_index: int,
        native_frame_index: int,
        action: np.ndarray,
        label: str,
    ) -> dict[str, Any]:
        nonlocal fresh_branch_count
        nonlocal max_branch_tracking_error
        nonlocal max_replay_tracking_error
        task = new_task(label)
        reached_state = False
        outcome: dict[str, Any] = {}
        reference_action = actions[state_index]

        def inspect_branch_frame() -> None:
            nonlocal reached_state
            if task.FRAME_IDX == native_frame_index:
                current = current_joint_vector(task)
                replay_tracking_error = float(
                    np.max(
                        np.abs(
                            current[np.asarray(ARM_ACTION_INDICES)]
                            - reference_action[np.asarray(ARM_ACTION_INDICES)]
                        )
                    )
                )
                before = consequence_feature(task)
                run_action(task, action)
                branch_tracking_error = float(
                    np.max(
                        np.abs(
                            current_joint_vector(task)[
                                np.asarray(ARM_ACTION_INDICES)
                            ]
                            - action[np.asarray(ARM_ACTION_INDICES)]
                        )
                    )
                )
                outcome.update(
                    {
                        "before": before,
                        "after": consequence_feature(task),
                        "replay_tracking_max_abs": replay_tracking_error,
                        "branch_tracking_max_abs": branch_tracking_error,
                    }
                )
                reached_state = True
                raise DiagnosticComplete
            task.FRAME_IDX += 1

        try:
            task._take_picture = inspect_branch_frame
            try:
                task.play_once()
            except DiagnosticComplete:
                pass
        finally:
            task.close_env()
        require(reached_state, f"fresh branch did not reach state {state_index}")
        fresh_branch_count += 1
        max_replay_tracking_error = max(
            max_replay_tracking_error, outcome["replay_tracking_max_abs"]
        )
        max_branch_tracking_error = max(
            max_branch_tracking_error, outcome["branch_tracking_max_abs"]
        )
        return outcome

    def diagnose_state(state_index: int, native_frame_index: int) -> None:
        nonlocal max_prefix_repeat
        nonlocal max_baseline_repeat

        reference_action = actions[state_index].copy()
        baselines = [
            run_fresh_branch(
                state_index=state_index,
                native_frame_index=native_frame_index,
                action=reference_action,
                label=f"state-{state_index:04d}-baseline-{repeat}",
            )
            for repeat in range(2)
        ]
        baseline = baselines[0]["after"]
        reference_prefix = baselines[0]["before"]
        state_outcomes = list(baselines)
        baseline_repeat = feature_distance(
            baseline,
            baselines[1]["after"],
            position_scale_m=args.position_scale_m,
            rotation_scale_rad=args.rotation_scale_rad,
        )
        max_baseline_repeat = max(
            max_baseline_repeat, baseline_repeat["combined"]
        )
        state_prefix_repeat = feature_distance(
            reference_prefix,
            baselines[1]["before"],
            position_scale_m=args.position_scale_m,
            rotation_scale_rad=args.rotation_scale_rad,
        )["combined"]

        state_scores: list[float] = []
        for action_index, label in zip(ARM_ACTION_INDICES, ARM_ACTION_LABELS):
            signed_scores: list[float] = []
            for sign in (-1, 1):
                alternative = reference_action.copy()
                alternative[action_index] += sign * perturbations[action_index]
                outcome = run_fresh_branch(
                    state_index=state_index,
                    native_frame_index=native_frame_index,
                    action=alternative,
                    label=(
                        f"state-{state_index:04d}-action-{action_index:02d}-"
                        f"sign-{'minus' if sign < 0 else 'plus'}"
                    ),
                )
                state_outcomes.append(outcome)
                prefix_repeat = feature_distance(
                    reference_prefix,
                    outcome["before"],
                    position_scale_m=args.position_scale_m,
                    rotation_scale_rad=args.rotation_scale_rad,
                )["combined"]
                state_prefix_repeat = max(state_prefix_repeat, prefix_repeat)
                consequence = feature_distance(
                    baseline,
                    outcome["after"],
                    position_scale_m=args.position_scale_m,
                    rotation_scale_rad=args.rotation_scale_rad,
                )
                normalized_error = abs(
                    alternative[action_index] - reference_action[action_index]
                ) / effective_scale[action_index]
                rows.append(
                    {
                        "state_index": state_index,
                        "native_frame_index": native_frame_index,
                        "action_index": action_index,
                        "action_label": label,
                        "sign": sign,
                        "reference_action": float(reference_action[action_index]),
                        "perturbed_action": float(alternative[action_index]),
                        "effective_scale": float(effective_scale[action_index]),
                        "normalized_action_error": float(normalized_error),
                        "replay_tracking_max_abs": outcome[
                            "replay_tracking_max_abs"
                        ],
                        "branch_tracking_max_abs": outcome[
                            "branch_tracking_max_abs"
                        ],
                        "prefix_repeat_combined": prefix_repeat,
                        "consequence": consequence,
                    }
                )
                signed_scores.append(consequence["combined"])
            state_scores.append(float(np.sqrt(np.mean(np.square(signed_scores)))))

        max_prefix_repeat = max(max_prefix_repeat, state_prefix_repeat)

        p10, p90 = np.quantile(state_scores, (0.1, 0.9))
        states.append(
            {
                "state_index": state_index,
                "native_frame_index": native_frame_index,
                "replay_tracking_max_abs": max(
                    outcome["replay_tracking_max_abs"]
                    for outcome in state_outcomes
                ),
                "branch_tracking_max_abs": max(
                    outcome["branch_tracking_max_abs"]
                    for outcome in state_outcomes
                ),
                "prefix_repeat_combined": state_prefix_repeat,
                "baseline_repeat": baseline_repeat,
                "joint_scores": dict(zip(ARM_ACTION_LABELS, state_scores)),
                "heterogeneity_p90_p10_ratio": float(p90 / max(p10, 1e-12)),
                "top3": [
                    ARM_ACTION_LABELS[index]
                    for index in sorted(top_set(state_scores))
                ],
            }
        )

    task = new_task("native-validation")

    def inspect_native_frame() -> None:
        native_joint_frames.append(current_joint_vector(task))
        task.FRAME_IDX += 1

    try:
        task._take_picture = inspect_native_frame
        task.play_once()
        native_expert_success = bool(task.plan_success and task.check_success())
    finally:
        task.close_env()

    raw_native_frame_count = len(native_joint_frames)
    aligned_native_frame_count = 0
    native_frame_alignment = "unmatched"
    hdf5_to_native_frame_indices: list[int] = []
    dropped_native_frame_index: int | None = None
    max_native_frame_tracking_error: float | None = None
    second_best_native_frame_tracking_error: float | None = None
    terminal_callback_delta: float | None = None
    arm_indices = np.asarray(ARM_ACTION_INDICES)

    def native_tracking_error(indices: list[int]) -> float:
        aligned_native_frames = np.asarray(
            [native_joint_frames[index] for index in indices]
        )
        return float(
            np.max(
                np.abs(
                    aligned_native_frames[:, arm_indices]
                    - actions[:, arm_indices]
                )
            )
        )

    if raw_native_frame_count == len(actions):
        hdf5_to_native_frame_indices = list(range(len(actions)))
        max_native_frame_tracking_error = native_tracking_error(
            hdf5_to_native_frame_indices
        )
        native_frame_alignment = "direct"
    elif raw_native_frame_count == len(actions) + 1:
        candidates: list[tuple[float, int, list[int]]] = []
        for dropped_index in range(raw_native_frame_count):
            indices = [
                index
                for index in range(raw_native_frame_count)
                if index != dropped_index
            ]
            candidates.append(
                (native_tracking_error(indices), dropped_index, indices)
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        (
            max_native_frame_tracking_error,
            dropped_native_frame_index,
            hdf5_to_native_frame_indices,
        ) = candidates[0]
        second_best_native_frame_tracking_error = candidates[1][0]
        native_frame_alignment = "one_extra_callback"
        terminal_callback_delta = float(
            np.max(np.abs(native_joint_frames[-1] - native_joint_frames[-2]))
        )
    aligned_native_frame_count = len(hdf5_to_native_frame_indices)

    for state_index in args.state_indices:
        native_frame_index = (
            hdf5_to_native_frame_indices[state_index]
            if len(hdf5_to_native_frame_indices) == len(actions)
            else state_index
        )
        diagnose_state(state_index, native_frame_index)

    require(
        [state["state_index"] for state in states] == args.state_indices,
        "native expert did not reach every selected state",
    )

    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(states):
        left_scores = list(left["joint_scores"].values())
        for right in states[left_index + 1 :]:
            right_scores = list(right["joint_scores"].values())
            pairwise.append(
                {
                    "left_state_index": left["state_index"],
                    "right_state_index": right["state_index"],
                    "spearman": spearman(left_scores, right_scores),
                    "top3_jaccard": jaccard(
                        top_set(left_scores), top_set(right_scores)
                    ),
                }
            )

    heterogeneity_state_count = sum(
        state["heterogeneity_p90_p10_ratio"] >= args.heterogeneity_ratio_threshold
        for state in states
    )
    minimum_required_states = math.ceil(len(states) / 2)
    min_spearman = (
        min(item["spearman"] for item in pairwise) if pairwise else None
    )
    min_top3_jaccard = (
        min(item["top3_jaccard"] for item in pairwise) if pairwise else None
    )
    validity = {
        "native_expert_success": native_expert_success,
        "native_frame_count": aligned_native_frame_count == len(actions),
        "native_frame_tracking": (
            max_native_frame_tracking_error is not None
            and max_native_frame_tracking_error <= args.tracking_tolerance
        ),
        "prefix_repeat": max_prefix_repeat <= args.prefix_repeat_tolerance,
        "baseline_repeat": max_baseline_repeat <= args.repeat_tolerance,
        "replay_tracking": max_replay_tracking_error <= args.tracking_tolerance,
        "branch_tracking": (
            max_branch_tracking_error <= args.branch_tracking_tolerance
        ),
    }
    heterogeneity_supported = heterogeneity_state_count >= minimum_required_states
    state_dependence_supported = bool(pairwise) and (
        min_spearman is not None
        and min_top3_jaccard is not None
        and (
            min_spearman <= args.spearman_threshold
            or min_top3_jaccard <= args.top3_jaccard_threshold
        )
    )
    valid = all(validity.values())
    supported = valid and heterogeneity_supported and state_dependence_supported
    decision = "support" if supported else "invalid" if not valid else "inconclusive"
    metrics = {
        "decision": decision,
        "valid": valid,
        "validity_gates": validity,
        "native_expert_success": native_expert_success,
        "generated_frame_count": raw_native_frame_count,
        "aligned_frame_count": aligned_native_frame_count,
        "expected_frame_count": len(actions),
        "native_frame_alignment": native_frame_alignment,
        "dropped_native_frame_index": dropped_native_frame_index,
        "max_native_frame_tracking_error": max_native_frame_tracking_error,
        "second_best_native_frame_tracking_error": (
            second_best_native_frame_tracking_error
        ),
        "terminal_callback_delta": terminal_callback_delta,
        "fresh_environment_branch_count": fresh_branch_count,
        "max_prefix_repeat_combined": max_prefix_repeat,
        "max_baseline_repeat_combined": max_baseline_repeat,
        "max_replay_tracking_error": max_replay_tracking_error,
        "max_branch_tracking_error": max_branch_tracking_error,
        "heterogeneity_state_count": heterogeneity_state_count,
        "heterogeneity_required_state_count": minimum_required_states,
        "minimum_pairwise_spearman": min_spearman,
        "minimum_top3_jaccard": min_top3_jaccard,
        "heterogeneity_supported": heterogeneity_supported,
        "state_dependence_supported": state_dependence_supported,
    }
    limitations = [
        "This is a one-episode feasibility pilot, not a task-level or cross-task confirmation.",
        (
            "The supplied planner trajectory is identified by SHA-256; its provenance link to the validated HDF5 must be qualified independently."
            if args.planner_trajectory is not None
            else "The native expert is regenerated from the recorded seed and scene identity because no planner trajectory was supplied."
        ),
        "Native success is checked on an unmodified replay; every baseline and intervention uses a fresh simulator and an independent regenerated expert prefix.",
        "The pilot probes one controller action horizon and cannot establish long-horizon success effects.",
        "Gripper event dimensions and off-diagonal joint directions are intentionally not tested.",
        "The combined consequence uses preregistered 1 cm and 5 degree feature scales.",
    ]
    receipt = {
        "schema": "acm.action_consequence_diagnostic.v1",
        "task": args.task,
        "episode": args.episode,
        "seed": input_receipt["seed"],
        "code_commit": git_output(args.repo_root, "rev-parse", "HEAD"),
        "robotwin_commit": input_receipt["robotwin_commit"],
        "runtime": runtime_identity(),
        "assets": {
            "root": str(args.asset_root),
            "revision": input_receipt["asset_revision"],
        },
        "inputs": {
            "task_config": str(args.task_config),
            "episode_hdf5": str(args.episode_hdf5),
            "scene_info": str(args.scene_info),
            "planner_trajectory": (
                str(args.planner_trajectory)
                if args.planner_trajectory is not None
                else None
            ),
            "sha256": input_receipt["input_sha256"],
        },
        "protocol": {
            "state_source": (
                "fresh seeded native expert transcript replay for every baseline and intervention at HDF5 save frames"
                if args.planner_trajectory is not None
                else "fresh seeded native expert replanning for every baseline and intervention at HDF5 save frames"
            ),
            "native_validation": (
                "separate unmodified seeded expert transcript replay"
                if args.planner_trajectory is not None
                else "separate unmodified seeded expert replanning replay"
            ),
            "native_frame_alignment_rule": (
                "accept exactly N callbacks, or N+1 after dropping exactly one callback "
                "by deterministic minimum-max-error one-to-one alignment; all N aligned "
                "arm-joint frames must satisfy the replay tracking tolerance"
            ),
            "hdf5_to_native_frame_indices": hdf5_to_native_frame_indices,
            "branch_isolation": (
                "one newly constructed simulator per baseline or intervention; "
                "no scene snapshot restoration"
            ),
            "action_dimensions": list(ARM_ACTION_INDICES),
            "action_labels": list(ARM_ACTION_LABELS),
            "excluded_dimensions": {
                "6": "left gripper is an event actuator and may saturate",
                "13": "right gripper is an event actuator and may saturate",
            },
            "state_indices": args.state_indices,
            "perturbation_fraction": args.perturbation_fraction,
            "empirical_q99_minus_q01": empirical.tolist(),
            "effective_scale_clip": {"lower": lower.tolist(), "upper": upper.tolist()},
            "effective_scale": effective_scale.tolist(),
            "position_scale_m": args.position_scale_m,
            "rotation_scale_rad": args.rotation_scale_rad,
            "thresholds": {
                "heterogeneity_p90_p10_ratio": args.heterogeneity_ratio_threshold,
                "spearman": args.spearman_threshold,
                "top3_jaccard": args.top3_jaccard_threshold,
                "prefix_repeat_tolerance": args.prefix_repeat_tolerance,
                "repeat_tolerance": args.repeat_tolerance,
                "tracking_tolerance": args.tracking_tolerance,
                "branch_tracking_tolerance": args.branch_tracking_tolerance,
            },
        },
        "metrics": metrics,
        "states": states,
        "pairwise_states": pairwise,
        "perturbations": rows,
        "limitations": limitations,
    }
    receipt.update(
        {
            "schema": "acm.action_consequence_result.v1",
            "decision": decision,
            "failed_gate": (
                next(name for name, passed in validity.items() if not passed)
                if not valid
                else None
            ),
            "summary": (
                "The native counterfactual pilot supports H1 under its registered gates."
                if supported
                else "The native counterfactual pilot was invalid under its fidelity gates."
                if not valid
                else "The valid pilot provides limited evidence for H1."
            ),
        }
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a native RoboTwin local action-consequence diagnostic"
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=os.environ.get("CTR_ROBOTWIN_ROOT"),
        required=os.environ.get("CTR_ROBOTWIN_ROOT") is None,
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--asset-revision", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--episode-hdf5", type=Path, required=True)
    parser.add_argument("--scene-info", type=Path, required=True)
    parser.add_argument("--planner-trajectory", type=Path)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--state-indices", type=parse_indices, required=True)
    parser.add_argument("--perturbation-fraction", type=float, default=0.05)
    parser.add_argument("--position-scale-m", type=float, default=0.01)
    parser.add_argument(
        "--rotation-scale-rad", type=float, default=math.radians(5.0)
    )
    parser.add_argument("--heterogeneity-ratio-threshold", type=float, default=3.0)
    parser.add_argument("--spearman-threshold", type=float, default=0.8)
    parser.add_argument("--top3-jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--prefix-repeat-tolerance", type=float, default=1e-3)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-3)
    parser.add_argument("--tracking-tolerance", type=float, default=0.1)
    parser.add_argument("--branch-tracking-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get("CTR_RUN_DIRECTORY"),
        required=os.environ.get("CTR_RUN_DIRECTORY") is None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global np
    args = build_parser().parse_args(argv)
    for name in (
        "repo_root",
        "robotwin_root",
        "asset_root",
        "task_config",
        "episode_hdf5",
        "scene_info",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.planner_trajectory is not None:
        args.planner_trajectory = args.planner_trajectory.resolve()
    try:
        import numpy as numpy_module

        np = numpy_module
    except Exception as error:
        result = invalid_result(
            gate="environment",
            reason=str(error),
            exception_type=type(error).__name__,
        )
        atomic_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    try:
        require(
            0 < args.perturbation_fraction <= 0.1,
            "invalid perturbation fraction",
            gate="input",
        )
        require(
            args.position_scale_m > 0,
            "position scale must be positive",
            gate="input",
        )
        require(
            args.rotation_scale_rad > 0,
            "rotation scale must be positive",
            gate="input",
        )
        require(args.seed >= 0, "seed must be nonnegative", gate="input")
        for name in (
            "prefix_repeat_tolerance",
            "repeat_tolerance",
            "tracking_tolerance",
            "branch_tracking_tolerance",
        ):
            require(
                getattr(args, name) > 0,
                f"{name} must be positive",
                gate="input",
            )
        result = execute(args)
    except ConsequenceError as error:
        result = invalid_result(gate=error.gate, reason=str(error))
    except Exception as error:
        result = invalid_result(
            gate="execution",
            reason=str(error),
            exception_type=type(error).__name__,
        )
    atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
