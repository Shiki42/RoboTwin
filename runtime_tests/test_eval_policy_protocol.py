from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import yaml


def load_eval_policy(monkeypatch, tmp_path: Path):
    envs = ModuleType("envs")
    envs.CONFIGS_PATH = f"{tmp_path / 'env_configs'}/"
    env_utils = ModuleType("envs.utils")
    create_actor = ModuleType("envs.utils.create_actor")

    class UnStableError(Exception):
        pass

    create_actor.UnStableError = UnStableError
    instructions = ModuleType("generate_episode_instructions")
    instructions.__all__ = []

    monkeypatch.setitem(sys.modules, "envs", envs)
    monkeypatch.setitem(sys.modules, "envs.utils", env_utils)
    monkeypatch.setitem(sys.modules, "envs.utils.create_actor", create_actor)
    monkeypatch.setitem(sys.modules, "generate_episode_instructions", instructions)

    module_path = Path(__file__).parents[1] / "script/eval_policy.py"
    spec = importlib.util.spec_from_file_location("eval_policy_protocol_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_main_config(tmp_path: Path) -> None:
    task_config = tmp_path / "task_config"
    env_configs = tmp_path / "env_configs"
    task_config.mkdir()
    env_configs.mkdir()
    (task_config / "canary.yml").write_text(
        yaml.safe_dump(
            {
                "embodiment": ["dual"],
                "camera": {
                    "head_camera_type": "head",
                    "collect_head_camera": True,
                    "wrist_camera_type": "wrist",
                    "collect_wrist_camera": False,
                },
                "domain_randomization": {
                    "cluttered_table": False,
                    "random_background": False,
                    "random_light": False,
                    "random_table_height": False,
                    "random_head_camera_dis": False,
                },
                "eval_video_log": False,
                "render_freq": 0,
                "clear_cache_freq": 8,
            }
        )
    )
    (env_configs / "_embodiment_config.yml").write_text(
        yaml.safe_dump({"dual": {"file_path": "/fake/dual"}})
    )
    (env_configs / "_camera_config.yml").write_text(
        yaml.safe_dump({"head": {"h": 64, "w": 64}})
    )


def test_main_honors_single_fixed_seed_without_expert_filter(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_eval_policy(monkeypatch, tmp_path)
    write_main_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    seed_table = tmp_path / "seeds.json"
    seed_table.write_text(
        json.dumps(
            {
                "schema_version": "parallelvla.robotwin_seed_table.v1",
                "task_name": "scan_object",
                "task_config": "canary",
                "seeds": [100113],
            }
        )
    )
    seed_sha256 = hashlib.sha256(seed_table.read_bytes()).hexdigest()
    captured = {}

    monkeypatch.setattr(module, "get_embodiment_config", lambda _: {
        "arm_joints_name": [["left_joint"], ["right_joint"]]
    })
    monkeypatch.setattr(module, "class_decorator", lambda _: object())
    monkeypatch.setattr(
        module,
        "eval_function_decorator",
        lambda policy_name, function_name: (lambda _: object()),
    )

    def fake_eval_policy(task_name, task, args, model, st_seed, **kwargs):
        captured.update(
            task_name=task_name,
            args=args,
            st_seed=st_seed,
            kwargs=kwargs,
        )
        return st_seed + 1, 0

    monkeypatch.setattr(module, "eval_policy", fake_eval_policy)
    module.main(
        {
            "task_name": "scan_object",
            "task_config": "canary",
            "ckpt_setting": "checkpoint",
            "policy_name": "DP3",
            "instruction_type": "seen",
            "seed_table_path": str(seed_table),
            "seed_table_sha256": seed_sha256,
            "seed_table_index": 0,
            "requested_seed": 100113,
            "test_num": 1,
            "expert_check": False,
        }
    )

    assert captured["st_seed"] == 100113
    assert captured["kwargs"]["test_num"] == 1
    assert captured["kwargs"]["start_episode_index"] == 0
    assert captured["args"]["expert_check"] is False
    assert captured["args"]["requested_seed"] == 100113


class FakeTask:
    def __init__(self) -> None:
        self.setup_seeds = []
        self.prepare_calls = 0
        self.eval_video_path = None
        self.eval_success = False
        self.render_freq = 0
        self.viewer = SimpleNamespace(close=lambda: None)

    def setup_demo(self, now_ep_num, seed, is_test, **args) -> None:
        self.setup_seeds.append(seed)
        self.take_action_cnt = 0
        self.step_lim = 99
        self.render_freq = args["render_freq"]

    def prepare_episode_metadata(self):
        self.prepare_calls += 1
        return {"info": {"{A}": "object", "{B}": "target"}}

    def play_once(self):
        raise AssertionError("expert planner must not run")

    def set_instruction(self, instruction) -> None:
        self.instruction = instruction

    def get_obs(self):
        return {"observation": True}

    def close_env(self, clear_cache=False) -> None:
        self.clear_cache = clear_cache


class FakeModel:
    inference_index = 2

    def __init__(self) -> None:
        self.events = []

    def set_episode_seed(self, seed) -> None:
        self.events.append("seed")
        self.seed = seed

    def episode_action_sha256(self):
        return "action-digest"


def test_policy_failure_emits_one_structured_metric_for_one_seed(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_eval_policy(monkeypatch, tmp_path)
    task = FakeTask()
    model = FakeModel()
    metrics_path = tmp_path / "rollout_metrics.jsonl"

    def fake_policy_step(task, model, observation) -> None:
        task.take_action_cnt += 1

    monkeypatch.setattr(
        module,
        "eval_function_decorator",
        lambda policy_name, function_name: (
            fake_policy_step
            if function_name == "eval"
            else lambda model: model.events.append("reset")
        ),
    )
    monkeypatch.setattr(
        module,
        "generate_episode_descriptions",
        lambda task_name, episode_info, test_num: [{"seen": ["do task"]}],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "InterarmContactMonitor",
        lambda task: SimpleNamespace(
            summary=lambda: {
                "collision": False,
                "interarm_contact_simulation_steps": 0,
                "first_contact_simulation_step": None,
                "contact_policy_step_indices": [],
                "monitored_simulation_steps": 4,
            },
            episode_action_sha256=lambda: "a" * 64,
            action_count=2,
        ),
    )

    next_seed, successes = module.eval_policy(
        "scan_object",
        task,
        {
            "task_name": "scan_object",
            "policy_name": "DP3",
            "task_config": "canary",
            "ckpt_setting": "checkpoint",
            "expert_check": False,
            "render_freq": 0,
            "clear_cache_freq": 8,
            "max_episode_steps": 2,
            "policy_rng": "episode_addressable",
            "metrics_output": str(metrics_path),
        },
        model,
        100113,
        test_num=1,
        instruction_type="seen",
    )

    assert (next_seed, successes) == (100114, 0)
    assert task.setup_seeds == [100113]
    assert task.prepare_calls == 1
    assert model.events == ["reset", "seed"]
    assert model.seed == 100113
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0] == {
        "task_name": "scan_object",
        "policy_name": "DP3",
        "task_config": "canary",
        "ckpt_setting": "checkpoint",
        "seed": 100113,
        "scene_seed": 100113,
        "requested_seed": 100113,
        "policy_seed": 100113,
        "policy_rng": "episode_addressable",
        "policy_action_sha256": "a" * 64,
        "policy_action_count": 2,
        "policy_inference_requests": 2,
        "episode_index": 0,
        "success": False,
        "episode_steps": 2,
        "max_episode_steps": 2,
        "elapsed_sec": records[0]["elapsed_sec"],
        "collision": False,
        "collision_summary": {
            "collision": False,
            "interarm_contact_simulation_steps": 0,
            "first_contact_simulation_step": None,
            "contact_policy_step_indices": [],
            "monitored_simulation_steps": 4,
        },
        "instruction": "do task",
    }
