from __future__ import annotations

import sys
from pathlib import Path


_PI05_ROOT = Path(__file__).resolve().parents[1] / "pi05"
sys.path.insert(0, str(_PI05_ROOT))

from robotwin_remote_model import RobotwinRemoteModel  # noqa: E402


def encode_obs(observation):
    images = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    return images, observation["joint_action"]["vector"]


def get_model(usr_args):
    port = int(usr_args["remote_policy_port"])
    if port < 1:
        raise ValueError("CASM requires a separately managed policy server")
    return RobotwinRemoteModel(
        host=str(usr_args.get("remote_policy_host", "127.0.0.1")),
        port=port,
    )


def eval(task_env, model, observation):
    if model.observation_window is None:
        model.set_language(task_env.get_instruction())
    images, state = encode_obs(observation)
    model.update_observation_window(images, state, action_executed=False)
    actions = model.get_action()[: model.execution_steps()]
    model.record_chunk(len(actions))
    for action in actions:
        model.record_action(action, state)
        task_env.take_action(action)
        images, state = encode_obs(task_env.get_obs())
        model.update_observation_window(images, state, action_executed=True)


def reset_model(model):
    model.reset_obsrvationwindows()
