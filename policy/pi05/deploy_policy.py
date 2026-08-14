import os
from pathlib import Path
import sys


import numpy as np

_CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_CURRENT_DIR))


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def encode_state(task_env):
    left = task_env.robot.get_left_arm_jointState()
    right = task_env.robot.get_right_arm_jointState()
    return np.asarray(left + right)


def get_model(usr_args):
    remote_port = int(usr_args.get("remote_policy_port", 0))
    if remote_port:
        from robotwin_remote_model import RobotwinRemoteModel

        if "UPVLA_RECOVERY_CHECKPOINT" in os.environ:
            from precision_adapter.deployment.robotwin_recovery import RecoveryPi05Model

            return RecoveryPi05Model.from_environment(usr_args)

        return RobotwinRemoteModel(
            host=usr_args.get("remote_policy_host", "127.0.0.1"),
            port=remote_port,
        )

    from pi_model import PI0

    checkpoint_dir = (
        _CURRENT_DIR
        / "checkpoints"
        / usr_args["train_config_name"]
        / usr_args["model_name"]
        / str(usr_args["checkpoint_id"])
    )
    return PI0(
        usr_args["train_config_name"],
        checkpoint_dir,
        usr_args["pi0_step"],
        async_phase_steps=int(usr_args.get("async_phase_steps", 0)),
        sync_action_chunk_steps=int(usr_args.get("sync_action_chunk_steps", 10)),
        phase_prompt_conditioning=bool(int(usr_args.get("phase_prompt_conditioning", 1))),
        boundary_phase_steps=int(usr_args.get("boundary_phase_steps", 20)),
    )


def eval(TASK_ENV, model, observation):  # noqa: N803
    if model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state, action_executed=False)
    actions = model.get_action()[: model.execution_steps()]
    executed_actions = 0
    if hasattr(model, "correct_actions"):
        actions = model.correct_actions(TASK_ENV, observation, actions)

    for action in actions:
        previous_step = TASK_ENV.take_action_cnt
        TASK_ENV.take_action(action)
        if TASK_ENV.take_action_cnt == previous_step:
            break
        model.record_action(action, input_state)
        model.advance_after_action()
        input_state = encode_state(TASK_ENV)
        executed_actions += 1
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break

    model.record_chunk(executed_actions)


def reset_model(model):
    model.reset_obsrvationwindows()
