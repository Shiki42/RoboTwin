from pathlib import Path
import sys

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


def get_model(usr_args):
    remote_port = int(usr_args.get("remote_policy_port", 0))
    if remote_port:
        from robotwin_remote_model import RobotwinRemoteModel

        return RobotwinRemoteModel(
            host=usr_args.get("remote_policy_host", "127.0.0.1"),
            port=remote_port,
        )

    from pi_model import PI0

    return PI0(
        usr_args["train_config_name"],
        usr_args["checkpoint_path"],
        usr_args["pi0_step"],
        async_scene_context_steps=int(usr_args.get("async_scene_context_steps", 0)),
        sync_action_chunk_steps=int(usr_args.get("sync_action_chunk_steps", 10)),
        boundary_context_steps=int(usr_args.get("boundary_context_steps", 20)),
        gate_sync_threshold=float(usr_args.get("gate_sync_threshold", 0.5)),
        gate_sync_confirmations=int(usr_args.get("gate_sync_confirmations", 2)),
    )


def eval(TASK_ENV, model, observation):  # noqa: N803
    if model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state, action_executed=False)
    actions = model.get_action()[: model.execution_steps()]
    model.record_chunk(len(actions))

    for action in actions:
        model.record_action(action, input_state)
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state, action_executed=True)


def reset_model(model):
    model.reset_obsrvationwindows()
