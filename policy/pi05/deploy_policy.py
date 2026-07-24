import time

from .pi_model import PI0


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    pi0_step = int(usr_args["pi0_step"])
    async_steps = int(usr_args.get("async_scene_context_steps", 0))
    server_host = str(usr_args["policy_server_host"])
    server_port = int(usr_args["policy_server_port"])
    boundary_observation = bool(usr_args.get("pi05_boundary_observation", False))
    return PI0(
        server_host,
        server_port,
        pi0_step,
        async_steps,
        boundary_observation,
        visual_phase_gate=bool(usr_args.get("visual_phase_gate", False)),
        sync_action_chunk_steps=int(usr_args.get("sync_action_chunk_steps", 10)),
        gate_sync_threshold=float(usr_args.get("gate_sync_threshold", 0.5)),
        gate_sync_confirmations=int(usr_args.get("gate_sync_confirmations", 2)),
    )


def eval(task_env, model, observation):

    if model.observation_window is None:
        instruction = task_env.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[: model.execution_steps()]
    model.record_chunk(len(actions))

    for action in actions:
        previous_step = task_env.take_action_cnt
        action_started = time.perf_counter()
        task_env.take_action(action)
        model.profile_action_s += time.perf_counter() - action_started
        model.profile_action_calls += 1
        if task_env.take_action_cnt == previous_step:
            break
        model.record_action(action, input_state)
        model.advance_after_action()
        if model.boundary_observation:
            if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                break
            continue
        observation = task_env.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
