import ast
import sys
import os
import gc
import subprocess
import json
import time

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *
from pi05.eval_instrumentation import InterarmContactMonitor, apply_episode_step_limit

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def compose_instruction(instruction, prefix=""):
    instruction = str(instruction).strip()
    prefix = str(prefix or "").strip()
    if not prefix:
        return instruction
    return f"{prefix} {instruction}"


def load_persistent_episode_schedule(path):
    if path is None:
        return None
    payload = json.loads(Path(path).resolve().read_text())
    if payload.get("schema") != "parallelvla.robotwin_persistent_episode_schedule.v1":
        raise ValueError("unsupported persistent episode schedule schema")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("persistent episode schedule must contain episodes")
    normalized = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        seed = int(row["seed"])
        policy_seed = int(row.get("policy_seed", seed))
        eval_run_id = str(row["eval_run_id"])
        instruction = row.get("instruction")
        if min(episode_index, seed, policy_seed) < 0 or not eval_run_id:
            raise ValueError(f"invalid persistent episode row: {row}")
        if instruction is not None and not str(instruction).strip():
            raise ValueError(f"empty persistent instruction: {row}")
        normalized.append({
            "episode_index": episode_index,
            "seed": seed,
            "policy_seed": policy_seed,
            "eval_run_id": eval_run_id,
            "instruction": str(instruction).strip() if instruction is not None else None,
        })
    for key in ("episode_index", "seed", "eval_run_id"):
        values = [row[key] for row in normalized]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {key} in persistent episode schedule")
    return normalized


def release_expert_motion_planners(task_env):
    """Release CuRobo planners that a qpos policy rollout cannot use."""
    robot = getattr(task_env, "robot", None)
    if robot is None:
        raise RuntimeError("cannot release expert planners before the robot is loaded")
    if bool(getattr(robot, "communication_flag", False)):
        raise RuntimeError("subprocess-backed expert planners cannot be released in-process")

    planner_names = [
        name
        for name in ("left_planner", "right_planner")
        if getattr(robot, name, None) is not None
    ]
    if set(planner_names) != {"left_planner", "right_planner"}:
        raise RuntimeError(f"expected two in-process expert planners, found {planner_names}")

    import torch

    reserved_before = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    for name in planner_names:
        setattr(robot, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reserved_after = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    print(
        "Released unused CuRobo expert planners: "
        f"torch_reserved_mib={reserved_before / 2**20:.1f}->{reserved_after / 2**20:.1f}"
    )


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def eval_function_decorator(policy_name, model_name):
    policy_model = importlib.import_module(policy_name)
    return getattr(policy_model, model_name)

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    if bool(usr_args.get("prune_unused_cameras", False)):
        from evostudio_benchmark_sdk.robotwin.runtime_optimizations import install_camera_pruning

        install_camera_pruning()
        print("Installed EvoStudio unused-camera pruning")
    if bool(usr_args.get("reuse_renderer", False)):
        from evostudio_benchmark_sdk.robotwin.runtime_optimizations import install_renderer_reuse

        install_renderer_reuse()
        print("Installed EvoStudio Engine/Renderer reuse")
    if bool(usr_args.get("pi05_control_program_fast_path", False)):
        from evostudio_benchmark_sdk.robotwin.runtime_optimizations import (
            install_control_program_fast_path,
        )

        install_control_program_fast_path(
            int(usr_args.get("pi05_control_hz", 250))
        )
        print("Installed EvoStudio Pi0.5 control-program fast path")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_run_id = str(usr_args.get("eval_run_id") or current_time)
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if bool(usr_args.get("pi05_minimal_observation", False)):
        args["data_type"]["pointcloud"] = False
        args["data_type"]["endpose"] = False
        print("Enabled Pi0.5 RGB+qpos-only observation")

    for runtime_key in (
        "eval_video_log",
        "render_freq",
        "clear_cache_freq",
        "expert_check",
        "instruction_prefix",
        "max_episode_steps",
        "policy_rng",
        "policy_seed",
        "release_expert_planners",
        "warm_then_gripper_planner",
        "pi05_defer_policy_render",
        "pi05_hoist_passive_force",
        "gripper_only_planner",
    ):
        if runtime_key in usr_args:
            args[runtime_key] = usr_args[runtime_key]

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    result_root = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}")
    save_dir = result_root / eval_run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = int(usr_args.get("start_seed", 100000 * (1 + seed)))
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    start_episode_index = int(usr_args.get("start_episode_index", 0))
    episode_schedule = load_persistent_episode_schedule(
        usr_args.get("persistent_episode_schedule")
    )
    if episode_schedule is not None:
        if args["eval_video_log"]:
            raise ValueError("persistent episode schedules do not support video logging")
        if bool(args.get("expert_check", True)):
            raise ValueError("persistent episode schedules require expert_check=False")
        test_num = len(episode_schedule)
        start_episode_index = episode_schedule[0]["episode_index"]
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   start_episode_index=start_episode_index,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   save_dir=save_dir,
                                   result_root=result_root,
                                   episode_schedule=episode_schedule)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                start_episode_index=0,
                video_size=None,
                instruction_type=None,
                save_dir=None,
                result_root=None,
                episode_schedule=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = bool(args.get("expert_check", True))
    if bool(args.get("gripper_only_planner", False)):
        if expert_check:
            raise RuntimeError("gripper-only planner cannot run expert_check=True")
        if args["policy_name"] != "pi05":
            raise RuntimeError("gripper-only planner is audited only for Pi0.5 qpos evaluation")
        from evostudio_benchmark_sdk.robotwin import install_gripper_only_planner

        install_gripper_only_planner()
        print("Installed EvoStudio GripperOnlyPlanner for policy evaluation")
    TASK_ENV.suc = 0
    TASK_ENV.test_num = int(start_episode_index)

    now_id = int(start_episode_index)
    succ_seed = 0
    completed_episodes = 0
    schedule_cursor = 0
    warm_then_gripper_installed = False
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while schedule_cursor < len(episode_schedule) if episode_schedule is not None else succ_seed < test_num:
        schedule_row = None
        if episode_schedule is not None:
            schedule_row = episode_schedule[schedule_cursor]
            now_seed = int(schedule_row["seed"])
            now_id = int(schedule_row["episode_index"])
            TASK_ENV.test_num = now_id
        episode_total_start_time = time.time()
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        TASK_ENV.defer_policy_render = bool(args.get("pi05_defer_policy_render", False))
        TASK_ENV.robot.hoist_passive_force = bool(args.get("pi05_hoist_passive_force", False))
        apply_episode_step_limit(TASK_ENV, args.get("max_episode_steps"))
        collision_monitor = InterarmContactMonitor(TASK_ENV)
        TASK_ENV.eval_collision_monitor = collision_monitor
        if not expert_check:
            episode_info = TASK_ENV.prepare_episode_metadata()
        if bool(args.get("warm_then_gripper_planner", False)) and not warm_then_gripper_installed:
            if expert_check or policy_name != "pi05":
                raise RuntimeError("warm-then-gripper planner is Pi0.5 eval-only")
            release_expert_motion_planners(TASK_ENV)
            from evostudio_benchmark_sdk.robotwin import install_gripper_only_planner

            install_gripper_only_planner()
            warm_then_gripper_installed = True
            print("Warmed CuRobo once, then installed EvoStudio GripperOnlyPlanner")
        elif bool(args.get("release_expert_planners", False)):
            if expert_check:
                raise RuntimeError("expert planners cannot be released when expert_check=True")
            if policy_name != "pi05":
                raise RuntimeError("expert planner release is audited only for the Pi0.5 qpos policy")
            release_expert_motion_planners(TASK_ENV)
        episode_info_list = [episode_info["info"]]
        instruction_seed = int(now_seed)
        frozen_instruction = schedule_row.get("instruction") if schedule_row is not None else None
        if frozen_instruction is not None:
            instruction = frozen_instruction
        else:
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction_rng = np.random.default_rng(instruction_seed)
            instruction = compose_instruction(
                instruction_rng.choice(results[0][instruction_type]),
                args.get("instruction_prefix", ""),
            )
        print(f"Evaluation instruction: {instruction}")
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        episode_start_time = time.time()
        policy_seed = int(
            schedule_row["policy_seed"]
            if schedule_row is not None
            else args.get("policy_seed", now_seed)
        )
        policy_rng = str(args.get("policy_rng", "legacy_stream"))
        set_episode_seed = getattr(model, "set_episode_seed", None)
        if policy_rng == "episode_addressable" and set_episode_seed is None:
            raise RuntimeError("episode-addressable policy RNG requires model.set_episode_seed")
        if set_episode_seed is not None:
            set_episode_seed(policy_seed)
        reset_func(model)
        profile_observation_s = 0.0
        profile_eval_s = 0.0
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            phase_started = time.perf_counter()
            observation = TASK_ENV.get_obs()
            profile_observation_s += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            eval_func(TASK_ENV, model, observation)
            profile_eval_s += time.perf_counter() - phase_started
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        episode_steps = int(TASK_ENV.take_action_cnt)
        episode_elapsed_sec = time.time() - episode_start_time
        setup_elapsed_sec = episode_start_time - episode_total_start_time
        total_episode_elapsed_sec = time.time() - episode_total_start_time
        collision = collision_monitor.summary()
        action_digest = getattr(model, "episode_action_sha256", lambda: None)()
        inference_requests = int(getattr(model, "inference_index", 0))
        rollout_activity_metrics = getattr(model, "rollout_metrics", lambda: {})()

        episode_save_dir = save_dir
        if schedule_row is not None:
            episode_save_dir = Path(result_root) / schedule_row["eval_run_id"]
            episode_save_dir.mkdir(parents=True, exist_ok=True)
        if episode_save_dir is not None:
            metrics_path = Path(episode_save_dir) / "rollout_metrics.jsonl"
            with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps({
                    "task_name": task_name,
                    "policy_name": args["policy_name"],
                    "task_config": args["task_config"],
                    "ckpt_setting": args["ckpt_setting"],
                    "seed": int(now_seed),
                    "scene_seed": int(now_seed),
                    "policy_seed": policy_seed,
                    "policy_rng": policy_rng,
                    "policy_action_sha256": action_digest,
                    "policy_inference_requests": inference_requests,
                    "rollout_metrics": rollout_activity_metrics,
                    "episode_index": int(TASK_ENV.test_num),
                    "success": bool(succ),
                    "episode_steps": episode_steps,
                    "max_episode_steps": int(TASK_ENV.step_lim),
                    "elapsed_sec": episode_elapsed_sec,
                    "setup_elapsed_sec": setup_elapsed_sec,
                    "total_episode_elapsed_sec": total_episode_elapsed_sec,
                    "profile_observation_sec": profile_observation_s,
                    "profile_eval_sec": profile_eval_s,
                    "profile_inference_sec": float(getattr(model, "profile_inference_s", 0.0)),
                    "profile_inference_calls": int(getattr(model, "profile_inference_calls", 0)),
                    "profile_action_sec": float(getattr(model, "profile_action_s", 0.0)),
                    "profile_action_calls": int(getattr(model, "profile_action_calls", 0)),
                    "collision": collision["collision"],
                    "collision_summary": collision,
                    "instruction": instruction,
                    "instruction_seed": instruction_seed,
                }) + "\n")

        print(
            f"Episode metrics: seed={now_seed}, policy_seed={policy_seed}, success={succ}, "
            f"steps={episode_steps}, collision={collision['collision']}, "
            f"elapsed_sec={episode_elapsed_sec:.3f}"
        )

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        completed_episodes += 1
        TASK_ENV.close_env(clear_cache=(completed_episodes % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        if schedule_row is None:
            TASK_ENV.test_num += 1
        else:
            TASK_ENV.test_num = now_id

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{completed_episodes}\033[0m => \033[95m{round(TASK_ENV.suc/completed_episodes*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        if schedule_row is None:
            now_seed += 1
        else:
            schedule_cursor += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
