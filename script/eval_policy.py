import sys
import os
import subprocess
import hashlib
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
from script.eval_instrumentation import InterarmContactMonitor, apply_episode_step_limit

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def compose_instruction(instruction, prefix=""):
    instruction = str(instruction).strip()
    prefix = str(prefix or "").strip()
    if not prefix:
        return instruction
    return f"{prefix} {instruction}"


def load_seed_table(usr_args, task_name, task_config):
    path = Path(usr_args["seed_table_path"])
    serialized = path.read_bytes()
    actual_sha256 = hashlib.sha256(serialized).hexdigest()
    expected_sha256 = str(usr_args["seed_table_sha256"]).lower()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"seed table SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    table = json.loads(serialized)
    if table.get("schema_version") != "parallelvla.robotwin_seed_table.v1":
        raise ValueError("unsupported parallelVLA seed table schema")
    if table.get("task_name") != task_name or table.get("task_config") != task_config:
        raise ValueError("seed table task metadata does not match evaluation config")
    seeds = table.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seed table must contain a non-empty seed list")
    index = int(usr_args.get("seed_table_index", 0))
    if not 0 <= index < len(seeds):
        raise ValueError(f"seed table index out of range: {index}")
    seed = seeds[index]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"invalid seed table entry at index {index}: {seed!r}")
    return seed


def reset_and_seed_policy(model, reset_func, *, policy_rng, policy_seed):
    policy_rng = str(policy_rng)
    if policy_rng not in {"episode_addressable", "legacy_stream"}:
        raise ValueError("policy_rng must be 'episode_addressable' or 'legacy_stream'")

    reset_func(model)
    if policy_rng == "legacy_stream":
        return
    set_episode_seed = getattr(model, "set_episode_seed", None)
    if set_episode_seed is None:
        raise RuntimeError(
            "episode-addressable policy RNG requires model.set_episode_seed"
        )
    set_episode_seed(int(policy_seed))


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

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
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    for runtime_key in (
        "eval_video_log",
        "render_freq",
        "clear_cache_freq",
        "expert_check",
        "instruction_prefix",
        "max_episode_steps",
        "policy_rng",
        "policy_seed",
        "metrics_output",
        "requested_seed",
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

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
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

    if "seed_table_path" in usr_args:
        st_seed = load_seed_table(usr_args, task_name, task_config)
        if "requested_seed" in usr_args:
            requested_seed = int(usr_args["requested_seed"])
            if requested_seed != st_seed:
                raise ValueError(
                    "requested seed does not match seed table: "
                    f"{requested_seed} != {st_seed}"
                )
    else:
        seed = int(usr_args["seed"])
        st_seed = int(usr_args.get("start_seed", 100000 * (1 + seed)))
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    start_episode_index = int(usr_args.get("start_episode_index", 0))
    if test_num < 1:
        raise ValueError("test_num must be positive")
    if start_episode_index < 0:
        raise ValueError("start_episode_index must be non-negative")
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
                                   save_dir=save_dir)
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
                save_dir=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = bool(args.get("expert_check", True))
    TASK_ENV.suc = 0
    TASK_ENV.test_num = int(start_episode_index)

    now_id = int(start_episode_index)
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
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
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
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
        apply_episode_step_limit(TASK_ENV, args.get("max_episode_steps"))
        collision_monitor = InterarmContactMonitor(TASK_ENV)
        TASK_ENV.eval_collision_monitor = collision_monitor
        if not expert_check:
            episode_info = TASK_ENV.prepare_episode_metadata()
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = compose_instruction(
            np.random.choice(results[0][instruction_type]),
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
        policy_rng = str(args.get("policy_rng", "legacy_stream"))
        requested_seed = int(args.get("requested_seed", now_seed))
        policy_seed = int(args.get("policy_seed", requested_seed))
        reset_and_seed_policy(
            model,
            reset_func,
            policy_rng=policy_rng,
            policy_seed=policy_seed,
        )
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        episode_steps = int(TASK_ENV.take_action_cnt)
        episode_elapsed_sec = time.time() - episode_start_time
        collision = collision_monitor.summary()
        action_digest = collision_monitor.episode_action_sha256()
        action_count = collision_monitor.action_count
        if action_count != episode_steps:
            raise RuntimeError(
                "native executed-action count does not match episode steps: "
                f"{action_count} != {episode_steps}"
            )
        TASK_ENV.eval_collision_monitor = None
        inference_requests = int(getattr(model, "inference_index", 0))

        metrics_path = args.get("metrics_output")
        if metrics_path is None and save_dir is not None:
            metrics_path = Path(save_dir) / "rollout_metrics.jsonl"
        if metrics_path is not None:
            metrics_path = Path(metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                metrics_file.write(
                    json.dumps(
                        {
                            "task_name": task_name,
                            "policy_name": args["policy_name"],
                            "task_config": args["task_config"],
                            "ckpt_setting": args["ckpt_setting"],
                            "seed": int(now_seed),
                            "scene_seed": int(now_seed),
                            "requested_seed": requested_seed,
                            "policy_seed": policy_seed,
                            "policy_rng": policy_rng,
                            "policy_action_sha256": action_digest,
                            "policy_action_count": action_count,
                            "policy_inference_requests": inference_requests,
                            "episode_index": int(TASK_ENV.test_num),
                            "success": bool(succ),
                            "episode_steps": episode_steps,
                            "max_episode_steps": int(TASK_ENV.step_lim),
                            "elapsed_sec": episode_elapsed_sec,
                            "collision": collision["collision"],
                            "collision_summary": collision,
                            "instruction": instruction,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

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
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

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
                value = eval(value)
            except:
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
