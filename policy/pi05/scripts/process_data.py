
import argparse
import json
import os

import cv2
import h5py
import numpy as np
import yaml


def native_dynamic_camera_indices(sample_count):
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.arange(sample_count, dtype=np.int64)


def load_hdf5(dataset_path, phase_metadata_path=None):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"dataset does not exist: {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = {}
        for cam_name in root["/observation/"]:
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
        phase_type_id = None
        arm_active_mask = None
        if "parallelvla" in root:
            if phase_metadata_path is not None:
                raise ValueError("phase metadata is present both internally and externally")
            phase_type_id = root["parallelvla/phase_type_id"][()]
            arm_active_mask = root["parallelvla/arm_active_mask"][()]

    if phase_metadata_path is not None:
        if not os.path.isfile(phase_metadata_path):
            raise FileNotFoundError(f"phase metadata does not exist: {phase_metadata_path}")
        with np.load(phase_metadata_path) as metadata:
            phase_type_id = metadata["phase_type_id"]
            arm_active_mask = metadata["arm_active_mask"]

    if phase_type_id is not None:
        expected_frames = left_gripper.shape[0]
        if phase_type_id.shape != (expected_frames,):
            raise ValueError(f"invalid phase metadata shape: {phase_type_id.shape}")
        if arm_active_mask.shape != (expected_frames, 2):
            raise ValueError(f"invalid arm mask shape: {arm_active_mask.shape}")

    return (
        left_gripper,
        left_arm,
        right_gripper,
        right_arm,
        image_dict,
        phase_type_id,
        arm_active_mask,
    )


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def data_transform(path, episode_num, save_path, phase_metadata_dir=None):
    begin = 0
    os.listdir(path)
    # assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):

        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path) as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            image_dict,
            phase_type_id,
            arm_active_mask,
        ) = load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5"),
            (
                os.path.join(phase_metadata_dir, f"episode{i}.npz")
                if phase_metadata_dir is not None
                else None
            ),
        )
        sample_count = left_gripper_all.shape[0] - 1
        main_camera_indices = native_dynamic_camera_indices(sample_count)

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []
        for j in range(left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.array([*left_arm.tolist(), left_gripper, *right_arm.tolist(), right_gripper])  # joints angle

            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][main_camera_indices[j]]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            if phase_type_id is not None:
                obs.create_dataset("phase_type_id", data=phase_type_id[1:])
                obs.create_dataset("arm_active_mask", data=arm_active_mask[1:])
                obs.attrs["main_camera_routing"] = "native_dynamic"
                obs.attrs["wrist_camera_routing"] = "native_dynamic_per_arm"
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument("--phase-metadata-dir")
    parser.add_argument("--target-dir")
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    begin = 0
    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = args.target_dir or f"processed_data/{task_name}-{setting}-{expert_data_num}"
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
        phase_metadata_dir=args.phase_metadata_dir,
    )
