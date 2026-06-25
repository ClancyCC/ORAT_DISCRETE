import argparse
import os
import math
import random
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from types import MethodType
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.io import read_image, write_jpeg
# from src.common.param import args
# from utils.logger import logger


ACTION_TO_NUM = {
    'h': 0,
    'w': 1,
    's': 2,
    'a': 3,
    'd': 4,
    'U': 5,
    'D': 6,
    'L': 7,
    'R': 8,
}

NUM_TO_ACTION = {v: k for k, v in ACTION_TO_NUM.items()}

# Legacy SAC/CQL code in this repo expects 2D continuous actions with the
# historical ranges used by `evaluate_deva()`:
#   action[0] in [-30, 30]
#   action[1] in [-100, 100]
#
# TrackUAV data uses 9 discrete actions:
#   h, w, s, a, d, U, D, L, R
# where L/R are yaw actions and the rest are translational/hold actions.
# To keep `load_Buffer()` unchanged, we project those 9 actions into the
# legacy 2D action space as a compatibility embedding.
ACTION_TO_LEGACY_CONTINUOUS = {
    0: np.array([0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 100.0], dtype=np.float32),
    2: np.array([0.0, -100.0], dtype=np.float32),
    3: np.array([-30.0, 0.0], dtype=np.float32),
    4: np.array([30.0, 0.0], dtype=np.float32),
    5: np.array([0.0, 50.0], dtype=np.float32),
    6: np.array([0.0, -50.0], dtype=np.float32),
    7: np.array([-15.0, 0.0], dtype=np.float32),
    8: np.array([15.0, 0.0], dtype=np.float32),
}

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def calc_ori_diff(pos, ori):
    x, y, z, w = ori

    norm = math.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    rotation_matrix = [
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ]

    rotation_matrix_T = [
        [rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]],
        [rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1]],
        [rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]]
    ]

    pos_array = np.array(pos)

    rotated_pos = [
        rotation_matrix_T[0][0] * pos_array[0] + rotation_matrix_T[0][1] * pos_array[1] + rotation_matrix_T[0][2] * pos_array[2],
        rotation_matrix_T[1][0] * pos_array[0] + rotation_matrix_T[1][1] * pos_array[1] + rotation_matrix_T[1][2] * pos_array[2],
        rotation_matrix_T[2][0] * pos_array[0] + rotation_matrix_T[2][1] * pos_array[1] + rotation_matrix_T[2][2] * pos_array[2]
    ]

    return [round(rotated_pos[0], 1), round(rotated_pos[1], 1), round(rotated_pos[2], 1)]


def resolve_frame_path(camera_dir, frame_id):
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = camera_dir / f"{frame_id:06d}{suffix}"
        if candidate.exists():
            return candidate
    return None


def compute_relative_metrics(relative_pos):
    rel = np.asarray(relative_pos, dtype=np.float32)
    forward = float(rel[0])
    lateral = float(rel[1])
    vertical = float(rel[2])
    planar_distance = float(np.hypot(forward, lateral))
    cross_track_distance = float(np.hypot(lateral, vertical))
    spatial_distance = float(np.linalg.norm(rel))
    horizontal_angle_deg = float(np.degrees(np.arctan2(lateral, forward)))
    vertical_angle_deg = float(np.degrees(np.arctan2(vertical, forward)))
    # `relative_pos` is already expressed in the tracker/body frame, so the
    # 3D angular error is the off-axis angle to the tracker forward axis.
    off_axis_angle_deg = float(np.degrees(np.arctan2(cross_track_distance, forward)))
    return {
        "forward": forward,
        "lateral": lateral,
        "vertical": vertical,
        "planar_distance": planar_distance,
        "cross_track_distance": cross_track_distance,
        "spatial_distance": spatial_distance,
        "direction_deg": horizontal_angle_deg,
        "horizontal_angle_deg": horizontal_angle_deg,
        "vertical_angle_deg": vertical_angle_deg,
        "off_axis_angle_deg": off_axis_angle_deg,
    }


def compute_tracking_reward(
    relative_pos,
    expected_distance=250.0,
    collision_distance=100.0,
    angle_half=180.0,
):
    metrics = compute_relative_metrics(relative_pos)
    angle_error = abs(metrics["off_axis_angle_deg"]) / angle_half
    distance_error = abs(expected_distance - metrics["spatial_distance"]) / 20
    reward = max(1.0 - angle_error - distance_error, -1.0)

    collision_proxy = int(
        metrics["spatial_distance"] < collision_distance
        and abs(metrics["off_axis_angle_deg"]) <= angle_half
    )
    reward -= float(collision_proxy)
    metrics["angle_error"] = angle_error
    metrics["distance_error"] = distance_error
    metrics["collision_proxy"] = collision_proxy
    return float(reward), metrics


def _append_tracking_anything_path():
    deva_root = Path(__file__).resolve().parents[1] / "Tracking-Anything-with-DEVA"
    deva_root_str = str(deva_root)
    if deva_root_str not in sys.path:
        sys.path.append(deva_root_str)


def init_text_processor_models(args):
    _append_tracking_anything_path()

    from deva.model.network import DEVA
    from deva import DEVAInferenceCore
    from deva.ext.grounding_dino import get_grounding_dino_model
    from deva.inference.result_utils import ResultSaver
    from deva.ext.with_text_processor import process_frame_with_text

    config = vars(args).copy()
    config["enable_long_term"] = not config.get("disable_long_term", False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = DEVA(config)
    if device == "cuda":
        network = network.cuda().eval()
    else:
        network = network.eval()

    model_weights = torch.load(args.DEVA_model, map_location=device)
    network.load_weights(model_weights)

    gd_model, sam_model = get_grounding_dino_model(config, device)

    return {
        "device": device,
        "network": network,
        "config": config,
        "gd_model": gd_model,
        "sam_model": sam_model,
        "deva_core_cls": DEVAInferenceCore,
        "result_saver_cls": ResultSaver,
        "process_frame": process_frame_with_text,
    }


def _save_mask_for_export(
    result_saver,
    prob,
    frame_name,
    need_resize=False,
    shape=None,
    save_the_mask=True,
    image_np=None,
    prompts=None,
    path_to_image=None,
):
    if need_resize:
        prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

    mask = torch.argmax(prob, dim=0).cpu()
    rgb_mask = np.zeros((*mask.shape[-2:], 3), dtype=np.uint8)

    if not result_saver.object_manager.use_long_id:
        return rgb_mask

    tmp_id_to_obj = dict(result_saver.object_manager.tmp_id_to_obj)
    segments_info = list(result_saver.object_manager.get_current_segments_info())
    mask_np = mask.numpy()
    unique_ids = np.unique(mask_np)

    if result_saver.is_first_frame:
        filtered_segments = []
        for segment in segments_info:
            if segment.get("category_id") != 0:
                continue
            segment_id = segment.get("id")
            matched_tmp_id = None
            for tmp_id, obj in tmp_id_to_obj.items():
                if obj.id == segment_id:
                    matched_tmp_id = tmp_id
                    break
            if matched_tmp_id is None:
                continue
            if np.any(mask_np == matched_tmp_id):
                filtered_segments.append(segment)

        if filtered_segments:
            highest_score_segment = max(filtered_segments, key=lambda x: x["score"])
            result_saver.target_id = highest_score_segment["id"]
            result_saver.is_first_frame = False

    tmp_to_obj_ids = {int(tmp_id): int(obj.id) for tmp_id, obj in tmp_id_to_obj.items()}
    segment_summaries = [
        {
            "id": int(s.get("id")),
            "cat": s.get("category_id"),
            "score": round(float(s.get("score")), 4) if s.get("score") is not None else None,
        }
        for s in segments_info
    ]
    # print(
    #     f"[EXPORT DEBUG][frame={frame_name}] unique={unique_ids[:20].tolist()} "
    #     f"tmp_to_obj={tmp_to_obj_ids} target_id={result_saver.target_id} "
    #     f"segments={segment_summaries}"
    # )

    for tmp_id, obj in tmp_id_to_obj.items():
        obj_mask = mask_np == tmp_id
        if not np.any(obj_mask):
            continue
        if obj.id == result_saver.target_id:
            rgb_mask[obj_mask] = np.array([255, 255, 255], dtype=np.uint8)
        else:
            rgb_mask[obj_mask] = result_saver.id2rgb_converter._id_to_rgb(obj.id)

    # print(f"[EXPORT DEBUG][frame={frame_name}] rgb_sum={int(rgb_mask.sum())}")
    return rgb_mask


class JPGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root_dir,
        batch_size=1,
        tokenizer=None,
        use_multiview=False,
        max_per_env=100,
    ):
        super().__init__()

        self.data_root_dir = data_root_dir
        self.batch_size = batch_size
        self._preload = []
        self.tokenizer = tokenizer
        self.use_multiview = use_multiview



        self.episode_dirs = []
        data_dir = Path(data_root_dir)
        csv_file = data_dir / 'run_summary.csv'
        data_env_nums = {}

        if csv_file.exists():
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                if 'final_error_NE_m' in row and row['final_error_NE_m'] >= 10:
                    continue
                relative_path = row['path'] if 'path' in row else None
                if relative_path:
                    episode_dir = data_dir / os.path.join(*relative_path.split('/')[-2:])
                else:
                    continue
                json_file = episode_dir / 'merged_data.json'
                if json_file.exists():
                    if row['environment'] not in data_env_nums:
                        data_env_nums[row['environment']] = 0

                    if data_env_nums[row['environment']] < max_per_env:
                        self.episode_dirs.append(episode_dir)
                        data_env_nums[row['environment']] += 1
        else:
            for episode_dir in sorted(data_dir.glob("*")):
                if episode_dir.is_dir() and episode_dir.name != 'log':
                    json_file = episode_dir / 'merged_data.json'
                    if json_file.exists():
                        self.episode_dirs.append(episode_dir)

        random.shuffle(self.episode_dirs)

        if len(self.episode_dirs) > 0:
            sample_dir = self.episode_dirs[0]
            sample_rgb = None
            for suffix in SUPPORTED_IMAGE_SUFFIXES:
                sample_candidates = sorted((sample_dir / 'frontcamera').glob(f"*{suffix}"))
                if sample_candidates:
                    sample_rgb = sample_candidates[0]
                    break
            if sample_rgb is None:
                raise FileNotFoundError(f"No supported image found under {sample_dir / 'frontcamera'}")
            sample_img = read_image(str(sample_rgb))
            if sample_img.shape[0] == 4:
                sample_img = sample_img[:3, :, :]
            orig_height, orig_width = sample_img.shape[1], sample_img.shape[2]
        else:
            orig_height, orig_width = 512, 512

        self.img_height = 512
        self.img_width = 512
        self.depth_height = 512
        self.depth_width = 512

        self.rgb_transform = T.Compose([
            T.Resize((self.img_height, self.img_width), antialias=True),
        ])

        if self.use_multiview:
            self.camera_views = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']
        #     print("END init JPG Dataset (MultiView) \t episodes: {} \t original size: {}x{} \t rgb resized to: {}x{} \t depth resized to: {}x{}".format(
        #         self.length, orig_width, orig_height, self.img_width, self.img_height, self.depth_width, self.depth_height))
        # else:
        #     print("END init JPG Dataset (SingleView) \t episodes: {} \t original size: {}x{} \t rgb resized to: {}x{} \t depth resized to: {}x{}".format(
        #         self.length, orig_width, orig_height, self.img_width, self.img_height, self.depth_width, self.depth_height))

    def _load_episode(self, episode_dir):
        json_file = episode_dir / 'merged_data.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
        trajectory = data['trajectory_raw_detailed']

        ref_json_file = os.path.join("/data/qingcheng.zhu/TravelUAV/datasets", str(episode_dir).split('/')[-2], str(episode_dir).split('/')[-1],  'merged_data.json')
        with open(ref_json_file, 'r') as f2:
            ref_trajectory = json.load(f2)['trajectory_raw_detailed']

        if isinstance(ref_trajectory, dict):
            ref_trajectory = list(ref_trajectory.values())

        relative_pos = []
        for i in range(len(trajectory)):
            pos = [ref_trajectory[i]['position'][j] - trajectory[i]['position'][j] for j in range(3)]
            ori = trajectory[i]['orientation']

            relative_pos.append(calc_ori_diff(pos, ori))

        action_numbers = data.get('action_numbers')
        actions_str = data.get('actions', [])
        frame_indices = data.get('index')

        if isinstance(frame_indices, list) and len(frame_indices) == len(trajectory):
            frame_ids = [int(x) for x in frame_indices]
        else:
            frame_ids = list(range(len(trajectory)))

        if isinstance(action_numbers, list) and len(action_numbers) == len(frame_ids):
            action_sequence = [int(x) for x in action_numbers]
        else:
            action_sequence = []
            for action_str in actions_str:
                action_sequence.append(ACTION_TO_NUM.get(action_str, 0))

        if len(action_sequence) < len(frame_ids):
            action_sequence.extend([0] * (len(frame_ids) - len(action_sequence)))
        elif len(action_sequence) > len(frame_ids):
            action_sequence = action_sequence[:len(frame_ids)]

        num_frames = len(frame_ids)

        obs_list = []
        for seq_idx in range(num_frames):
            frame_id = frame_ids[seq_idx]
            if self.use_multiview:
                rgb_imgs = []

                for view_name in self.camera_views:
                    camera_dir = episode_dir / view_name

                    rgb_path = resolve_frame_path(camera_dir, frame_id)

                    if rgb_path is not None:
                        rgb_tensor = read_image(str(rgb_path))
                        if rgb_tensor.shape[0] == 4:
                            rgb_tensor = rgb_tensor[:3, :, :]
                        rgb_tensor = self.rgb_transform(rgb_tensor)
                        rgb_img = rgb_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                        rgb_imgs.append(rgb_img)

                if len(rgb_imgs) > 0:
                    rgb_img = np.stack(rgb_imgs, axis=0)
                else:
                    rgb_img = None
            else:
                frontcamera_dir = episode_dir / 'frontcamera'

                rgb_path = resolve_frame_path(frontcamera_dir, frame_id)

                rgb_img = None

                if rgb_path is not None:
                    rgb_tensor = read_image(str(rgb_path))
                    if rgb_tensor.shape[0] == 4:
                        rgb_tensor = rgb_tensor[:3, :, :]
                    rgb_tensor = self.rgb_transform(rgb_tensor)
                    rgb_img = rgb_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

            obs = {
                'rgb': rgb_img,
                'frame_id': frame_id,
                'source_path': str(rgb_path) if rgb_path is not None else None,
            }
            obs_list.append(obs)

        return obs_list, action_sequence, relative_pos

    def _prepare_sample(self, obs_list, oracle_actions, relative_pos, max_len=None):
        states = []
        actions = []
        rel_pos = []
        dones = []
        
        seq_len = len(obs_list) if max_len is None else max_len
        
        for i in range(seq_len):
            if i < len(obs_list):
                frame_obs = obs_list[i]
                rgb = frame_obs.get('rgb')
                
                if rgb is not None:
                    states.append(rgb)
                else:
                    states.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
                
                actions.append(oracle_actions[i])
                rel_pos.append(relative_pos[i])
                
                # done 标记：序列最后一个位置为 1，其他为 0
                done = 1.0 if (i == seq_len - 1) else 0.0
                dones.append(done)
            else:
                # padding
                states.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
                actions.append(0)
                rel_pos.append(np.zeros(3, dtype=np.float32))
                dones.append(0.0)

        states = torch.from_numpy(np.stack(states))
        actions = torch.from_numpy(np.array(actions, dtype=np.int64))
        rel_pos = torch.from_numpy(np.stack(rel_pos))
        dones = torch.from_numpy(np.array(dones, dtype=np.float32))

        return states, actions, rel_pos, dones

    def __getitem__(self, idx):
        episode_dir = self.episode_dirs[idx]
        obs_list, oracle_actions, relative_pos = self._load_episode(episode_dir)
        return self._prepare_sample(obs_list, oracle_actions, relative_pos)

    def __len__(self):
        return len(self.episode_dirs)

    def export_legacy_buffer_pt(
        self,
        output_dir,
        use_text_processor=True,
        processor_bundle=None,
        expected_distance=250.0,
        collision_distance=100.0,
        overwrite=False,
        max_episodes=0,
        export_img_size=240,
        to_bgr=True,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        episode_dirs = self.episode_dirs
        if max_episodes > 0:
            episode_dirs = episode_dirs[:max_episodes]

        export_count = 0
        skip_count = 0
        for episode_dir in episode_dirs:
            obs_list, action_sequence, relative_pos = self._load_episode(episode_dir)

            output_name = f"{episode_dir.parent.name}__{episode_dir.name}.pt"
            output_path = output_dir / output_name
            if output_path.exists() and not overwrite:
                skip_count += 1
                continue

            image = []
            action = []
            reward = []
            is_first_flags = []
            is_last_flags = []

            T = len(obs_list)

            processor_state = None
            if use_text_processor:
                if processor_bundle is None:
                    raise ValueError("processor_bundle is required when use_text_processor=True")
                processor_state = self._init_episode_processor_state(processor_bundle, episode_dir.name)

            for step_idx in range(T):
                frame_obs = obs_list[step_idx]
                rgb = frame_obs.get("rgb")
                if rgb is None:
                    rgb = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
                else:
                    rgb = np.asarray(rgb, dtype=np.uint8)

                if use_text_processor:
                    processed_rgb = self._process_frame_with_text_processor(
                        rgb=rgb,
                        step_idx=step_idx,
                        frame_obs=frame_obs,
                        processor_state=processor_state,
                    )
                else:
                    processed_rgb = rgb

                # --- 图像缩放 (torchvision) ---
                img_tensor = torch.from_numpy(processed_rgb).permute(2, 0, 1).contiguous()  # (H,W,C) → (C,H,W)
                img_tensor = TF.resize(img_tensor, (export_img_size, export_img_size), antialias=True)
                processed_rgb = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)  # (C,H,W) → (H,W,C)

                # --- RGB → BGR (匹配 DataCollection.py 采集格式) ---
                if to_bgr:
                    processed_rgb = processed_rgb[:, :, ::-1]

                action_id = int(action_sequence[step_idx]) if step_idx < len(action_sequence) else 0
                # action_vec = ACTION_TO_LEGACY_CONTINUOUS.get(
                #     action_id,
                #     ACTION_TO_LEGACY_CONTINUOUS[0],
                # ).astype(np.float32)

                reward_value = 0.0
                reward_metrics = None
                if step_idx + 1 < len(relative_pos):
                    reward_value, reward_metrics = compute_tracking_reward(
                        relative_pos[step_idx + 1],
                        expected_distance=expected_distance,
                        collision_distance=collision_distance,
                    )

                image.append(processed_rgb)
                # action.append(action_vec.copy().reshape(1, 2))
                action.append(action_id)
                reward.append(np.array([reward_value], dtype=np.float32))
                is_first_flags.append(step_idx == 0)
                is_last_flags.append(step_idx == T - 1)

            # test_output_dir = output_dir / "test"
            # test_output_dir.mkdir(parents=True, exist_ok=True)
            # sample_count = min(3, len(image))
            # if sample_count > 0:
            #     sampled_indices = random.sample(range(len(image)), sample_count)
            #     for sampled_idx in sampled_indices:
            #         sampled_image = np.asarray(image[sampled_idx], dtype=np.uint8)
            #         if to_bgr:
            #             sampled_image = sampled_image[:, :, ::-1]  # BGR → RGB for write_jpeg
            #         sampled_tensor = torch.from_numpy(sampled_image).permute(2, 0, 1).contiguous()
            #         sampled_path = test_output_dir / f"{episode_dir.parent.name}__{episode_dir.name}__step_{sampled_idx:03d}.jpg"
            #         write_jpeg(sampled_tensor, str(sampled_path), quality=95)

            torch.save(
                {
                    "image": image,
                    "action": action,
                    "reward": reward,
                    "is_first": np.array(is_first_flags, dtype=np.bool_),
                    "is_last": np.array(is_last_flags, dtype=np.bool_),
                },
                output_path,
            )

            export_count += 1

        return {
            "exported": export_count,
            "skipped": skip_count,
            "output_dir": str(output_dir),
        }

    def _init_episode_processor_state(self, processor_bundle, episode_name):
        config = processor_bundle["config"].copy()
        config["temporal_setting"] = "online"
        config["enable_long_term_count_usage"] = True

        deva = processor_bundle["deva_core_cls"](processor_bundle["network"], config=config)
        deva.next_voting_frame = config["num_voting_frames"] - 1
        deva.enabled_long_id()

        result_saver = processor_bundle["result_saver_cls"](
            output_root=str(Path("/tmp") / "orat_deva_export"),
            video_name=episode_name,
            dataset="demo",
            object_manager=deva.object_manager,
        )
        result_saver.save_mask = MethodType(_save_mask_for_export, result_saver)

        return {
            "config": config,
            "deva": deva,
            "result_saver": result_saver,
            "gd_model": processor_bundle["gd_model"],
            "sam_model": processor_bundle["sam_model"],
            "process_frame": processor_bundle["process_frame"],
        }

    @torch.inference_mode()
    def _process_frame_with_text_processor(self, rgb, step_idx, frame_obs, processor_state):
        """
        Run DEVA segmentation on a single frame and return the colored mask as RGB.

        Replicates the original paper's pipeline:
          - GroundingDINO + SAM detects the target every N frames
          - DEVA propagates the mask in between
          - Output is an RGB mask with target in white and distractors in distinct colors,
            matching what load_Buffer() expects for deva_cnn_lstm input_type.
        """
        deva = processor_state["deva"]
        gd_model = processor_state["gd_model"]
        sam_model = processor_state["sam_model"]
        result_saver = processor_state["result_saver"]
        process_frame = processor_state["process_frame"]

        image_np = rgb.astype(np.uint8)
        frame_name = frame_obs.get("source_path") or f"{step_idx:06d}.png"

        result = process_frame(
            deva,
            gd_model,
            sam_model,
            str(frame_name),
            result_saver,
            step_idx,
            image_np=image_np,
        )

        if result is None:
            h, w = image_np.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        if isinstance(result, Image.Image):
            return np.array(result, dtype=np.uint8)

        return np.asarray(result, dtype=np.uint8)


def collate_fn(batch, max_len=None):
    """
    将多个 episode 的样本 collate 成一个 batch
    
    Args:
        batch: list of (states, actions, relative_pos, dones)
        max_len: 如果指定，则 padding 到该长度；否则使用该 batch 中最长 episode 的长度
    
    Returns:
        batched_states: [B, T, 512, 512, 3]
        batched_actions: [B, T]
        batched_rel_pos: [B, T, 3]
        batched_dones: [B, T]
    """
    if max_len is None:
        # 使用 batch 中最长 episode 的长度
        max_len = max(len(states) for states, _, _, _ in batch)
    
    batched_states = []
    batched_actions = []
    batched_rel_pos = []
    batched_dones = []
    
    for states, actions, rel_pos, dones in batch:
        seq_len = len(states)
        
        if seq_len < max_len:
            # padding
            pad_len = max_len - seq_len
            
            pad_states = torch.zeros((pad_len, 512, 512, 3), dtype=torch.uint8)
            pad_actions = torch.zeros(pad_len, dtype=torch.int64)
            pad_rel_pos = torch.zeros((pad_len, 3), dtype=torch.float32)
            pad_dones = torch.zeros(pad_len, dtype=torch.float32)
            
            states = torch.cat([states, pad_states], dim=0)
            actions = torch.cat([actions, pad_actions], dim=0)
            rel_pos = torch.cat([rel_pos, pad_rel_pos], dim=0)
            dones = torch.cat([dones, pad_dones], dim=0)
        
        batched_states.append(states)
        batched_actions.append(actions)
        batched_rel_pos.append(rel_pos)
        batched_dones.append(dones)
    
    return (
        torch.stack(batched_states),
        torch.stack(batched_actions),
        torch.stack(batched_rel_pos),
        torch.stack(batched_dones),
    )


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="JPG dataset inspect/export utility")
    parser.add_argument("--mode", choices=["inspect", "export"], default="inspect")
    parser.add_argument(
        "--data-root-dir",
        type=str,
        default="/data/qingcheng.zhu/TrackUAV/results/collect/trainset_1080_0519",
    )
    parser.add_argument("--output-pt-dir", type=str, default=None)
    parser.add_argument("--expected-distance", type=float, default=250.0)
    parser.add_argument("--collision-distance", type=float, default=100.0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-text-processor", action="store_true",
                        help="Skip DEVA/GroundingDINO/SAM segmentation, use raw RGB instead")
    parser.add_argument("--batch-size", type=int, default=3)

    _append_tracking_anything_path()
    from deva.inference.eval_args import add_common_eval_args
    from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args

    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)

    args = parser.parse_args()

    # 把 DEVA 相关的默认相对路径解析到 Tracking-Anything-with-DEVA/ 下
    deva_root = Path(__file__).resolve().parents[1] / "Tracking-Anything-with-DEVA"
    for attr in ("DEVA_model", "GROUNDING_DINO_CONFIG_PATH",
                 "GROUNDING_DINO_CHECKPOINT_PATH", "SAM_CHECKPOINT_PATH"):
        val = getattr(args, attr, None)
        if val and not os.path.isabs(val):
            resolved = str(deva_root / val.lstrip("./"))
            setattr(args, attr, resolved)

    dataset = JPGDataset(args.data_root_dir)

    if args.mode == "export":
        if not args.output_pt_dir:
            raise ValueError("--output-pt-dir is required in export mode")

        processor_bundle = None
        if not args.no_text_processor:
            processor_bundle = init_text_processor_models(args)

        export_summary = dataset.export_legacy_buffer_pt(
            output_dir=args.output_pt_dir,
            use_text_processor=not args.no_text_processor,
            processor_bundle=processor_bundle,
            expected_distance=args.expected_distance,
            collision_distance=args.collision_distance,
            overwrite=args.overwrite,
            max_episodes=args.max_episodes,
        )
        print(json.dumps(export_summary, indent=2, ensure_ascii=False))
    else:
        print(f"数据集 episode 数量: {len(dataset)}")
        print()

        print("=" * 50)
        print("【提取 1 条数据】")
        print("=" * 50)
        states, actions, rel_pos, dones = dataset[0]
        print(f"states    shape: {states.shape}   # [T, 512, 512, 3]")
        print(f"actions   shape: {actions.shape}   # [T]")
        print(f"rel_pos   shape: {rel_pos.shape}   # [T, 3]")
        print(f"dones    shape: {dones.shape}     # [T]")
        print(f"         序列长度 T: {states.shape[0]}")
        print(f"         最后一个 dones: {dones[-1].item()}")
        print()

        print("=" * 50)
        print(f"【提取 {args.batch_size} 条数据的 batch】")
        print("=" * 50)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

        batch_iter = iter(dataloader)
        batch_states, batch_actions, batch_rel_pos, batch_dones = next(batch_iter)

        print(f"batch_states  shape: {batch_states.shape}")
        print(f"batch_actions  shape: {batch_actions.shape}")
        print(f"batch_rel_pos  shape: {batch_rel_pos.shape}")
        print(f"batch_dones   shape: {batch_dones.shape}")
        print(f"              其中 T_max: {batch_states.shape[1]} (该 batch 中最长 episode 的长度)")
        print()
