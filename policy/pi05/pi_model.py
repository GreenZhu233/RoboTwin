#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
import yaml

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, denoise_steps_range = None, 
                 left_robot_file=None, right_robot_file=None, window_size=5):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        # Build robot config for forward kinematics (supports multiple robot types)
        robot_config = self._build_robot_config(left_robot_file, right_robot_file)
        
        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id,
            denoise_steps_range=denoise_steps_range,
            pi0_step=pi0_step,
            robot_config=robot_config,
            window_size=window_size
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
    
    def _build_robot_config(self, left_robot_file=None, right_robot_file=None):
        """Build robot configuration for forward kinematics from embodiment config.
        
        Supports multiple robot types by loading separate configs for left and right arms.
        
        Args:
            left_robot_file: Path to left arm robot config file (e.g., "assets/embodiments/aloha-agilex/")
            right_robot_file: Path to right arm robot config file
            
        Returns:
            dict: Robot configuration containing URDF paths, joint names, etc.
        """
        robot_config = {}
        
        def load_single_robot_config(config_file):
            """Load config from a single robot config file path."""
            if config_file is None:
                return None
            config_path = os.path.join(config_file, "config.yml")
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                return None
        
        def build_arm_config(config, base_path, prefix):
            """Build arm configuration from loaded config.
            
            Args:
                config: The loaded robot config dict
                base_path: Base path for URDF/SRDF files
                prefix: Prefix for link names (e.g., "fl_" or "fr_")
            """
            if config is None:
                return {}
            # Get joint names from config
            joints = config.get("arm_joints_name", [["fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"]])[0]
            # Convert joint names to link names (fl_joint1 -> fl_link1)
            links = [j.replace("joint", "link") for j in joints]
            
            # Build paths - use absolute paths
            urdf_rel = config.get("urdf_path", "urdf/arx5_description_isaac.urdf")
            srdf_rel = config.get("srdf_path", "srdf/arx5_description_isaac.srdf")
            
            # Convert relative path to absolute
            if not os.path.isabs(urdf_rel):
                urdf_path = os.path.abspath(os.path.join(base_path, urdf_rel))
            else:
                urdf_path = urdf_rel
            
            if not os.path.isabs(srdf_rel):
                srdf_path = os.path.abspath(os.path.join(base_path, srdf_rel))
            else:
                srdf_path = srdf_rel
            
            return {
                "urdf_path": urdf_path,
                "srdf_path": srdf_path,
                "move_group": config.get("move_group", ["fl_link6", "fr_link6"])[0] if "fl_" in prefix else (config.get("move_group", ["fl_link6", "fr_link6"])[1] if len(config.get("move_group", [])) > 1 else config.get("move_group", ["fl_link6"])[0]),
                "ee_link_name": links[-1] if links else "fl_link6",  # Last link is the end-effector
                "joints": joints,
                "gripper_bias": config.get("gripper_bias", 0.12),
                "links": links,
            }
        
        # Load left arm config (prefix "fl_" for left arm)
        left_base = os.path.dirname(left_robot_file) + "/" if left_robot_file else "assets/embodiments/aloha-agilex/"
        left_config = load_single_robot_config(left_robot_file)
        if left_config is None and left_robot_file is None:
            # Use default
            default_path = "assets/embodiments/aloha-agilex/config.yml"
            try:
                with open(default_path, 'r') as f:
                    left_config = yaml.safe_load(f)
                left_base = "assets/embodiments/aloha-agilex/"
            except FileNotFoundError:
                print(f"Warning: Default robot config not found, using empty config")
                left_config = {}
        left_arm_config = build_arm_config(left_config if left_config else {}, left_base, "fl_")
        
        # Load right arm config (prefix "fr_" for right arm)
        right_base = os.path.dirname(right_robot_file) + "/" if right_robot_file else left_base
        if right_robot_file == left_robot_file:
            right_config = left_config
        else:
            right_config = load_single_robot_config(right_robot_file)
            if right_config is None and right_robot_file is None:
                right_config = left_config
        right_arm_config = build_arm_config(right_config if right_config else {}, right_base, "fr_")
        
        # Populate robot_config with left and right arm configs
        for key in ["urdf_path", "srdf_path", "move_group", "ee_link_name", "joints", "gripper_bias", "links"]:
            robot_config[f"left_{key}"] = left_arm_config.get(key, "")
            robot_config[f"right_{key}"] = right_arm_config.get(key, "")
        
        # Handle robot origin poses
        if left_config:
            robot_poses = left_config.get("robot_pose", [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]])
            robot_config["left_robot_origin_pose"] = robot_poses[0] if robot_poses else [0, -0.65, 0.0, 0.707, 0, 0, 0.707]
        else:
            robot_config["left_robot_origin_pose"] = [0, -0.65, 0.0, 0.707, 0, 0, 0.707]
        
        if right_config and right_config != left_config:
            robot_poses = right_config.get("robot_pose", [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]])
            robot_config["right_robot_origin_pose"] = robot_poses[0] if robot_poses else robot_config["left_robot_origin_pose"]
        else:
            robot_config["right_robot_origin_pose"] = robot_config["left_robot_origin_pose"]
        
        return robot_config

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
