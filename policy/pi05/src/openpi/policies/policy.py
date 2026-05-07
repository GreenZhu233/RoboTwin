from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.robot_fk import RobotFKCalculator
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.models.cuda_timing import CudaTimer

BasePolicy: TypeAlias = _base_policy.BasePolicy

# ---- Debug flags for _compute_trajectory_distance ----
# Set env var TD_DEBUG=1 to enable, TD_DEBUG_PATH=/path/to/log to override log file
_TD_DEBUG = os.environ.get("TD_DEBUG", "0") == "1"
_TD_DEBUG_PATH = os.environ.get("TD_DEBUG_PATH", "/nvme/cch/RoboTwin/td_debug.log")


class WindowedTrajectoryDistance:
    """Helper class to track a window of trajectory distances for dynamic adjustment of denoising steps."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.distances = []
    
    def add(self, distance: float):
        """Add a new trajectory distance to the window."""
        self.distances.append(distance)
        if len(self.distances) > self.window_size:
            self.distances.pop(0)
    
    def get_last(self) -> float:
        """Get the most recent trajectory distance."""
        return self.distances[-1] if self.distances else 0.0
    
    def get_max(self) -> float:
        """Get the maximum trajectory distance in the window."""
        return max(self.distances) if self.distances else np.inf
    
    def get_min(self) -> float:
        """Get the minimum trajectory distance in the window."""
        return min(self.distances) if self.distances else 0.0

class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        denoise_steps_range: tuple[int, int] | None = None,
        pi0_step: int = 16,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            denoise_steps_range: The range of denoise steps to use for the policy.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # Initialize forward kinematics calculators if robot config is provided in metadata
        self._left_fk_calculator = None
        self._right_fk_calculator = None
        if metadata is not None and "robot_config" in metadata:
            robot_config = metadata["robot_config"]
            self._init_fk_calculators(robot_config)

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        if denoise_steps_range is not None:
            if sample_kwargs and "num_steps" in sample_kwargs:
                logging.warning(f"'denoise_steps_range' is set to {denoise_steps_range}, the keyword argument 'num_steps' will not be adopted.")
            self._denoise_steps_range = denoise_steps_range
            
            self._min_steps, self._max_steps = denoise_steps_range
            self._replan_steps = pi0_step
            self._windowed_trajectory_distance = WindowedTrajectoryDistance(window_size=5)

    def _init_fk_calculators(self, robot_config: dict):
        """Initialize forward kinematics calculators from robot config.
        
        Args:
            robot_config: Dictionary containing robot configuration with keys:
                - left_urdf_path: Path to left arm URDF
                - right_urdf_path: Path to right arm URDF
                - left_srdf_path: Path to left arm SRDF (optional)
                - right_srdf_path: Path to right arm SRDF (optional)
                - left_move_group: Left arm move group name
                - right_move_group: Right arm move group name
                - left_links: List of left arm link names
                - right_links: List of right arm link names
                - left_joints: List of left arm joint names
                - right_joints: List of right arm joint names
                - left_robot_origin_pose: Left robot origin pose [x, y, z, qw, qx, qy, qz]
                - right_robot_origin_pose: Right robot origin pose [x, y, z, qw, qx, qy, qz]
                - left_ee_link_name: Left end-effector link name
                - right_ee_link_name: Right end-effector link name
                - left_gripper_bias: Left gripper bias (default: 0.0)
                - right_gripper_bias: Right gripper bias (default: 0.0)
        """
        # Initialize left arm FK calculator
        self._left_fk_calculator = RobotFKCalculator(
            urdf_path=robot_config["left_urdf_path"],
            srdf_path=robot_config.get("left_srdf_path"),
            move_group=robot_config["left_move_group"],
            links=robot_config["left_links"],
            joints=robot_config["left_joints"],
            robot_origin_pose=np.array(robot_config["left_robot_origin_pose"]),
            ee_link_name=robot_config["left_ee_link_name"],
            gripper_bias=robot_config.get("left_gripper_bias", 0.0),
        )
        
        # Initialize right arm FK calculator
        self._right_fk_calculator = RobotFKCalculator(
            urdf_path=robot_config["right_urdf_path"],
            srdf_path=robot_config.get("right_srdf_path"),
            move_group=robot_config["right_move_group"],
            links=robot_config["right_links"],
            joints=robot_config["right_joints"],
            robot_origin_pose=np.array(robot_config["right_robot_origin_pose"]),
            ee_link_name=robot_config["right_ee_link_name"],
            gripper_bias=robot_config.get("right_gripper_bias", 0.0),
        )
        
        logging.info("Forward kinematics calculators initialized successfully.")

    def _compute_trajectory_distance(self, current_joint_state: np.ndarray, discarded_actions: np.ndarray) -> float:
        """Compute the maximum trajectory distance from discarded actions.
        
        Args:
            current_joint_state: Current joint state (14,) [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
            discarded_actions: Actions to compute distance for (N, 14)
            
        Returns:
            Maximum trajectory distance across all actions
        """
        if self._left_fk_calculator is None or self._right_fk_calculator is None:
            # Fallback: compute based on action magnitude
            return float(np.linalg.norm(np.diff(discarded_actions, axis=0), axis=1).sum())
        
        left_arm_dim = 6
        distance = 0
        cumulative_left_qpos = current_joint_state[:left_arm_dim].copy()
        cumulative_right_qpos = current_joint_state[left_arm_dim+1:left_arm_dim+left_arm_dim+1].copy()
        
        prev_left_ee = None
        prev_right_ee = None
        
        # -- debug log --
        _debug = _TD_DEBUG
        if _debug:
            _p = _TD_DEBUG_PATH
            import datetime as _dt
            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            with open(_p, "a") as _f:
                _f.write(f"\n{'='*60}\n")
                _f.write(f"[{_ts}] _compute_trajectory_distance called\n")
                _f.write(f"current_joint_state: {np.array2string(current_joint_state, precision=6, suppress_small=True)}\n")
                _f.write(f"discarded_actions shape: {discarded_actions.shape}\n")
                _f.write(f"discarded_actions:\n{np.array2string(discarded_actions, precision=6, suppress_small=True)}\n")
        
        for i in range(discarded_actions.shape[0]):
            action = discarded_actions[i]
            
            # Update cumulative joint positions (actions are relative)
            cumulative_left_qpos = cumulative_left_qpos + action[:left_arm_dim]
            cumulative_right_qpos = cumulative_right_qpos + action[left_arm_dim+1:left_arm_dim+left_arm_dim+1]
            
            if _debug:
                with open(_p, "a") as _f:
                    _f.write(f"\n--- step {i} ---\n")
                    _f.write(f"action[{i}]: {np.array2string(action, precision=6, suppress_small=True)}\n")
                    _f.write(f"cumulative_left_qpos:  {np.array2string(cumulative_left_qpos, precision=6, suppress_small=True)}\n")
                    _f.write(f"cumulative_right_qpos: {np.array2string(cumulative_right_qpos, precision=6, suppress_small=True)}\n")
            
            # Compute end-effector positions
            left_ee = self._left_fk_calculator.get_fk(cumulative_left_qpos)
            right_ee = self._right_fk_calculator.get_fk(cumulative_right_qpos)
            
            if _debug:
                with open(_p, "a") as _f:
                    _f.write(f"left_ee  position: {np.array2string(left_ee['position'], precision=6, suppress_small=True)}\n")
                    _f.write(f"left_ee  quaternion: {np.array2string(left_ee['quaternion'], precision=6, suppress_small=True)}\n")
                    _f.write(f"right_ee position: {np.array2string(right_ee['position'], precision=6, suppress_small=True)}\n")
                    _f.write(f"right_ee quaternion: {np.array2string(right_ee['quaternion'], precision=6, suppress_small=True)}\n")
            
            # Compute displacement
            if prev_left_ee is not None:
                left_displacement = np.linalg.norm(left_ee['position'] - prev_left_ee['position'])
                right_displacement = np.linalg.norm(right_ee['position'] - prev_right_ee['position'])
                distance += left_displacement + right_displacement
                
                if _debug:
                    with open(_p, "a") as _f:
                        _f.write(f"left_displacement:  {left_displacement:.8f}\n")
                        _f.write(f"right_displacement: {right_displacement:.8f}\n")
                        _f.write(f"cumulative distance: {distance:.8f}\n")
            
            prev_left_ee = left_ee
            prev_right_ee = right_ee
        
        if _debug:
            with open(_p, "a") as _f:
                _f.write(f"\n>>> final trajectory_distance: {distance:.8f}\n")
        
        return distance

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        timer = CudaTimer()
        timer.start()
        if hasattr(self, "_denoise_steps_range"):
            # Calculate the number of denoising steps based on the trajectory distance of the last action chunk. The smaller the distance, the more steps (up to max_steps).
            # Keep computations on GPU as long as possible, only convert to Python int at the end
            td = self._windowed_trajectory_distance.get_last()
            max_td = self._windowed_trajectory_distance.get_max()
            min_td = self._windowed_trajectory_distance.get_min()
            num_steps = np.ceil(self._max_steps + (self._min_steps - self._max_steps) * (td - min_td) / (max_td - min_td + 1e-5))
            sample_kwargs["num_steps"] = int(num_steps)
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        if hasattr(self, "_denoise_steps_range"):
            # calculate the trajectory distance of the end effector in the discarded actions
            discarded_actions = outputs["actions"][self._replan_steps:, :]
            current_joint_state = obs["state"]
            
            # Compute trajectory distance using forward kinematics
            trajectory_distance = self._compute_trajectory_distance(current_joint_state, discarded_actions)
            
            # Add to windowed trajectory distance tracker
            self._windowed_trajectory_distance.add(trajectory_distance)
        inference_time = timer.stop(actions)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_s": inference_time,
        }
        if hasattr(self, "_denoise_steps_range"):
            outputs["num_steps"] = sample_kwargs["num_steps"]
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
