"""Forward kinematics calculator using mplib's Pinocchio backend.

Provides ``RobotFKCalculator``: given move_group joint positions, computes
the end-effector pose in world frame via Pinocchio forward kinematics.
"""
import logging
import os

import numpy as np

# ---- Debug flags ----
# Set FK_DEBUG=1 to enable, FK_DEBUG_PATH=/path/to/log to override output file
_FK_DEBUG = os.environ.get("FK_DEBUG", "0") == "1"
_FK_DEBUG_PATH = os.environ.get("FK_DEBUG_PATH", "/tmp/fk_debug.log")


class RobotFKCalculator:
    """Forward kinematics calculator using mplib for joint positions to end-effector positions."""

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str | None,
        move_group: str,
        links: list,
        joints: list,
        robot_origin_pose: np.ndarray,  # [x, y, z, qw, qx, qy, qz]
        ee_link_name: str,
        gripper_bias: float = 0.0,
    ):
        """Initialize the forward kinematics calculator.

        Args:
            urdf_path: Path to the robot URDF file.
            srdf_path: Path to the SRDF file (optional).
            move_group: Move group name for planning.
            links: List of link names.
            joints: List of joint names.
            robot_origin_pose: Robot base pose in world frame [x, y, z, qw, qx, qy, qz].
            ee_link_name: Name of the end-effector link.
            gripper_bias: Bias offset for gripper pose.
        """
        import mplib

        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            move_group=move_group,
            user_link_names=[],
            user_joint_names=[],
        )
        self.planner.set_base_pose(self._pose_to_sapien(robot_origin_pose))
        self.gripper_bias = gripper_bias
        self.joint_names = joints

        # ---- Resolve end-effector link ----
        # Derive from the move_group's actual end effectors (robust against
        # config typos), falling back to the user-provided ee_link_name.
        ee_links = self.planner.robot.get_move_group_end_effectors()
        if ee_links:
            self.ee_link_name = ee_links[-1]  # last EE in the chain
        else:
            self.ee_link_name = ee_link_name  # fallback to config value

        # ---- Performance: cache frequently-used references ----
        self._pin_model = self.planner.pinocchio_model
        self._robot = self.planner.robot

        # Pre-compute end-effector link index (search once, not per FK call)
        self._ee_link_idx = self._find_link_index(self.ee_link_name)

        logging.info(
            "RobotFKCalculator initialized: ee_link=%s (idx=%d), move_group_joints=%s",
            self.ee_link_name,
            self._ee_link_idx,
            self.planner.move_group_joint_indices,
        )

    def _find_link_index(self, target_name: str) -> int:
        """Find the pinocchio link index matching the target name.

        Tries exact match first, then prefix-based fallback (e.g., "fl_link6"
        from "fl_link6_right"), then returns the last link index as last resort.
        """
        link_names = self.planner.user_link_names  # resolved after Planner.__init__
        for i, name in enumerate(link_names):
            if name == target_name:
                return i

        # Fallback: prefix match (e.g., "fl_link" from "fl_link6")
        target_prefix = target_name.rsplit("link", 1)[0]
        for i, name in enumerate(link_names):
            if target_prefix in name and "link" in name:
                return i

        # Last resort: last link (usually the end-effector)
        return len(link_names) - 1

    def _pose_to_sapien(self, pose: np.ndarray):
        """Convert pose array to sapien Pose."""
        import sapien.core as sapien

        return sapien.Pose(pose[:3], pose[3:])

    def get_active_joint_names(self) -> list:
        """Get the names of active (non-fixed) joints."""
        return self.joint_names

    def get_fk(self, joint_positions: np.ndarray) -> dict:
        """Compute forward kinematics for given joint positions.

        Uses robot.set_qpos to correctly assign move_group joints to the full
        joint vector, then runs Pinocchio FK and transforms to world frame.

        Args:
            joint_positions: Joint position array for move_group joints (dof,).

        Returns:
            dict with 'position' [x, y, z] and 'quaternion' [qw, qx, qy, qz]
        """
        import transforms3d as t3d

        # Use robot.set_qpos(full=False) to correctly assign move_group joints
        # to the right indices in the full joint vector.
        self._robot.set_qpos(joint_positions, full=False)
        full_qpos = self._robot.get_qpos()

        # -- debug: dump FK internals --
        _debug = _FK_DEBUG
        if _debug:
            _p = _FK_DEBUG_PATH
            with open(_p, "a") as _f:
                _f.write(
                    f"\n  [FK debug] input qpos ({len(joint_positions)}): "
                    f"{np.array2string(joint_positions, precision=6, suppress_small=True)}\n"
                )
                _f.write(
                    f"  [FK debug] full_qpos ({len(full_qpos)}): "
                    f"{np.array2string(full_qpos, precision=6, suppress_small=True)}\n"
                )
                _f.write(f"  [FK debug] full_qpos dtype={full_qpos.dtype}, shape={full_qpos.shape}\n")
                _f.write(f"  [FK debug] ee_link_idx={self._ee_link_idx}, ee_link_name={self.ee_link_name}\n")
                _f.write(f"  [FK debug] user_link_names={self.planner.user_link_names}\n")
                _f.write(f"  [FK debug] move_group_joint_indices={self.planner.move_group_joint_indices}\n")

        # Pinocchio FK: compute, then read link pose
        self._pin_model.compute_forward_kinematics(full_qpos)
        ee_pose_base = self._pin_model.get_link_pose(self._ee_link_idx)

        if _debug:
            with open(_p, "a") as _f:
                _f.write(
                    f"  [FK debug] ee_pose_base.p="
                    f"{np.array2string(np.array(ee_pose_base.p), precision=6, suppress_small=True)}\n"
                )
                _f.write(
                    f"  [FK debug] ee_pose_base.q="
                    f"{np.array2string(np.array(ee_pose_base.q), precision=6, suppress_small=True)}\n"
                )
                base_p = np.array(self._robot.get_base_pose().p)
                base_q = np.array(self._robot.get_base_pose().q)
                _f.write(
                    f"  [FK debug] base_pose.p="
                    f"{np.array2string(base_p, precision=6, suppress_small=True)}\n"
                )
                _f.write(
                    f"  [FK debug] base_pose.q="
                    f"{np.array2string(base_q, precision=6, suppress_small=True)}\n"
                )

        # Transform to world frame: T_world_ee = T_world_base * T_base_ee
        ee_pose_world = self._robot.get_base_pose() * ee_pose_base

        # Extract position and quaternion (Pose.q = [qw, qx, qy, qz])
        pos = np.array(ee_pose_world.p, dtype=np.float64)
        quat = np.array(ee_pose_world.q, dtype=np.float64)

        if _debug:
            with open(_p, "a") as _f:
                _f.write(
                    f"  [FK debug] world pos="
                    f"{np.array2string(pos, precision=6, suppress_small=True)}\n"
                )
                _f.write(
                    f"  [FK debug] world quat="
                    f"{np.array2string(quat, precision=6, suppress_small=True)}\n"
                )

        # Apply gripper bias offset along x-axis of gripper frame
        if self.gripper_bias != 0:
            rot_mat = t3d.quaternions.quat2mat(quat)
            pos = pos + rot_mat[:, 0] * self.gripper_bias  # x-axis * bias

        ee_pose_7d = np.concatenate([pos, quat])
        return {
            "position": pos,
            "quaternion": quat,
            "pose_7d": ee_pose_7d,
        }
