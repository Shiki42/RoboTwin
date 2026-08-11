from __future__ import annotations

import hashlib

import numpy as np


class InterarmContactMonitor:
    """Record executed policy actions and native inter-arm contacts."""

    def __init__(self, task) -> None:
        self.task = task
        self.left_names = self._arm_entity_names("left")
        self.right_names = self._arm_entity_names("right")
        shared = self.left_names & self.right_names
        self.left_names -= shared
        self.right_names -= shared
        if not self.left_names or not self.right_names:
            raise ValueError("could not resolve disjoint left/right robot link names")
        self.simulation_steps = 0
        self.contact_simulation_steps = 0
        self.first_contact_simulation_step = None
        self.contact_policy_steps: set[int] = set()
        self._action_hasher = hashlib.sha256()
        self.action_count = 0

    def _arm_joints(self, arm: str):
        if arm == "left":
            return self.task.robot.left_arm_joints
        return self.task.robot.right_arm_joints

    def _gripper_joints(self, arm: str):
        if arm == "left":
            return self.task.robot.left_gripper
        return self.task.robot.right_gripper

    def _arm_entity_names(self, arm: str) -> set[str]:
        names = set()
        for joint in self._arm_joints(arm):
            names.update((joint.parent_link.entity.name, joint.child_link.entity.name))
        for joint, _, _ in self._gripper_joints(arm):
            names.update((joint.parent_link.entity.name, joint.child_link.entity.name))
        return names

    def _has_interarm_contact(self) -> bool:
        for contact in self.task.scene.get_contacts():
            first = contact.bodies[0].entity.name
            second = contact.bodies[1].entity.name
            if (first in self.left_names and second in self.right_names) or (
                first in self.right_names and second in self.left_names
            ):
                return True
        return False

    def record_action(self, action) -> None:
        array = np.ascontiguousarray(action)
        if array.dtype.hasobject:
            raise TypeError("policy actions must not use an object dtype")
        components = (
            array.dtype.str.encode("ascii"),
            np.asarray(array.shape, dtype=">i8").tobytes(),
            array.tobytes(),
        )
        for component in components:
            self._action_hasher.update(len(component).to_bytes(8, "big"))
            self._action_hasher.update(component)
        self.action_count += 1

    def observe(self) -> None:
        simulation_step = self.simulation_steps
        self.simulation_steps += 1
        if not self._has_interarm_contact():
            return
        self.contact_simulation_steps += 1
        if self.first_contact_simulation_step is None:
            self.first_contact_simulation_step = simulation_step
        self.contact_policy_steps.add(int(self.task.take_action_cnt))

    def summary(self) -> dict:
        if self.simulation_steps < 1:
            raise RuntimeError("native collision monitor observed no simulation steps")
        return {
            "collision": self.contact_simulation_steps > 0,
            "interarm_contact_simulation_steps": self.contact_simulation_steps,
            "first_contact_simulation_step": self.first_contact_simulation_step,
            "contact_policy_step_indices": sorted(self.contact_policy_steps),
            "monitored_simulation_steps": self.simulation_steps,
        }

    def episode_action_sha256(self) -> str:
        if self.action_count < 1:
            raise RuntimeError("native episode monitor recorded no policy actions")
        return self._action_hasher.hexdigest()


def apply_episode_step_limit(task, max_episode_steps: int | None) -> None:
    if max_episode_steps is None:
        return
    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise TypeError("max_episode_steps must be an integer")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    task.step_lim = max_episode_steps
