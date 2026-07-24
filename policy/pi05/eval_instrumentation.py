from __future__ import annotations


class InterarmContactMonitor:
    """Count native simulator contact steps between the two robot arms."""

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

    def _arm_joints(self, arm: str):
        return self.task.robot.left_arm_joints if arm == "left" else self.task.robot.right_arm_joints

    def _gripper_joints(self, arm: str):
        return self.task.robot.left_gripper if arm == "left" else self.task.robot.right_gripper

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
        return {
            "collision": self.contact_simulation_steps > 0,
            "interarm_contact_simulation_steps": self.contact_simulation_steps,
            "first_contact_simulation_step": self.first_contact_simulation_step,
            "contact_policy_step_indices": sorted(self.contact_policy_steps),
            "monitored_simulation_steps": self.simulation_steps,
        }


def apply_episode_step_limit(task, max_episode_steps: int | None) -> None:
    if max_episode_steps is None:
        return
    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise TypeError("max_episode_steps must be an integer")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    task.step_lim = max_episode_steps
