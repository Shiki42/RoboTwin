from types import SimpleNamespace

import pytest

from eval_instrumentation import InterarmContactMonitor, apply_episode_step_limit


class Entity:
    def __init__(self, name):
        self.name = name


class Link:
    def __init__(self, name):
        self.entity = Entity(name)


class Joint:
    def __init__(self, parent, child):
        self.parent_link = Link(parent)
        self.child_link = Link(child)


def task_with_contacts(contact_pairs):
    left_joints = [Joint("base", "left_upper"), Joint("left_upper", "left_wrist")]
    right_joints = [Joint("base", "right_upper"), Joint("right_upper", "right_wrist")]
    contacts = [
        SimpleNamespace(bodies=(Link(first), Link(second)))
        for first, second in contact_pairs
    ]
    scene = SimpleNamespace(get_contacts=lambda: contacts)
    robot = SimpleNamespace(
        left_arm_joints=left_joints,
        right_arm_joints=right_joints,
        left_gripper=[],
        right_gripper=[],
    )
    return SimpleNamespace(robot=robot, scene=scene, take_action_cnt=7, step_lim=500)


def test_interarm_contact_monitor_uses_native_contacts():
    task = task_with_contacts([("left_wrist", "right_upper")])
    monitor = InterarmContactMonitor(task)

    monitor.observe()
    monitor.observe()

    assert monitor.summary() == {
        "collision": True,
        "interarm_contact_simulation_steps": 2,
        "first_contact_simulation_step": 0,
        "contact_policy_step_indices": [7],
        "monitored_simulation_steps": 2,
    }


def test_interarm_contact_monitor_ignores_same_arm_contacts():
    monitor = InterarmContactMonitor(task_with_contacts([("left_upper", "left_wrist")]))

    monitor.observe()

    assert monitor.summary()["collision"] is False


def test_apply_episode_step_limit_requires_positive_integer():
    task = SimpleNamespace(step_lim=500)
    apply_episode_step_limit(task, 700)
    assert task.step_lim == 700

    with pytest.raises(ValueError, match="positive"):
        apply_episode_step_limit(task, 0)
