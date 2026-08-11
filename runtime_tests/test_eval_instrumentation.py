from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from script.eval_instrumentation import (
    InterarmContactMonitor,
    apply_episode_step_limit,
)


class Scene:
    def __init__(self) -> None:
        self.contacts = []

    def get_contacts(self):
        return self.contacts


def body(name: str):
    return SimpleNamespace(entity=SimpleNamespace(name=name))


def joint(parent: str, child: str):
    return SimpleNamespace(parent_link=body(parent), child_link=body(child))


def task():
    robot = SimpleNamespace(
        left_arm_joints=[joint("shared_base", "left_link")],
        right_arm_joints=[joint("shared_base", "right_link")],
        left_gripper=[(joint("left_link", "left_finger"), None, None)],
        right_gripper=[(joint("right_link", "right_finger"), None, None)],
    )
    return SimpleNamespace(robot=robot, scene=Scene(), take_action_cnt=0)


def contact(first: str, second: str):
    return SimpleNamespace(bodies=(body(first), body(second)))


def test_monitor_reports_collision_free_native_receipt() -> None:
    episode = task()
    monitor = InterarmContactMonitor(episode)
    monitor.record_action(np.array([0.25, -0.5], dtype=np.float32))
    episode.take_action_cnt = 1
    monitor.observe()

    assert monitor.summary() == {
        "collision": False,
        "interarm_contact_simulation_steps": 0,
        "first_contact_simulation_step": None,
        "contact_policy_step_indices": [],
        "monitored_simulation_steps": 1,
    }
    assert monitor.action_count == 1
    assert len(monitor.episode_action_sha256()) == 64


def test_monitor_reports_native_interarm_contact_steps() -> None:
    episode = task()
    monitor = InterarmContactMonitor(episode)
    monitor.record_action(np.array([0.0], dtype=np.float32))
    episode.take_action_cnt = 1
    monitor.observe()
    episode.scene.contacts = [contact("left_link", "right_finger")]
    monitor.record_action(np.array([1.0], dtype=np.float32))
    episode.take_action_cnt = 2
    monitor.observe()

    assert monitor.summary() == {
        "collision": True,
        "interarm_contact_simulation_steps": 1,
        "first_contact_simulation_step": 1,
        "contact_policy_step_indices": [2],
        "monitored_simulation_steps": 2,
    }


def test_executed_action_digest_is_reproducible_and_content_sensitive() -> None:
    first = InterarmContactMonitor(task())
    second = InterarmContactMonitor(task())
    changed = InterarmContactMonitor(task())
    actions = (
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([[2.0, 3.0]], dtype=np.float64),
    )

    for monitor in (first, second, changed):
        monitor.record_action(actions[0])
    first.record_action(actions[1])
    second.record_action(actions[1].copy())
    changed.record_action(np.array([[2.0, 4.0]], dtype=np.float64))

    assert first.episode_action_sha256() == second.episode_action_sha256()
    assert first.episode_action_sha256() != changed.episode_action_sha256()
    assert first.action_count == 2


def test_monitor_requires_native_simulation_and_action_evidence() -> None:
    without_simulation = InterarmContactMonitor(task())
    without_simulation.record_action(np.array([0.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="no simulation steps"):
        without_simulation.summary()

    without_action = InterarmContactMonitor(task())
    without_action.observe()
    with pytest.raises(RuntimeError, match="no policy actions"):
        without_action.episode_action_sha256()


def test_monitor_rejects_object_actions() -> None:
    monitor = InterarmContactMonitor(task())
    with pytest.raises(TypeError, match="object dtype"):
        monitor.record_action(np.array([object()], dtype=object))


def test_episode_step_limit_validation() -> None:
    episode = SimpleNamespace(step_lim=700)
    apply_episode_step_limit(episode, None)
    assert episode.step_lim == 700
    apply_episode_step_limit(episode, 12)
    assert episode.step_lim == 12
    with pytest.raises(TypeError):
        apply_episode_step_limit(episode, True)
    with pytest.raises(ValueError):
        apply_episode_step_limit(episode, 0)


def test_base_task_records_action_before_native_execution() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "envs" / "_base_task.py"
    ).read_text()
    start = source.index("    def take_action(")
    end = source.index("    def save_camera_images(", start)
    take_action = source[start:end]

    record = take_action.index("collision_monitor.record_action(action)")
    action_increment = take_action.index("self.take_action_cnt += 1")
    scene_step = take_action.index("self.scene.step()")
    observe = take_action.index("collision_monitor.observe()")

    assert record < action_increment
    assert scene_step < observe
