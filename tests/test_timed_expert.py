import importlib.util
from pathlib import Path
import sys
import pytest

spec = importlib.util.spec_from_file_location('timed_expert', Path(__file__).parents[1]/'envs/timed_expert.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def motion(side, n):
    for _ in range(n):
        yield {side: 1}


@pytest.mark.parametrize('offset', [-10, -0.25, 0, 0.25, 10])
def test_independent_start_offset_and_duration(offset):
    samples = []
    clock = m.Timeline(0.01, lambda controls, step: samples.append((step, controls)))
    clock.run({'left': motion('left', 100), 'right': motion('right', 40)}, offset)
    starts = {arm: next(step for step, c in samples if arm in c) for arm in ['left','right']}
    assert starts['right']-starts['left'] == round(offset/0.01)
    assert sum('left' in c for _,c in samples) == 100
    assert sum('right' in c for _,c in samples) == 40


@pytest.mark.parametrize('offset,expected_wait', [(0, True), (0.5, False), (-0.5, False)])
def test_conditional_wait_and_mutual_exclusion(offset, expected_wait):
    clock = m.Timeline(0.01, lambda c,s: None)
    def lane(side):
        yield from motion(side, 10)
        yield m.Acquire('box')
        yield from motion(side, 20)
        yield m.Release('box')
    clock.run({'left':lane('left'), 'right':lane('right')}, offset)
    events = clock.events
    assert any(e['event']=='wait_outside_shared_workspace_start' for e in events) == expected_wait
    occupants = set()
    for e in events:
        if e['event']=='shared_workspace_enter':
            assert not occupants
            occupants.add(e['arm'])
        elif e['event']=='shared_workspace_exit':
            occupants.remove(e['arm'])
    assert not occupants


def test_reject_nonfinite_offset():
    with pytest.raises(ValueError):
        m.Timeline(0.01, lambda c,s:None).run({}, float('nan'))
