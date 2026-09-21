from types import SimpleNamespace
import numpy as np
import pytest
from envs.timed_expert import TimedExpert

@pytest.mark.parametrize('project', [True,False])
def test_native_motion_preserves_constraint_frame(project):
    captured={}
    def plan(pose,*,constraint_pose,project_to_goal_frame):
        captured.update(weights=constraint_pose,project=project_to_goal_frame)
        return {'position':np.zeros((1,6)),'velocity':np.zeros((1,6))}
    task=SimpleNamespace(scene=SimpleNamespace(get_timestep=lambda:.004),plan_success=True,right_move_to_pose=plan)
    action=SimpleNamespace(action='move',target_pose=[.2,0,1,1,0,0,0],args={'constraint_pose':[1,1,1,1,0,1],'project_to_goal_frame':project})
    controls=list(TimedExpert(task).motion('withdraw',('right',[action])))
    assert len(controls)==1 and 'right_arm' in controls[0]
    assert captured=={'weights':[1,1,1,1,0,1],'project':project}


def test_source_duration_preserves_path_and_scales_velocity():
    position=np.arange(18,dtype=float).reshape(3,6)
    def plan(*args,**kwargs):
        return {'position':position.copy(),'velocity':np.ones((3,6))*6}
    task=SimpleNamespace(scene=SimpleNamespace(get_timestep=lambda:.004),plan_success=True,right_move_to_pose=plan)
    action=SimpleNamespace(action='move',target_pose=[0]*7,args={'duration_s':.020})
    rows=list(TimedExpert(task).motion('return',('right',[action])))
    assert len(rows)==5
    actual=np.concatenate([row['right_arm']['position'] for row in rows])
    np.testing.assert_array_equal(actual[::2],position)
    np.testing.assert_array_equal(actual[1],(position[0]+position[1])/2)
    assert all(np.all(row['right_arm']['velocity']==3) for row in rows)


@pytest.mark.parametrize('count',[15,31])
def test_common_duration_has_identical_sample_count(count):
    def plan(*args,**kwargs):
        return {'position':np.tile(np.linspace(0,1,count)[:,None],(1,6)), 'velocity':np.ones((count,6))}
    task=SimpleNamespace(scene=SimpleNamespace(get_timestep=lambda:.004),plan_success=True,right_move_to_pose=plan)
    action=SimpleNamespace(action='move',target_pose=[0]*7,args={'duration_s':1.0})
    rows=list(TimedExpert(task).motion('withdraw',('right',[action])))
    assert len(rows)==250
    np.testing.assert_array_equal(rows[0]['right_arm']['position'],np.zeros((1,6)))
    np.testing.assert_array_equal(rows[-1]['right_arm']['position'],np.ones((1,6)))
