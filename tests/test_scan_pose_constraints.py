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
