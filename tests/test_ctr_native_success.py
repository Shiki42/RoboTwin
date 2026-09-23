import numpy as np
from envs.ctr_success import StableWindow,resting_in_target

def test_scan_return_region_requires_support_height_position_and_stationarity():
    def check(**kw):
        args=dict(position=[-.24,-.15,.75],target=[-.24,-.15,.75],linear_velocity=[0,0,0],angular_velocity=[0,0,0],table_contact=True,xy_radius=.035,z_tolerance=.035)
        args.update(kw);return resting_in_target(**args)
    assert check()
    assert not check(position=[-.20,-.15,.75])
    assert not check(position=[-.24,-.15,.80])
    assert not check(table_contact=False)
    assert not check(linear_velocity=[.021,0,0])
    assert not check(angular_velocity=[0,0,.101])

def test_stability_is_physics_time_not_repeated_queries_and_resets():
    gate=StableWindow()
    assert not gate.update(True,100,.004)
    for _ in range(100):assert not gate.update(True,100,.004)
    assert not gate.update(True,149,.004)
    assert gate.update(True,150,.004)
    assert not gate.update(False,151,.004)
    assert not gate.update(True,152,.004)
    assert not gate.update(True,201,.004)
    assert gate.update(True,202,.004)


def task_success_method(task):
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).parents[1]/'envs'/f'{task}.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==task)
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check_success')
    from types import SimpleNamespace
    ns={'np':np,'scan_object':SimpleNamespace(check_success=lambda t:t.geometry)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'task_success','exec'),ns)
    return ns['check_success']


def fake_task(names):
    from types import SimpleNamespace,MethodType
    from envs.ctr_success import NativeCtrEvaluation
    class Entity:
        def find_component_by_type(self,cls):return SimpleNamespace(linear_velocity=np.zeros(3),angular_velocity=np.zeros(3))
    t=SimpleNamespace(eval_mode=True,eval_physics_steps=0,take_action_cnt=1,info={},table=Entity(),released=False,geometry=False,scan_complete=False,return_complete=False,_native_success_window=StableWindow())
    t.robot=SimpleNamespace(get_measured_gripper_val=lambda:[1,1] if t.released else [0,0])
    actors=[]
    for name in names:
        actor=Entity();pose=SimpleNamespace(p=np.array([0.,0.,.95]));wrapped=SimpleNamespace(actor=actor,get_pose=lambda p=pose:p)
        setattr(t,name,wrapped);actors.append(actor)
    t.scene=SimpleNamespace(get_timestep=lambda:.004,get_contacts=lambda:[SimpleNamespace(bodies=[SimpleNamespace(entity=t.table),SimpleNamespace(entity=a)],points=[SimpleNamespace(separation=0)]) for a in actors])
    t.native_targets_success=MethodType(NativeCtrEvaluation.native_targets_success,t)
    return t


def test_scan_policy_latches_scan_and_requires_both_returns_without_expert_flags():
    from types import SimpleNamespace
    check=task_success_method('scan_object_ctr');t=fake_task(['object','scanner'])
    t.return_poses={'left':SimpleNamespace(p=np.array([-.24,-.15,.75])),'right':SimpleNamespace(p=np.array([.24,-.15,.75]))}
    matrix=np.eye(4);matrix[:3,:3]=[[0,0,-1],[1,0,0],[0,-1,0]]
    t.scanner.get_functional_point=lambda *a:matrix
    t.complete_scan=lambda:setattr(t,'scan_complete',True)
    assert not check(t) and not t.scan_complete
    t.geometry=True
    assert not check(t) and t.scan_complete
    t.geometry=False;t.released=True
    t.object.get_pose().p=t.return_poses['left'].p.copy()
    assert not check(t)  # only one returned
    t.scanner.get_pose().p=t.return_poses['right'].p.copy()
    assert not check(t)
    t.eval_physics_steps=50
    assert check(t) and not t.return_complete  # no expert-owned completion flag
    t.scan_complete=False
    assert not check(t)  # correct placement alone is insufficient


def test_blocks_policy_uses_four_positions_and_release_without_home_flag():
    check=task_success_method('blocks_ranking_rgb_ctr');t=fake_task(['red','yellow','blue','green'])
    t.table_z_bias=0;t.block_half_size=.02;t.targets={n:[x,-.17,.74] for n,x in zip(['red','yellow','blue','green'],[-.31,-.21,.21,.31])}
    for n,p in t.targets.items():getattr(t,n).get_pose().p=np.array([p[0],p[1],.76])
    assert not check(t) # grippers not released
    t.released=True
    assert not check(t)
    t.eval_physics_steps=50
    assert check(t) and not t.return_complete
    t.yellow.get_pose().p[0]=-.4
    assert not check(t)
