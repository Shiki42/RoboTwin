"""Qualify native task methods using a frozen expert program, without expert flags."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first',choices=['left','right'])
    parser.add_argument('--delay-fraction',type=float,default=0.)
    args=parser.parse_args();os.chdir(ROOT)
    from collect_paired_datasets import config,setup,compare_scene
    from collect_ctr_experts import signature
    from envs.ctr_timing import StagedProgram,StageClock
    from envs.paired_timing import apply_control
    from envs.ctr_safety import CrossArmSafety
    source=json.loads((args.source/'source.json').read_text());name=source['task']
    task=getattr(importlib.import_module('envs.'+name),name)()
    cfg=config(name);cfg.update(eval_mode=True,is_test=True,save_data=False)
    args.output.mkdir(parents=True,exist_ok=False)
    setup(task,cfg,source['seed']);compare_scene(source['initial'],signature(task))
    task.set_instruction('Complete the fixed-role CTR task and release all items in their target regions.')
    assert not task.check_success() and not task.return_complete
    initial_obs=task.get_obs();assert 'observation' in initial_obs
    action=np.array(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState())
    task.take_action(action,action_type='qpos')
    assert task.take_action_cnt==1 and task.eval_physics_steps>0 and not task.eval_success
    program=StagedProgram.load(args.source/'program.npz')
    clock=StageClock(program,first=args.first,delay_fraction=args.delay_fraction) if args.first else StageClock(program)
    safety=CrossArmSafety(task);scan_step=None;success_step=None
    while not clock.finished():
        controls,_,_,end=clock.next()
        for side,row in controls.items():apply_control(task,side,row)
        task.scene.step();task.eval_physics_steps+=1;safety.check(clock.step)
        success=task.check_success()
        if name=='scan_object_ctr' and task.scan_complete and scan_step is None:scan_step=clock.step
        if success and success_step is None:success_step=clock.step
        if end:clock.advance_stage()
    for _ in range(100):
        task.scene.step();task.eval_physics_steps+=1;safety.check(clock.step)
        if task.check_success() and success_step is None:success_step=clock.step
    assert task.check_success(),task.info.get('evaluation')
    assert not task.return_complete  # no expert play_once/stage callback was used
    if name=='scan_object_ctr':assert scan_step is not None and success_step>=scan_step
    before=task.eval_physics_steps
    action=np.array(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState())
    action[0]+=.001
    task.take_action(action,action_type='qpos')
    assert task.eval_success and task.eval_physics_steps>before
    result=dict(success=True,task=name,seed=source['seed'],source=str(args.source),source_hashes=program.hashes(),
        scan_step=scan_step,success_step=success_step,native_action_calls=task.take_action_cnt,
        native_physics_steps=task.eval_physics_steps,expert_completion_flag=task.return_complete,
        evaluation=task.info['evaluation'],first=args.first,delay_fraction=args.delay_fraction)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    task.close_env();print(json.dumps(result),flush=True)

if __name__=='__main__':main()
