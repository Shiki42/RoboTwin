"""Plan one explicit source or physically replay one explicit CTR timing variant."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import getpass
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'script'))
import numpy as np
from collect_paired_datasets import config, setup, H5Capture, compare_scene, json_write, sha
from preview_paired_datasets import preview
from envs.ctr_timing import StagedProgram, StageClock, delay_masks, SIDES, REASONS, PHASES
from envs.paired_timing import apply_control, SETTLE, CandidateRejected
from envs.ctr_safety import CrossArmSafety


def signature(task):
    result = {name:dict(pose=np.r_[getattr(task,name).get_pose().p,getattr(task,name).get_pose().q].tolist(),
                       config=json.loads(json.dumps(getattr(task,name).config))) for name in task.record_actor_names}
    result['robot_qpos'] = np.r_[task.robot.left_entity.get_qpos(),task.robot.right_entity.get_qpos()].tolist()
    return result


def provenance():
    return dict(host=socket.gethostname(),account=getpass.getuser(),python=sys.executable,
                command=sys.argv,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip(),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['scan_object_ctr','blocks_ranking_rgb_ctr'])
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--variant',default='source')
    parser.add_argument('--u',type=float)
    parser.add_argument('--first',choices=SIDES)
    parser.add_argument('--delay-fraction',type=float)
    collect_episode(parser.parse_args())


def collect_episode(args, task=None):
    os.chdir(ROOT)
    args.output.mkdir(parents=True,exist_ok=False)
    if task is None:
        task=getattr(importlib.import_module('envs.'+args.task),args.task)()
    cfg=config(args.task)
    setup(task,cfg,args.seed)
    initial=signature(task)
    identity=provenance()
    if args.source is None:
        try:
            task.play_once()
        except RuntimeError as error:
            task.recorded_expert.program().save(args.output/'failed-program.npz')
            json_write(args.output/'failure.json',dict(error=str(error),initial=initial,
                final=signature(task),provenance=identity,
                scan_translation_audit=getattr(task,'scan_translation_audit',None),
                ready_diagnostic=getattr(task,'ready_diagnostic',None),
                events=task.recorded_expert.timeline.events))
            raise
        settling=0
        while not task.check_success() and settling<750:
            task.scene.step();task.recorded_expert.safety.check(settling);settling+=1
        success=bool(task.check_success())
        print(json.dumps(dict(stage='source',task=args.task,success=success)),flush=True)
        if not success:
            json_write(args.output/'failure.json',dict(initial=initial,final=signature(task),provenance=identity))
            raise CandidateRejected('source expert failed completion criterion')
        program=task.recorded_expert.program()
        program.save(args.output/'program.npz')
        receipt=dict(task=args.task,seed=args.seed,success=True,initial=initial,hashes=program.hashes(),
                     stages=program.stages,source_events=task.info['timing']['events'],provenance=identity,minimum_clearance_m=task.recorded_expert.safety.minimum_clearance if args.task=='blocks_ranking_rgb_ctr' else None)
        if args.task=='scan_object_ctr':
            receipt.update(ready_diagnostic=task.ready_diagnostic,scan_angle=task.scan_angle,scan_offset=task.scan_offset.tolist(),
                           scanner_base_functional_target=task.scanner_base_functional_target,indicator_projection=task.indicator_projection,scan_translation_audit=task.scan_translation_audit,return_motion_audit=task.return_motion_audit,return_paths={side:{key:value.tolist() for key,value in paths.items()} for side,paths in task.return_paths.items()})
        json_write(args.output/'source.json',receipt)
        task.close_env()
        del task.recorded_expert
        return receipt
    source=json.loads((args.source/'source.json').read_text())
    if source['task'] != args.task or source['seed'] != args.seed:
        raise ValueError('source identity mismatch')
    compare_scene(source['initial'],initial)
    program=StagedProgram.load(args.source/'program.npz')
    if program.hashes()!=source['hashes']:
        raise AssertionError('source program corrupted')
    clock=StageClock(program,u=args.u,first=args.first,delay_fraction=args.delay_fraction)
    safety=CrossArmSafety(task)
    writer=H5Capture(args.output/'episode.hdf5',task)
    reasons=[];indices=[];stage_indices=[];scan_steps=[];stage_geometry=[]
    try:
        while not clock.finished():
            step=clock.step
            if step%10==0: writer.capture(step)
            stage_index=clock.stage_index
            controls,why,index,end=clock.next()
            for side,row in controls.items():apply_control(task,side,row)
            task.scene.step();safety.check(step)
            if args.task=='scan_object_ctr' and clock.name=='put_back':
                for side,row in controls.items():
                    if PHASES[int(row[1])]=='release':
                        task.return_released[side]=True
                task.sample_return_audit()
            if args.task=='scan_object_ctr' and clock.name=='scan_align':
                task.validate_scan_translation()
            reasons.append(why);indices.append(index);stage_indices.append(stage_index)
            if end:
                if args.task=='scan_object_ctr' and clock.name=='prepare':
                    task.begin_scan_translation()
                stage_geometry.append(dict(stage=clock.name,step=clock.step,actors={name:np.r_[getattr(task,name).get_pose().p,getattr(task,name).get_pose().q].tolist() for name in task.record_actor_names}))
                if clock.name=='scan_align':
                    task.complete_scan();task.begin_return_audit();scan_steps.append(clock.step)
                clock.advance_stage()
        task.return_complete=True
        settling=0
        while not task.check_success() and settling<750:
            if clock.step%10==0:writer.capture(clock.step)
            task.scene.step();safety.check(clock.step)
            reasons.append([SETTLE,SETTLE]);indices.append([-1,-1]);stage_indices.append(-1)
            clock.step+=1;settling+=1
        success=bool(task.check_success())
        print(json.dumps(dict(task=args.task,variant=args.variant,success=success)),flush=True)
        writer.capture(clock.step)
        hashes=clock.close()
        receipt=dict(task=args.task,variant=args.variant,seed=args.seed,success=success,u=args.u,
                     first=clock.first,delay_fraction=clock.fraction,stages=clock.schedules,events=clock.events,
                     physics_dt_s=writer.dt,physics_steps=clock.step,scan_success_steps=scan_steps,
                     actual_offset_steps=clock.schedules[0]['delay_steps']*(1 if clock.first=='left' else -1),
                     source_program_sha256=sha(args.source/'program.npz'),control_hashes=hashes,stage_geometry=stage_geometry,
                     minimum_clearance_m=safety.minimum_clearance if args.task=='blocks_ranking_rgb_ctr' else None,provenance=identity,settling_steps=settling,
                     idle_definition='imposed_independent_stage_start_delay_only',result_lifecycle='reported_audit_pending')
        if args.task=='scan_object_ctr':
            receipt['scan_translation_audit']=task.scan_translation_audit
            receipt['return_motion_audit']=task.return_motion_audit
        masks=delay_masks(reasons,writer.steps)
        writer.f.create_dataset('observation/arm_active_mask',data=(~masks).astype(np.float32))
        for side,values in zip(SIDES,masks.T):writer.f.create_dataset('retime/'+side+'_idle',data=values)
        writer.f.create_dataset('physics/reasons',data=np.asarray(reasons,dtype=np.uint8),compression='lzf')
        writer.f.create_dataset('physics/source_index',data=np.asarray(indices,dtype=np.int32),compression='lzf')
        writer.f.create_dataset('physics/stage_index',data=np.asarray(stage_indices,dtype=np.int16),compression='lzf')
        writer.f.attrs.update(reason_names=json.dumps(REASONS),rgb_encoding='jpeg_rgb',wrist_camera_preset='centered_fovy90',
                              wrist_camera_receipt=json.dumps(task.wrist_camera_receipt),receipt=json.dumps(receipt))
        receipt.update(observations=len(writer.steps),training_frames=len(masks),idle_counts=masks.sum(axis=0).tolist())
        json_write(args.output/'result.json',receipt)
    finally:
        writer.close()
    preview(args.output,args.output/'preview.mp4')
    task.close_env()
    if not success:raise CandidateRejected('frozen replay failed completion criterion')
    return receipt

if __name__=='__main__':
    main()
