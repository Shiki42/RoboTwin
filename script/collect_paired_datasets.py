"""Collect the user-specified 3 x (50+50+50+100) frozen-control episodes."""
import argparse
from copy import deepcopy
import gc
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import h5py
import numpy as np
import yaml
from envs.paired_timing import (RecordExpert, Program, Clock, CandidateRejected,
                               SIDES, REASONS, WORKSPACE, SETTLE, apply_control, action_masks)
from envs.timed_expert import TimedExpert
from envs.utils import UnStableError

TASKS = ('pick_dual_bottles','scan_object','place_dual_shoes')
CAMERAS = ('head_camera','left_camera','right_camera')


def json_write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def signature(task):
    names = {'pick_dual_bottles':['bottle1','bottle2'],
             'scan_object':['scanner','object'],
             'place_dual_shoes':['left_shoe','right_shoe','shoe_box']}[task.task_name]
    result={}
    for name in names:
        actor=getattr(task,name)
        result[name] = dict(pose=np.r_[actor.get_pose().p,actor.get_pose().q].tolist(),
                           config=actor.config)
    result['robot_qpos'] = np.r_[task.robot.left_entity.get_qpos(),task.robot.right_entity.get_qpos()].tolist()
    return deepcopy(result)


def compare_scene(a,b):
    if a.keys()!=b.keys(): raise AssertionError('scene fields differ')
    for key in a:
        if key=='robot_qpos':
            if not np.allclose(a[key],b[key],atol=1e-7,rtol=0):
                raise AssertionError('initial robot state changed')
        else:
            if a[key]['config'] != b[key]['config'] or not np.allclose(a[key]['pose'],b[key]['pose'],atol=1e-7,rtol=0):
                raise AssertionError(f'initial actor changed: {key}')


def config(task):
    cfg=yaml.safe_load((ROOT/'task_config/demo_clean.yml').read_text())
    robot=ROOT/'assets/embodiments/aloha-agilex'
    embodiment=yaml.safe_load((robot/'config.yml').read_text())
    cfg.update(task_name=task,task_config='paired_frozen_v1',
               left_robot_file=str(robot),right_robot_file=str(robot),
               left_embodiment_config=embodiment,right_embodiment_config=deepcopy(embodiment),
               dual_arm_embodied=True,need_plan=True,save_data=False,save_freq=10,
               render_freq=0,right_start_offset_s=0.0)
    cfg['camera']['wrist_camera_preset']='centered_fovy90'
    return cfg


def setup(task,cfg,seed):
    random.seed(seed)
    try:
        task.setup_demo(seed=seed,now_ep_num=0,**cfg)
    except UnStableError as e:
        raise CandidateRejected('unstable initial scene: '+str(e)) from e


def collision_step(driver,step):
    try:
        driver.tick({},step)
    except RuntimeError as e:
        if str(e).startswith('inter-arm/object collision:'):
            raise CandidateRejected(str(e)) from e
        raise


def validate_wait(driver,side):
    try:
        driver.validate_wait(side)
    except RuntimeError as e:
        if 'waiting shoe is not outside' in str(e):
            raise CandidateRejected(str(e)) from e
        raise


class H5Capture:
    def __init__(self,path,task):
        self.path=path
        self.task=task
        self.f=h5py.File(path,'w')
        self.datasets={}
        self.steps=[]
        self.dt=float(task.scene.get_timestep())

    def append(self,path,value,jpeg=False):
        array=np.asarray(value)
        if path not in self.datasets:
            parent,name=path.rsplit('/',1) if '/' in path else ('',path)
            group=self.f.require_group(parent) if parent else self.f
            if jpeg:
                self.datasets[path]=group.create_dataset(name,shape=(0,),maxshape=(None,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
            else:
                if array.dtype.kind not in 'biuf': raise TypeError(f'non-numeric observation: {path}')
                self.datasets[path]=group.create_dataset(name,shape=(0,)+array.shape,maxshape=(None,)+array.shape,
                                     dtype=array.dtype,chunks=True,compression='lzf')
        d=self.datasets[path]; n=len(d); d.resize(n+1,axis=0); d[n]=value

    def capture(self,step):
        if self.steps and self.steps[-1]==step: return
        obs=self.task.get_obs()
        camera_validation=self.task.validate_wrist_cameras()
        for side,v in camera_validation.items():
            self.append('camera_validation/'+side+'_mount',v['actual_mount'])
            self.append('camera_validation/'+side+'_fovy_deg',v['fovy_deg'])
        real=np.asarray(self.task.robot.get_left_arm_real_jointState()+self.task.robot.get_right_arm_real_jointState())
        real_grippers=self.task.robot.get_normal_real_gripper_val()
        real[6],real[13]=real_grippers
        self.append('observation/state',real)
        names={'pick_dual_bottles':['bottle1','bottle2'], 'scan_object':['scanner','object'],
               'place_dual_shoes':['left_shoe','right_shoe','shoe_box']}[self.task.task_name]
        for name in names:
            pose=getattr(self.task,name).get_pose()
            self.append('actor_pose/'+name,np.r_[pose.p,pose.q])
        def visit(value,path=''):
            for key,item in value.items():
                child=f'{path}/{key}' if path else key
                if path=='observation' and key not in CAMERAS: continue
                if isinstance(item,dict): visit(item,child)
                elif key=='rgb':
                    okay,encoded=cv2.imencode('.jpg',cv2.cvtColor(item,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not okay: raise RuntimeError('JPEG encoding failed')
                    self.append(child,encoded.reshape(-1),jpeg=True)
                elif item is not None and np.asarray(item).size:
                    self.append(child,item)
        visit(obs)
        self.append('simulation_time_s',step*self.dt)
        self.append('physics_step',np.int64(step))
        self.steps.append(step)

    def finish(self,reasons,indices,receipt):
        masks,overlap=action_masks(reasons,self.steps)
        for side,col in zip(SIDES,masks.T): self.f.create_dataset('retime/'+side+'_idle',data=col)
        self.f.create_dataset('retime/overlap',data=overlap)
        self.f.create_dataset('physics/reasons',data=np.asarray(reasons,dtype=np.uint8),compression='lzf')
        self.f.create_dataset('physics/source_index',data=np.asarray(indices,dtype=np.int32),compression='lzf')
        self.f.attrs['reason_names']=json.dumps(REASONS)
        self.f.attrs['rgb_encoding']='jpeg_rgb'
        self.f.attrs['wrist_camera_preset']='centered_fovy90'
        self.f.attrs['wrist_camera_receipt']=json.dumps(self.task.wrist_camera_receipt)
        self.f.attrs['receipt']=json.dumps(receipt)
        self.f.flush()
        return dict(observations=len(self.steps),training_frames=len(masks),
                    left_idle_frames=int(masks[:,0].sum()),right_idle_frames=int(masks[:,1].sum()),
                    overlap_frames=int(overlap.sum()))

    def close(self): self.f.close()


def replay(task,p,task_name,u,path,expected_scene):
    compare_scene(expected_scene,signature(task))
    clock=Clock(p,task_name,u)
    driver=TimedExpert(task)
    if p.gate:
        task.add_prohibit_area(task.shoe_box,padding=0)
        bounds=task.prohibited_area.pop()
        driver.box_half_width=max(abs(bounds[0]),abs(bounds[2]))
    reasons=[]; indices=[]
    writer=H5Capture(path,task)
    try:
        while not clock.finished():
            step=clock.step
            if step%10==0: writer.capture(step)
            controls,why,source_index=clock.next()
            for s,row in controls.items(): apply_control(task,s,row)
            for a,s in enumerate(SIDES):
                if why[a]==WORKSPACE: validate_wait(driver,s)
            collision_step(driver,step)
            reasons.append(why); indices.append(source_index)
        hashes=clock.close()
        # Only physically necessary settling; no unconditional idle tail.
        settling=0
        while not task.check_success() and settling < 750:
            if clock.step%10==0: writer.capture(clock.step)
            collision_step(driver,clock.step)
            reasons.append([SETTLE,SETTLE]); indices.append([-1,-1])
            settling+=1; clock.step+=1
        success=bool(task.check_success())
        if not success: raise CandidateRejected('native success criterion failed after frozen replay')
        writer.capture(clock.step)
        receipt=dict(success=success,task=task_name,u=u,
                     nominal_offset_steps=clock.raw_delta,actual_offset_steps=clock.delta,
                     starts=clock.starts,durations={s:len(getattr(p,s)) for s in SIDES},
                     physics_dt_s=writer.dt,physics_steps=clock.step,settling_steps=settling,
                     control_hashes=hashes,events=clock.events,wrist_camera=task.wrist_camera_receipt,
                     workspace_wait_steps={s:int(np.count_nonzero(np.asarray(reasons)[:,a]==WORKSPACE)) for a,s in enumerate(SIDES)},
                     cross_arm_collisions=driver.contacts)
        receipt.update(writer.finish(reasons,indices,receipt))
        return receipt
    finally:
        writer.close()


def excluded_seeds(ctr,task):
    path=ctr/'eval/robotwin/seeds'/f'{task}.json'
    if not path.exists(): return set()
    return set(json.loads(path.read_text())['seeds'])


def collect(args):
    os.chdir(ROOT)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    tasks=list(TASKS) if args.task=='all' else [args.task]
    identity=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  host=socket.gethostname(),account=getpass.getuser(),
                  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).strip(),
                  runtime=sys.executable,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  requested_slots=args.slots,max_candidates=args.max_candidates)
    json_write(output/'invocation.json',identity)
    for name in tasks:
        start=time.monotonic()
        directory=output/name; directory.mkdir(exist_ok=True)
        manifest_path=directory/'manifest.json'
        accepted=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        if len(accepted)==args.slots: continue
        attempts_path=directory/'attempts.jsonl'
        attempts=[json.loads(line) for line in attempts_path.read_text().splitlines()] if attempts_path.exists() else []
        # A seed with an explicit rejection is never silently retried on resume.
        seed_start=max([a['seed'] for a in attempts]+[-1])+1
        blocked=excluded_seeds(args.ctr_root,name)
        cls=getattr(importlib.import_module('envs.'+name+'_timed'),name+'_timed')
        task=cls(); task.expert_driver_type=RecordExpert
        cfg=config(name)
        cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        for seed in range(seed_start,args.max_candidates):
            if len(accepted)==args.slots: break
            if seed in blocked: continue
            slot=len(accepted)
            stage='source'; attempt_start=time.monotonic()
            staging=directory/f'.candidate-{seed}'
            if staging.exists(): shutil.rmtree(staging)
            staging.mkdir()
            print(json.dumps(dict(phase='candidate',task=name,seed=seed,slot=slot)),flush=True)
            try:
                setup(task,cfg,seed)
                initial=signature(task)
                json_write(staging/'initial.json',initial)
                try:
                    task.play_once()
                except RuntimeError as e:
                    if str(e).startswith(('inter-arm/object collision:',)) or 'waiting shoe is not outside' in str(e):
                        raise CandidateRejected(str(e)) from e
                    raise
                if not task.plan_success or not task.check_success():
                    raise CandidateRejected('native source expert unsuccessful')
                p=task.recorded_expert.program()
                if min(len(p.left),len(p.right))==0: raise AssertionError('empty independent lane')
                p.save(staging/'program.npz')
                variants=[('concurrent',None),('left_first',0.0),('right_first',1.0),
                          ('uniform_0',slot/(2*args.slots)),('uniform_1',(slot+args.slots)/(2*args.slots))]
                results=[]
                for variant,u in variants:
                    stage=variant
                    setup(task,cfg,seed)
                    target=staging/variant; target.mkdir()
                    r=replay(task,p,name,u,target/'episode.hdf5',initial)
                    r.update(seed=seed,slot=slot,variant=variant,camera_config_sha256=cfg_hash,**identity)
                    json_write(target/'result.json',r)
                    results.append(r)
                    print(json.dumps(dict(phase='variant_success',task=name,seed=seed,slot=slot,variant=variant)),flush=True)
                destination=directory/'episodes'/f'{slot:03d}'
                destination.parent.mkdir(exist_ok=True)
                if destination.exists(): raise FileExistsError(destination)
                staging.rename(destination)
                row=dict(slot=slot,seed=seed,path=str(destination),variants=results,
                         control_hashes=p.hashes(),camera_config_sha256=cfg_hash)
                accepted.append(row); json_write(manifest_path,accepted)
                outcome=dict(seed=seed,slot=slot,status='accepted',seconds=time.monotonic()-attempt_start)
            except CandidateRejected as e:
                outcome=dict(seed=seed,slot=slot,status='rejected',stage=stage,reason=str(e),
                             seconds=time.monotonic()-attempt_start)
                json_write(directory/'failures'/f'seed-{seed}.json',outcome)
                shutil.rmtree(staging)
            with attempts_path.open('a') as f: f.write(json.dumps(outcome)+'\n')
            json_write(output/'progress.json',dict(task=name,accepted=len(accepted),target=args.slots,
                       last_attempt=outcome,elapsed_s=time.monotonic()-start))
            print(json.dumps(dict(phase='candidate_result',task=name,accepted=len(accepted),**outcome)),flush=True)
            gc.collect()
        if len(accepted)!=args.slots:
            raise RuntimeError(f'{name}: only {len(accepted)} common seeds after {args.max_candidates} candidates')
        print(json.dumps(dict(phase='task_success',task=name,accepted=len(accepted),episodes=5*len(accepted))),flush=True)
        del task
        gc.collect()
    json_write(output/'collection-complete.json',dict(tasks=tasks,slots=args.slots,episodes=len(tasks)*args.slots*5))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ctr-root',type=Path,required=True)
    p.add_argument('--task',choices=['all',*TASKS],default='all')
    p.add_argument('--slots',type=int,default=50)
    p.add_argument('--max-candidates',type=int,default=500)
    args=p.parse_args()
    if args.slots<1 or args.max_candidates<args.slots: p.error('invalid candidate/slot limit')
    collect(args)

if __name__=='__main__': main()
