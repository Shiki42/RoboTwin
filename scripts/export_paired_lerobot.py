"""Export accepted frozen-control groups with the native LeRobot v3 writer."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig

TASKS=('pick_dual_bottles','scan_object','place_dual_shoes')
VARIANTS=('concurrent','left_first','right_first','uniform')
CAMS=dict(head_camera='top',left_camera='left_wrist',right_camera='right_wrist')
PROMPTS=dict(pick_dual_bottles='Pick up two bottles.',scan_object='Scan the object.',place_dual_shoes='Put both shoes in the shoe box.')


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def intervals(values):
    values=np.asarray(values,dtype=bool)
    changes=np.diff(np.r_[False,values,False].astype(np.int8))
    return [[int(a),int(b)] for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]


def features():
    joints=[f'{s}_joint_{i}.pos' for s in ('left','right') for i in range(1,7)]
    joints=joints[:6]+['left_gripper.pos']+joints[6:]+['right_gripper.pos']
    result={k:dict(dtype='float32',shape=(14,),names=joints) for k in ('observation.state','action')}
    for name in ('left_idle','right_idle','overlap'):
        result['retime.'+name]=dict(dtype='bool',shape=(1,),names=None)
    for name in ('source_seed','source_slot','grid_index'):
        result['retime.'+name]=dict(dtype='int64',shape=(1,),names=None)
    for name in ('u','offset_s'):
        result['retime.'+name]=dict(dtype='float32',shape=(1,),names=None)
    for cam in CAMS.values():
        result['observation.images.'+cam]=dict(dtype='video',shape=(240,320,3),names=['height','width','channels'])
    return result


def selected_rows(manifest,variant,slots):
    if len(manifest)<slots:raise ValueError(f'only {len(manifest)} complete slots, expected {slots}')
    result=[]
    for slot in manifest[:slots]:
        variants=['uniform_0','uniform_1'] if variant=='uniform' else [variant]
        for label in variants:
            entry=next(r for r in slot['variants'] if r['variant']==label)
            result.append((slot,label,entry))
    return result


def tracking_audit(h,program_path,receipt):
    with np.load(program_path,allow_pickle=False) as f:
        programs={s:f[s] for s in ('left','right','tail')}
    hashes={s:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for s,a in programs.items()}
    if hashes!=receipt['control_hashes']:raise AssertionError('stored source program hash mismatch')
    if h.attrs['wrist_camera_preset']!='centered_fovy90':raise AssertionError('New FOV missing')
    camera_receipt=json.loads(h.attrs['wrist_camera_receipt'])
    for side in ('left','right'):
        if not np.allclose(h['camera_validation/'+side+'_fovy_deg'][:],90,atol=0.001,rtol=0):raise AssertionError('FOVY90 violation')
        if not np.allclose(h['camera_validation/'+side+'_mount'][:],camera_receipt['gripper_to_camera_matrix'],atol=2e-6,rtol=0):raise AssertionError('centered mount violation')
    reasons=h['physics/reasons'][:];indices=h['physics/source_index'][:]
    samples=h['physics_step'][:];state=h['observation/state'][:]
    tail=[e['step'] for e in receipt['events'] if e['event']=='scan_barrier_release']
    tail_start=tail[0] if tail else len(reasons)
    expected=state[0].copy();errors=[];cursor=0
    for frame,end in enumerate(samples):
        while cursor<end:
            for a,side in enumerate(('left','right')):
                if reasons[cursor,a]!=0:continue
                index=int(indices[cursor,a])
                row=programs['tail'][index,1:] if cursor>=tail_start else programs[side][index]
                if row[0]==0:expected[a*7:a*7+6]=row[2:8]
                else:expected[a*7+6]=row[14]
            cursor+=1
        errors.append(state[frame]-expected)
    error=np.asarray(errors)
    camera_hashes={}
    for cam in CAMS:
        intrinsic=h['observation/'+cam+'/intrinsic_cv'][:]
        if not np.array_equal(intrinsic,np.broadcast_to(intrinsic[0],intrinsic.shape)):
            raise AssertionError('camera intrinsics changed inside an episode')
        camera_hashes[cam]=hashlib.sha256(intrinsic[0].tobytes()).hexdigest()
    return dict(control_hashes=hashes,camera_intrinsic_hashes=camera_hashes,
                joint_tracking_max_abs_rad={s:float(np.abs(error[:,a*7:a*7+6]).max()) for a,s in enumerate(('left','right'))},
                joint_tracking_rms_rad={s:float(np.sqrt(np.mean(error[:,a*7:a*7+6]**2))) for a,s in enumerate(('left','right'))})


def export_one(source,output,task,variant,slots):
    root=source/task
    manifest=json.loads((root/'manifest.json').read_text())
    selected=selected_rows(manifest,variant,slots)
    target=output/f'{task}-{variant}'
    receipt_path=target/'export-receipt.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if old['episodes']!=len(selected):raise ValueError('existing export episode count differs')
        return old
    if target.exists():raise FileExistsError(f'incomplete export requires explicit recovery: {target}')
    dataset=LeRobotDataset.create(repo_id='CTR/'+target.name,root=target,fps=25,
            robot_type='aloha_agilex',features=features(),streaming_encoding=True,
            encoder_threads=1,metadata_buffer_size=1,
            rgb_encoder=RGBEncoderConfig(vcodec='h264',g=25,crf=18,preset='fast'))
    episode_meta=[];frames_total=0
    for episode,(slot,label,r) in enumerate(selected):
        directory=Path(slot['path'])/label
        source_file=directory/'episode.hdf5'
        with h5py.File(source_file) as h:
            states=h['observation/state'][:]
            targets=h['joint_action/vector'][:]
            if len(states)<2 or states.shape[1]!=14 or not np.isfinite(states).all():
                raise ValueError('invalid joint state sequence')
            masks=np.column_stack([h['retime/'+s+'_idle'][:] for s in ('left','right')])
            overlap=h['retime/overlap'][:]
            n=len(states)-1
            if len(masks)!=n or np.any(np.all(masks,axis=1)):raise ValueError('invalid idle masks')
            times=h['simulation_time_s'][:]
            if not np.allclose(times[:-1],np.arange(n)/25,rtol=0,atol=1e-4):
                raise ValueError('nonuniform training timestamps')
            u=r['u'] if r['u'] is not None else r['durations']['left']/sum(r['durations'].values())
            grid=slot['slot']+(slots if label=='uniform_1' else 0) if variant=='uniform' else -1
            for i in range(n):
                frame={'task':PROMPTS[task],'observation.state':states[i].astype(np.float32),
                       'action':targets[i+1].astype(np.float32)}
                for key,value in [('source_seed',r['seed']),('source_slot',r['slot']),('grid_index',grid)]:
                    frame['retime.'+key]=np.array([value],dtype=np.int64)
                for key,value in [('u',u),('offset_s',r['actual_offset_steps']*r['physics_dt_s'])]:
                    frame['retime.'+key]=np.array([value],dtype=np.float32)
                for key,value in [('left_idle',masks[i,0]),('right_idle',masks[i,1]),('overlap',overlap[i])]:
                    frame['retime.'+key]=np.array([value],dtype=bool)
                for original,name in CAMS.items():
                    image=cv2.imdecode(h['observation/'+original+'/rgb'][i],cv2.IMREAD_COLOR)
                    if image is None:raise ValueError('invalid JPEG observation')
                    frame['observation.images.'+name]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
            record=dict(episode_index=episode,source_seed=r['seed'],source_slot=r['slot'],variant=label,
                        grid_index=grid,normalized_u=u,frames=n,raw_hdf5=str(source_file),
                        raw_hdf5_sha256=sha(source_file),receipt=r,
                        idle_frame_intervals={s:intervals(masks[:,a]) for a,s in enumerate(('left','right'))},
                        overlap_frame_intervals=intervals(overlap))
            reasons=h['physics/reasons'][:]
            reason_names=json.loads(h.attrs['reason_names'])
            record['tracking_audit']=tracking_audit(h,Path(slot['path'])/'program.npz',r)
            record['physics_reason_intervals']={s:{reason:intervals(reasons[:,a]==k) for k,reason in enumerate(reason_names)}
                                                for a,s in enumerate(('left','right'))}
            episode_meta.append(record)
        frames_total+=n
        program=Path(slot['path'])/'program.npz'
        dest=target/'meta/source_programs'/f'seed-{r["seed"]}.npz'
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(program,dest)
            shutil.copyfile(Path(slot['path'])/'initial.json',dest.with_suffix('.scene.json'))
        print(json.dumps(dict(phase='export_episode',task=task,variant=variant,episode=episode+1,total=len(selected))),flush=True)
    dataset.finalize()
    dump(target/'meta/retime_manifest.json',episode_meta)
    del dataset
    info=json.loads((target/'meta/info.json').read_text())
    if info['total_episodes']!=len(selected) or info['total_frames']!=frames_total:
        raise AssertionError('LeRobot totals disagree')
    tables=[pq.read_table(p) for p in sorted((target/'data').rglob('*.parquet'))]
    if sum(len(t) for t in tables)!=frames_total:raise AssertionError('parquet frame count differs')
    for table in tables:
        l=np.asarray(table['retime.left_idle'].to_pylist()).reshape(-1)
        rr=np.asarray(table['retime.right_idle'].to_pylist()).reshape(-1)
        if np.any(l & rr):raise AssertionError('both-idle exported row')
    with (target/'SHA256SUMS').open('w') as f:
        for path in sorted(target.rglob('*')):
            if path.is_file() and path.name!='SHA256SUMS':f.write(f'{sha(path)}  {path.relative_to(target)}\n')
    receipt=dict(task=task,variant=variant,episodes=len(selected),frames=frames_total,
                 unique_seeds=sorted({r['seed'] for _,_,r in selected}),path=str(target),
                 codebase_version=info['codebase_version'],fps=25,success=True,
                 exporter_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    dump(receipt_path,receipt)
    print(json.dumps(dict(phase='export_success',**receipt)),flush=True)
    return receipt


def make_reports(source,output,slots):
    reports=source/'reports';reports.mkdir(exist_ok=True)
    records=[];summaries={}
    for task in TASKS:
        manifest=json.loads((source/task/'manifest.json').read_text())
        if len(manifest)!=slots:raise AssertionError('incomplete task')
        seeds=[s['seed'] for s in manifest]
        if len(set(seeds))!=slots:raise AssertionError('duplicate source seed')
        exported=[json.loads((output/f'{task}-{v}'/'export-receipt.json').read_text()) for v in VARIANTS]
        for r in exported:
            if r['unique_seeds']!=sorted(seeds):raise AssertionError('seed coverage mismatch')
        attempts=[json.loads(x) for x in (source/task/'attempts.jsonl').read_text().splitlines()]
        summaries[task]=dict(seeds=seeds,intersection=sorted(seeds),differences={v:[] for v in VARIANTS},
                            accepted=slots,candidates=len(attempts),rejected=[x for x in attempts if x['status']=='rejected'],
                            datasets=exported)
        for v in VARIANTS:
            for episode,(slot,label,r) in enumerate(selected_rows(manifest,v,slots)):
                with h5py.File(Path(slot['path'])/label/'episode.hdf5') as h:
                    reasons=h['physics/reasons'][:]
                dt=r['physics_dt_s']
                waits={s:float(np.count_nonzero(reasons[:,a]==4)*dt) for a,s in enumerate(('left','right'))}
                overlap_s=float(np.count_nonzero(np.all(reasons==0,axis=1))*dt)
                records.append(dict(task=task,dataset=v,episode_index=episode,seed=r['seed'],slot=r['slot'],
                    source_variant=label,u=r['u'],requested_offset_s=r['nominal_offset_steps']*r['physics_dt_s'],
                    actual_offset_s=r['actual_offset_steps']*r['physics_dt_s'],
                    left_wait_s=r['workspace_wait_steps']['left']*dt+waits['left'],
                    right_wait_s=r['workspace_wait_steps']['right']*dt+waits['right'],
                    left_workspace_wait_s=r['workspace_wait_steps']['left']*dt,
                    right_workspace_wait_s=r['workspace_wait_steps']['right']*dt,
                    left_scan_barrier_wait_s=waits['left'],right_scan_barrier_wait_s=waits['right'],
                    overlap_s=overlap_s,
                    frames=r['training_frames']))
    dump(reports/'seed-report.json',dict(tasks=summaries,episodes=records))
    with (reports/'seed-report.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# CTR 配对数据 seed 报告','',f'交付 {len(records)} 条 Episode，12 个数据集。Mixed 本轮未生成。',
           '', '| 任务 | 共同seed | 候选数 | 淘汰数 | Concurrent | 左先 | 右先 | Uniform |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,s in summaries.items():
        lines.append(f'| {task} | {slots} | {s["candidates"]} | {len(s["rejected"])} | {slots} | {slots} | {slots} | {2*slots} |')
        lines.extend(['',f'## {task}','', '共同 seed（按槽位顺序）：'+', '.join(map(str,s['seeds'])),
                      '', '四个集合的 seed 差集均为空。Uniform 中每个 seed 出现两次。'])
    lines.extend(['','每条 Episode 的偏移、必要等待及映射见同目录 CSV/JSON；失败原因保存在 JSON 中。',
                  '结果已报告；实验归档确认仍需单独的审计批准。'])
    (reports/'seed-report.md').write_text('\n'.join(lines)+'\n')
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=['all',*TASKS],default='all')
    p.add_argument('--variant',choices=['all',*VARIANTS],default='all')
    p.add_argument('--slots',type=int,default=50)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for task in TASKS if args.task=='all' else [args.task]:
        for variant in VARIANTS if args.variant=='all' else [args.variant]:
            export_one(args.source,args.output,task,variant,args.slots)
    if args.task=='all' and args.variant=='all':make_reports(args.source,args.output,args.slots)

if __name__=='__main__':main()
