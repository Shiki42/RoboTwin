"""Normalize and merge the two frozen paired cohorts using native LeRobot v3 aggregation."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.feature_utils import get_hf_features_from_features
from export_paired_lerobot import exact_stats, sha, dump

TASKS=('pick_dual_bottles','scan_object','place_dual_shoes')
VARIANTS=('concurrent','left_first','right_first','uniform')
PARENTS=(Path('/home/coder/share/ctr-paired-datasets-newfov-20260911-output/lerobot'),
         Path('/home/coder/share/ctr-paired-newseeds-e742/lerobot'))


def video_info(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,width,height,r_frame_rate','-of','json',str(path)]))['streams'][0]


def verify_sums(root):
    manifest=root/'SHA256SUMS'
    for line in manifest.read_text().splitlines():
        expected,name=line.split('  ',1)
        if sha(root/name)!=expected:raise AssertionError(f'source hash mismatch: {root/name}')
    return sha(manifest)


def read_tables(root,folder):
    return pa.concat_tables([pq.read_table(p) for p in sorted((root/folder).rglob('*.parquet'))])


def reconstruct(program,steps,reasons,indices,events):
    """Requested gripper targets at each sampled state; holds do not advance controls."""
    tail=[e['step'] for e in events if e['event']=='scan_barrier_release']
    tail_start=tail[0] if tail else len(reasons)
    targets=[];current=np.ones(2);cursor=0;first_command=np.full(2,len(reasons),dtype=np.int64)
    for end in steps:
        for tick in range(cursor,int(end)):
            for a,side in enumerate(('left','right')):
                if reasons[tick,a]!=0:continue
                index=int(indices[tick,a])
                if tick>=tail_start:
                    row=program['tail'][index];assert int(row[0])==a;row=row[1:]
                else:row=program[side][index]
                if row[0]==1:
                    first_command[a]=min(first_command[a],tick)
                    current[a]=np.clip(row[14],0,1)
        targets.append(current.copy());cursor=int(end)
    idle=np.array([np.all(reasons[a:b]==1,axis=0) for a,b in zip(steps[:-1],steps[1:])])
    assert not idle.all(axis=1).any()
    active=(~idle).astype(np.float32)
    assert np.all(np.diff(active,axis=0)>=0)
    return np.asarray(targets,dtype=np.float32)[1:],active,first_command


def rewrite_stats(root,features):
    tables=read_tables(root,'data');data=tables.to_pydict()
    old=json.loads((root/'meta/stats.json').read_text())
    stats={k:v for k,v in old.items() if k.startswith('observation.images.')}
    for k,v in stats.items():stats[k]={s:a for s,a in v.items() if not s.startswith('q')}
    for k in data:
        a=np.asarray(data[k],dtype=np.float64)
        if a.ndim==1:a=a[:,None]
        stats[k]=exact_stats(a)
    dump(root/'meta/stats.json',stats)
    for path in sorted((root/'meta/episodes').rglob('*.parquet')):
        rows=pq.read_table(path).to_pylist()
        for row in rows:
            for k in list(row):
                if k.startswith('stats/') and (not k.startswith('stats/observation.images.') or k.rsplit('/',1)[1].startswith('q')):del row[k]
            selection=np.asarray(data['episode_index'])==row['episode_index']
            assert selection.sum()==row['length']
            for k in data:
                a=np.asarray(data[k],dtype=np.float64)[selection]
                if a.ndim==1:a=a[:,None]
                for stat,value in exact_stats(a).items():row[f'stats/{k}/{stat}']=value
        pq.write_table(pa.Table.from_pylist(rows),path)
    dump(root/'meta/exact_global_statistics.json',dict(algorithm='numpy.quantile',method='linear',
        quantiles=[.01,.1,.5,.9,.99],frames=len(tables),sample_weight='each valid published row once',
        padding='excluded; no padded rows exist',mask_exclusion=False,image_quantiles='omitted',
        stats_sha256=sha(root/'meta/stats.json')))


def normalize(source,target,cohort,task,variant):
    if target.exists():raise FileExistsError(target)
    digest=verify_sums(source)
    target.mkdir(parents=True)
    for folder in ('meta','videos'):
        shutil.copytree(source/folder,target/folder)
    for p in list((target/'meta/episodes').rglob('*.parquet')):p.unlink()
    info=json.loads((source/'meta/info.json').read_text())
    features=info['features']
    for name in ('retime.left_idle','retime.right_idle','retime.overlap'):features.pop(name,None)
    if variant=='uniform':features['observation.arm_active_mask']=dict(dtype='float32',shape=[2],names=['left','right'])
    else:features.pop('observation.arm_active_mask',None)
    features['retime.source_cohort']=dict(dtype='int64',shape=[1],names=None)
    data=read_tables(source,'data').to_pydict()
    for name in ('retime.left_idle','retime.right_idle','retime.overlap'):data.pop(name,None)
    rows=json.loads((source/'meta/retime_manifest.json').read_text())
    action=np.asarray(data['action'],dtype=np.float32);original=action.copy();masks=[];mapping=[];corrections=0
    for row in rows:
        ep=row['episode_index'];select=np.asarray(data['episode_index'])==ep
        trace_path=source/row['source_index_trace'];program_path=source/'meta/source_programs'/f'seed-{row["source_seed"]}.npz'
        with np.load(trace_path,allow_pickle=False) as trace,np.load(program_path,allow_pickle=False) as program:
            steps=trace['sample_physics_step'];reasons=trace['physics_reason'];indices=trace['physics_source_index']
            target_grip,active,first_command=reconstruct(program,steps,reasons,indices,row['receipt']['events'])
        assert len(target_grip)==select.sum()==row['frames']
        old=action[select][:,[6,13]];different=np.abs(old-target_grip)>1e-6
        if cohort==1 and different.any():raise AssertionError('qualified E742 commands changed')
        if different.any():
            # Historical defect is restricted to stale zero labels before the first gripper command.
            assert np.all(old[different]==0) and np.all(target_grip[different]==1)
            assert not np.any(different & (steps[1:,None]>first_command[None,:]))
        corrections+=int(different.sum());part=action[select].copy();part[:,[6,13]]=target_grip;action[select]=part
        masks.append(active)
        r=row['receipt'];entry=dict(episode_index=ep,seed=row['source_seed'],source_cohort=cohort,
            source_dataset=source.name,source_episode_index=ep,source_slot=row['source_slot'],
            source_variant=row['variant'],frames=row['frames'],u=row['normalized_u'],
            offset_s=r['actual_offset_steps']*r['physics_dt_s'],requested_offset_s=r['nominal_offset_steps']*r['physics_dt_s'],
            starts=r['starts'],durations=r['durations'],physics_dt_s=r['physics_dt_s'],workspace_wait_steps=r['workspace_wait_steps'],
            control_hashes=r['control_hashes'],camera_config_sha256=r['camera_config_sha256'],
            source_manifest_sha256=digest,source_trace_sha256=sha(trace_path),source_program_sha256=sha(program_path))
        mapping.append(entry)
    np.testing.assert_array_equal(action[:,[0,1,2,3,4,5,7,8,9,10,11,12]],original[:,[0,1,2,3,4,5,7,8,9,10,11,12]])
    data['action']=action.tolist();data['retime.source_cohort']=[cohort]*len(action)
    data['retime.source_slot']=(np.asarray(data['retime.source_slot'])+cohort*50).tolist()
    if variant=='uniform':data['observation.arm_active_mask']=np.concatenate(masks).tolist()
    else:data.pop('observation.arm_active_mask',None)
    data_path=target/'data/chunk-000/file-000.parquet';data_path.parent.mkdir(parents=True)
    Dataset.from_dict(data,features=get_hf_features_from_features({k:{**v,'shape':tuple(v['shape'])} for k,v in features.items()})).to_parquet(data_path)
    metadata=read_tables(source,'meta/episodes').to_pylist()
    for row in metadata:row['data/chunk_index']=0;row['data/file_index']=0;row['meta/episodes/chunk_index']=0;row['meta/episodes/file_index']=0
    meta_path=target/'meta/episodes/chunk-000/file-000.parquet';meta_path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(metadata),meta_path)
    dump(target/'meta/info.json',info);rewrite_stats(target,features)
    dump(target/'meta/merge_episode_map.json',mapping)
    receipt=dict(source=str(source),cohort=cohort,task=task,variant=variant,source_manifest_sha256=digest,
                 corrected_gripper_values=corrections,episodes=len(rows),frames=len(action),observations='preserved byte-for-byte numeric values; no reconstruction of historical physical measurements')
    dump(target/'normalization-receipt.json',receipt)
    for path in sorted((target/'videos').rglob('*.mp4')):
        v=video_info(path);assert v['width']==320 and v['height']==240 and v['r_frame_rate']=='25/1'
        assert int(v['nb_frames'])==len(action),(path,v,len(action))
        assert sha(path)==sha(source/path.relative_to(target))
    print(json.dumps(dict(phase='normalized',**receipt)),flush=True)
    return target


def merge(roots,target,repo_id,method):
    aggregate_datasets([f'CTR/{p.name}' for p in roots],repo_id,roots=roots,aggr_root=target,
                       concatenate_videos=False,concatenate_data=False)
    mapping=[]
    for root in roots:
        for row in json.loads((root/'meta/merge_episode_map.json').read_text()):
            row=deepcopy(row);row['episode_index']=len(mapping);mapping.append(row)
    dump(target/'meta/merge_episode_map.json',mapping)
    info=json.loads((target/'meta/info.json').read_text());rewrite_stats(target,info['features'])
    assert len(mapping)==info['total_episodes']
    seeds=sorted({r['seed'] for r in mapping});assert len(seeds)==100
    if method=='sequential':assert all(r['source_variant']=='left_first' for r in mapping[:100]) and all(r['source_variant']=='right_first' for r in mapping[100:])
    counts=Counter(r['seed'] for r in mapping);assert set(counts.values())==({1} if method in ('concurrent','left_first','right_first') else {2})
    by_seed=defaultdict(list)
    for r in mapping:by_seed[str(r['seed'])].append({k:r[k] for k in ('episode_index','source_cohort','source_variant','u','offset_s','requested_offset_s','starts','durations','workspace_wait_steps')})
    dump(target/'meta/seed_timing_map.json',dict(seeds=seeds,by_seed=dict(by_seed)))
    fields=['episode_index','seed','source_cohort','source_dataset','source_episode_index','source_slot','source_variant','u','offset_s','requested_offset_s','frames']
    with (target/'meta/episode_seed_timing.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in mapping)
    dump(target/'.ctr-revision.json',dict(schema='ctr.local_frozen_dataset.v1',repo_id=repo_id,
        revision=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest(),source_revisions=sorted({r['source_manifest_sha256'] for r in mapping}),episodes=len(mapping),frames=info['total_frames']))
    title=repo_id.split('/')[-1]
    order='Episodes 0–99 (the first 100) are left-first; episodes 100–199 (the last 100) are right-first. Each block contains the original50 seeds followed by the E74250 new seeds.' if method=='sequential' else 'Original cohort first, then E742 cohort; the two seed sets are disjoint.'
    card=f'''---
tags:
- lerobot
- robotwin
- ctr
- aloha-agilex
---
# {title}

{len(mapping)} episodes, {info['total_frames']} frames, **100 unique scene seeds**, LeRobot v3,25FPS.

## Episode order / 数据集备注

{order}
'''
    if method=='sequential':card+='\n前100条为先左后右，后100条为先右后左；两个100条分组使用相同的100个seed。\n'
    card+='''
## Seeds and timing

The complete seed list and per-seed episode/timing mapping are in [seed_timing_map.json](meta/seed_timing_map.json),
[episode_seed_timing.csv](meta/episode_seed_timing.csv) and [merge_episode_map.json](meta/merge_episode_map.json).
`retime.source_seed` and `retime.source_cohort` are also stored on every numeric row.
Positive `offset_s` delays the right arm; negative values delay the left arm.
Nominal offsets are rounded to the nearest250Hz physical step. Workspace waits are separately recorded.
'''
    if method=='uniform':card+='''
Each of100seeds has exactly two original timings. Each50-seed source cohort used u=slot/100 and u=(slot+50)/100.
Thus the merged200episodes retain two copies of the0%–99% grid, not a newly sampled200-point grid.
The per-seed map explicitly lists both u values, requested/quantized offsets and episode indices.
For Scan Object, timing varies only grasp/lift; the fixed cooperative scan tail starts after both arms arrive.
For shoes, shared workspace waits do not advance frozen controls and remain supervised.
'''
    card+='''
## Cameras, labels and provenance

All sources use New FOV `centered_fovy90`:320×240, vertical90degrees, calibrated centered optical-axis mount;
head plus both wrist cameras,250Hz physics,25FPS recordings. Original raster rendering is retained.
This is a physical-video merge, with no resimulation, cross-time image composition or retiming of controls.
Source videos are copied without re-encoding, and video files remain separate across source boundaries.

The first cohort is the20260911 paired750 collection; the second is E742 paired750.
Historical first-cohort stale pre-command zero gripper targets are corrected to the frozen initial-open command;
all other action values and all observation values are preserved. First-cohort observations retain their original
capture/accessor provenance; unavailable historical raw physical qpos is not reconstructed. E742 records measured
qpos and normalized measured grippers. Per-source corrections and hashes are in meta/source_provenance.json.
Uniform alone includes observation.arm_active_mask:0 only for the initial artificial start delay,1 for all later
responsibilities including finished holds, scan barriers and workspace waits. Other methods have no IdleMask.
Numeric statistics are exact global linear quantiles over all valid published rows once, without padding or mask
exclusion; no mean of per-episode quantiles. Image quantiles are omitted.

The E742 Scan Uniform left-wrist video was restored from original HDF5 after a native exporter dropped one frame;
the verified repaired parent is used. Physical success and collision checks follow source receipts (2mm penetration gate).
Publication does not imply user audit approval or authorization for model training. No train/test split is added.

Full seed list: '''+', '.join(map(str,seeds))+'\n'
    (target/'README.md').write_text(card)
    return dict(repo_id=repo_id,path=str(target),episodes=len(mapping),frames=info['total_frames'],method=method,seeds=seeds)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='';out=args.output;results=[];normalization=[]
    for task in TASKS:
        sources={}
        for variant in VARIANTS:
            sources[variant]=[]
            for cohort,parent in enumerate(PARENTS):
                source=parent/f'{task}-{variant}';target=out/'normalized'/f'{task}-{variant}-cohort{cohort}'
                sources[variant].append(normalize(source,target,cohort,task,variant))
                normalization.append(json.loads((target/'normalization-receipt.json').read_text()))
        slug=task.replace('_','-')
        for direction in ('left_first','right_first'):
            merge(sources[direction],out/'intermediate'/f'{task}-{direction}',f'CTR/{task}-{direction}',direction)
        for method,count,roots in [('concurrent',100,sources['concurrent']),('sequential',200,[out/'intermediate'/f'{task}-{d}' for d in ('left_first','right_first')]),('uniform',200,sources['uniform'])]:
            name=f'ctr-{slug}-{count}ep-{method}';target=out/'datasets'/name
            result=merge(roots,target,'Shiki42/'+name,method);results.append(result)
            dump(target/'meta/source_provenance.json',[r for r in normalization if r['task']==task and (r['variant']==method or method=='sequential' and r['variant'] in ('left_first','right_first'))])
            with (target/'SHA256SUMS').open('w') as f:
                for file in sorted(target.rglob('*')):
                    if file.is_file() and file.name!='SHA256SUMS':f.write(f'{sha(file)}  {file.relative_to(target)}\n')
            print(json.dumps(dict(phase='merged',**result)),flush=True)
    dump(out/'assembly-receipt.json',dict(status='assembled',datasets=results,normalization=normalization,episodes=sum(x['episodes'] for x in results)))


if __name__=='__main__':main()
