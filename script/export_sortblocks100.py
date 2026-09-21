"""Stream the explicitly selected paired Blocks100 episodes into LeRobot v3."""
import argparse
import json
import time
from pathlib import Path
import subprocess
import shutil
import sys
import numpy as np
import cv2
import h5py
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig
from export_paired_lerobot import features as paired_features, CAMS, dump, sha, intervals

PROMPT='Place the red block to the left of the yellow block on the left side, and the blue block to the left of the green block on the right side.'
METHODS=('sequential','concurrent','ctr','mixed')

def recipe(method):
    if method!='mixed':return [(i,method) for i in range(100)]
    rows=[(i,'sequential') for i in range(34)]+[(i,'sequential') for i in range(50,83)]
    rows += [(i,'concurrent') for i in list(range(34,50))+list(range(83,100))]
    assert len(rows)==len({i for i,_ in rows})==100
    return rows

def features(method):
    result=paired_features()
    del result['retime.overlap']
    if method=='ctr':
        result['observation.arm_active_mask']=dict(dtype='float32',shape=(2,),names=['left','right'])
    else:
        del result['retime.left_idle'];del result['retime.right_idle']
    return result

def wait_for_slot(collection,slot,exit_path):
    path=collection/'manifest.json'
    while True:
        if path.exists():
            rows=json.loads(path.read_text());match=[r for r in rows if r['slot']==slot]
            if match:
                assert len(match)==1 and match[0]['status']=='accepted'
                return match[0]
        if exit_path.exists():
            raise RuntimeError(f'collection ended without accepted slot {slot}; exit={exit_path.read_text().strip()}')
        time.sleep(5)

def exact_stats(values):
    values=np.asarray(values,dtype=np.float64)
    assert values.ndim==2 and np.isfinite(values).all()
    result={k:v.tolist() for k,v in dict(mean=values.mean(axis=0),std=values.std(axis=0),min=values.min(axis=0),max=values.max(axis=0)).items()}
    for q in (.01,.1,.5,.9,.99):result[f'q{int(q*100):02d}']=np.quantile(values,q,axis=0,method='linear').tolist()
    result['count']=[len(values)]
    assert np.all(np.array(result['q01'])<=result['q99'])
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--collection-exit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS,required=True)
    args=parser.parse_args();target=args.output
    if target.exists():raise FileExistsError(target)
    repo_id='Shiki42/ctr-sortblocks-100ep-'+args.method
    dataset=LeRobotDataset.create(repo_id=repo_id,root=target,fps=25,robot_type='aloha_agilex',
        features=features(args.method),streaming_encoding=True,encoder_threads=1,metadata_buffer_size=1,
        rgb_encoder=RGBEncoderConfig(vcodec='h264',g=25,crf=18,preset='fast'))
    metadata=[];numeric={'observation.state':[],'action':[]};total=0;camera_intrinsics=None
    for episode,(slot,label) in enumerate(recipe(args.method)):
        accepted=wait_for_slot(args.collection,slot,args.collection_exit)
        directory=Path(accepted['path'])/label
        receipt=json.loads((directory/'result.json').read_text())
        source=Path(accepted['path'])/'source'
        source_receipt=json.loads((source/'source.json').read_text())
        assert receipt['success'] and receipt['minimum_clearance_m']>=.1
        assert receipt['control_hashes']==source_receipt['hashes']
        with h5py.File(directory/'episode.hdf5') as h:
            states=h['observation/state'][:-1].astype(np.float32)
            actions=h['joint_action/vector'][1:].astype(np.float32)
            n=len(states);assert n==len(actions)==receipt['training_frames'] and states.shape[1]==14
            assert np.isfinite(states).all() and np.isfinite(actions).all()
            assert np.allclose(h['simulation_time_s'][:-1],np.arange(n)/25,atol=1e-5,rtol=0)
            idle=np.stack([h['retime/left_idle'][:],h['retime/right_idle'][:]],axis=1)
            active=h['observation/arm_active_mask'][:]
            assert np.array_equal(active,(~idle).astype(np.float32)) and not np.any(np.all(idle,axis=1))
            camera_receipt=json.loads(h.attrs['wrist_camera_receipt'])
            intrinsics={}
            for side in ('left','right'):
                assert np.allclose(h['camera_validation/'+side+'_fovy_deg'][:],90,atol=1e-4,rtol=0)
                assert np.allclose(h['camera_validation/'+side+'_mount'][:],camera_receipt['gripper_to_camera_matrix'],atol=2e-6,rtol=0)
            for cam in CAMS:
                k=h['observation/'+cam+'/intrinsic_cv'][:];assert np.allclose(k,k[0],atol=1e-5)
                intrinsics[cam]=k[0].tolist()
            if camera_intrinsics is None:camera_intrinsics=intrinsics
            assert intrinsics==camera_intrinsics
            durations=receipt['stages'][0]['durations']
            u=(slot/100 if label=='ctr' else (0.0 if slot<50 else 1.0) if label=='sequential' else durations['left']/sum(durations.values()))
            for i in range(n):
                frame={'task':PROMPT,'observation.state':states[i],'action':actions[i]}
                for key,value in [('source_seed',accepted['seed']),('source_slot',slot),('grid_index',slot if label=='ctr' else -1)]:frame['retime.'+key]=np.array([value],dtype=np.int64)
                for key,value in [('u',u),('offset_s',receipt['actual_offset_steps']*receipt['physics_dt_s'])]:frame['retime.'+key]=np.array([value],dtype=np.float32)
                if args.method=='ctr':
                    frame['observation.arm_active_mask']=active[i].astype(np.float32)
                    for a,side in enumerate(('left','right')):frame['retime.'+side+'_idle']=np.array([idle[i,a]],dtype=bool)
                for cam,name in CAMS.items():
                    image=cv2.imdecode(h['observation/'+cam+'/rgb'][i],cv2.IMREAD_COLOR)
                    if image is None:raise ValueError('invalid source RGB')
                    frame['observation.images.'+name]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
            numeric['observation.state'].append(states);numeric['action'].append(actions);total+=n
            entry=dict(episode_index=episode,source_slot=slot,source_seed=accepted['seed'],source_variant=label,
                       frames=n,normalized_u=u,source_hdf5_sha256=sha(directory/'episode.hdf5'),control_hashes=receipt['control_hashes'],
                       minimum_clearance_m=receipt['minimum_clearance_m'],timing=receipt['stages'],source_commit=receipt['provenance']['commit'])
            if args.method=='ctr':entry['idle_intervals']={s:intervals(idle[:,a]) for a,s in enumerate(('left','right'))}
            metadata.append(entry)
        destination=target/'meta/source_programs'/f'slot-{slot:03d}'
        destination.mkdir(parents=True)
        shutil.copyfile(source/'program.npz',destination/'program.npz')
        dump(destination/'scene.json',dict(seed=accepted['seed'],initial=source_receipt['initial'],hashes=source_receipt['hashes']))
        print(json.dumps(dict(method=args.method,episodes=episode+1,frames=total)),flush=True)
    dataset.finalize()
    stats_path=target/'meta/stats.json';stats=json.loads(stats_path.read_text())
    for key,chunks in numeric.items():stats[key]=exact_stats(np.concatenate(chunks))
    dump(stats_path,stats)
    dump(target/'meta/retime_manifest.json',metadata)
    info=json.loads((target/'meta/info.json').read_text());assert info['total_episodes']==100 and info['total_frames']==total
    tables=[pq.read_table(p) for p in sorted((target/'data').rglob('*.parquet'))]
    assert sum(len(t) for t in tables)==total
    for table in tables:
        assert 'retime.overlap' not in table.column_names
        if args.method=='ctr':
            active=np.asarray(table['observation.arm_active_mask'].to_pylist(),dtype=np.float32)
            left=np.asarray(table['retime.left_idle'].to_pylist()).reshape(-1)
            right=np.asarray(table['retime.right_idle'].to_pylist()).reshape(-1)
            assert np.array_equal(active,np.stack([~left,~right],axis=1))
        else:assert not any('mask' in k or k.endswith('_idle') for k in table.column_names)
    export_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    dump(target/'meta/exact_global_statistics.json',dict(algorithm='numpy.quantile',method='linear',values='published float32 rows promoted to float64',
        sampling='all valid training rows once; no padding; no per-episode quantile averaging',quantiles=[.01,.1,.5,.9,.99],features=['observation.state','action'],frames=total,stats_sha256=sha(stats_path)))
    dump(target/'.ctr-revision.json',dict(schema='ctr.generated_dataset_revision.v1',repo_id=repo_id,origin='generated native expert',
        revision=sha(target/'meta/retime_manifest.json'),revision_kind='sha256-scene-and-control-manifest',exporter_commit=export_commit,episodes=100,frames=total))
    dump(target/'meta/export-receipt.json',dict(success=True,repo_id=repo_id,episodes=100,frames=total,fps=25,method=args.method,
        idle_mask=args.method=='ctr',exporter_commit=export_commit,camera_intrinsics=camera_intrinsics,source_slots=[s for s,_ in recipe(args.method)]))
    with (target/'SHA256SUMS').open('w') as f:
        for path in sorted(target.rglob('*')):
            if path.is_file() and path.name!='SHA256SUMS' and '.cache' not in path.parts:f.write(f'{sha(path)}  {path.relative_to(target)}\n')
    print(json.dumps(dict(method=args.method,complete=True,episodes=100,frames=total)),flush=True)

if __name__=='__main__':main()
