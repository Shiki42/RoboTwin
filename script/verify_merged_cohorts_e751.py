from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,sys
import numpy as np
sys.path.insert(0,'/home/coder/share/robotwin-paired-merge-e751/script')
from merge_paired_cohorts import read_tables,sha,video_info,verify_sums,PARENTS
from export_paired_lerobot import dump
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root=Path('/home/coder/share/ctr-paired-merge-e751')
parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['pick_dual_bottles','scan_object','place_dual_shoes']);args=parser.parse_args()
if args.task:
    selected=[]
    for method,count in [('concurrent',100),('sequential',200),('uniform',200)]:
        name=f'ctr-{args.task.replace("_","-")}-{count}ep-{method}';path=root/'datasets'/name
        info=json.loads((path/'meta/info.json').read_text());selected.append(dict(repo_id='Shiki42/'+name,path=str(path),episodes=count,frames=info['total_frames'],method=method))
    assembly=dict(datasets=selected)
else:assembly=json.loads((root/'assembly-receipt.json').read_text())
receipts=[];seed_sets={} 
for ds in assembly['datasets']:
    path=Path(ds['path']);verify_sums(path)
    info=json.loads((path/'meta/info.json').read_text());mapping=json.loads((path/'meta/merge_episode_map.json').read_text())
    meta=read_tables(path,'meta/episodes').to_pylist();meta.sort(key=lambda x:x['episode_index'])
    data=read_tables(path,'data').to_pydict();n=len(data['index'])
    assert n==info['total_frames']==ds['frames'] and len(meta)==ds['episodes']
    np.testing.assert_array_equal(data['index'],np.arange(n))
    assert sorted(set(data['episode_index']))==list(range(ds['episodes']))
    arrays={k:np.asarray(v) for k,v in data.items()};stats=json.loads((path/'meta/stats.json').read_text())
    for key,value in arrays.items():
        assert np.isfinite(value).all(),key
        v=value.astype(np.float64);v=v[:,None] if v.ndim==1 else v
        for q in (.01,.1,.5,.9,.99):np.testing.assert_allclose(stats[key][f'q{int(q*100):02d}'],np.quantile(v,q,axis=0,method='linear'),rtol=0,atol=1e-12)
    mode=ds['method'];assert ('observation.arm_active_mask' in data)==(mode=='uniform')
    assert not {'retime.left_idle','retime.right_idle','retime.overlap'}&set(data)
    source_cache={};norm_cache={};video_hashes={};offset=0;boundaries={0,n-1}
    for row,ep in zip(mapping,meta,strict=True):
        e=row['episode_index'];assert ep['episode_index']==e
        length=row['frames'];assert ep['length']==length and ep['dataset_from_index']==offset and ep['dataset_to_index']==offset+length
        select=slice(offset,offset+length);assert np.all(arrays['episode_index'][select]==e)
        np.testing.assert_array_equal(arrays['frame_index'][select],np.arange(length))
        np.testing.assert_allclose(arrays['timestamp'][select],np.arange(length)/25,atol=2e-6,rtol=0)
        assert np.all(arrays['retime.source_seed'][select]==row['seed']) and np.all(arrays['retime.source_cohort'][select]==row['source_cohort'])
        np.testing.assert_allclose(arrays['retime.u'][select],row['u'],atol=1e-7,rtol=0)
        source=PARENTS[row['source_cohort']]/row['source_dataset'];norm=root/'normalized'/f'{row["source_dataset"]}-cohort{row["source_cohort"]}'
        if str(source) not in source_cache:
            parent_rows=read_tables(source,'meta/episodes').to_pylist();parent_rows.sort(key=lambda r:r['episode_index'])
            cursors=defaultdict(int)
            for parent_row in parent_rows:
                for cam in ('observation.images.top','observation.images.left_wrist','observation.images.right_wrist'):
                    pre='videos/'+cam;key=(cam,parent_row[pre+'/chunk_index'],parent_row[pre+'/file_index']);a=cursors[key];b=a+parent_row['length']
                    for suffix,expected in [('from_timestamp',a/25),('to_timestamp',b/25)]:
                        if abs(parent_row[pre+'/'+suffix]-expected)>1e-6:
                            assert source==PARENTS[1]/'scan_object-uniform' and cam=='observation.images.left_wrist' and parent_row['episode_index'] in (98,99)
                        parent_row[pre+'/'+suffix]=expected
                    cursors[key]=b
            source_cache[str(source)]={r['episode_index']:r for r in parent_rows}
            norm_cache[str(norm)]={k:np.asarray(v) for k,v in read_tables(norm,'data').to_pydict().items()}
        original=source_cache[str(source)][row['source_episode_index']];normal=norm_cache[str(norm)]
        original_select=np.asarray(normal['episode_index'])==row['source_episode_index']
        for key in data:
            if key not in ('index','episode_index','task_index'):np.testing.assert_array_equal(arrays[key][select],np.asarray(normal[key])[original_select])
        for cam in ('observation.images.top','observation.images.left_wrist','observation.images.right_wrist'):
            prefix='videos/'+cam;start=ep[prefix+'/from_timestamp'];end=ep[prefix+'/to_timestamp']
            assert abs(start-original[prefix+'/from_timestamp'])<1e-6 and abs(end-original[prefix+'/to_timestamp'])<1e-6
            assert abs((end-start)*25-length)<1e-3
            v=path/info['video_path'].format(video_key=cam,chunk_index=ep[prefix+'/chunk_index'],file_index=ep[prefix+'/file_index'])
            original_video=source/info['video_path'].format(video_key=cam,chunk_index=original[prefix+'/chunk_index'],file_index=original[prefix+'/file_index'])
            if str(v) not in video_hashes:
                assert sha(v)==sha(original_video);probe=video_info(v);assert (probe['width'],probe['height'],probe['r_frame_rate'])==(320,240,'25/1')
                video_hashes[str(v)]=int(probe['nb_frames']);boundaries.update([offset,offset+length-1])
            assert end*25<=video_hashes[str(v)]+1e-3
        if mode=='uniform':
            m=arrays['observation.arm_active_mask'][select];assert np.isin(m,[0,1]).all() and np.all(m.any(axis=1)) and np.all(np.diff(m,axis=0)>=0)
        offset+=length
    assert offset==n
    for cam in ('observation.images.top','observation.images.left_wrist','observation.images.right_wrist'):
        assert sum(count for v,count in video_hashes.items() if '/'+cam+'/' in v)==n
    if mode=='sequential':assert all(r['source_variant']=='left_first' for r in mapping[:100]) and all(r['source_variant']=='right_first' for r in mapping[100:])
    seeds={r['seed'] for r in mapping};assert len(seeds)==100
    task=mapping[0]['source_dataset'].rsplit('-',1)[0];seed_sets.setdefault(task,[]).append(seeds)
    if mode=='uniform':
        assert set(Counter(r['seed'] for r in mapping).values())=={2}
        assert Counter(round(r['u']*100) for r in mapping)==Counter({i:2 for i in range(100)})
        for seed in seeds:
            pair=[r for r in mapping if r['seed']==seed];assert abs(abs(pair[0]['u']-pair[1]['u'])-.5)<1e-12
    reader=LeRobotDataset(ds['repo_id'],root=path,video_backend='pyav')
    for index in sorted(boundaries):
        sample=reader[index];assert tuple(sample['action'].shape)==(14,)
        for cam in ('observation.images.top','observation.images.left_wrist','observation.images.right_wrist'):assert tuple(sample[cam].shape)==(3,240,320)
    receipts.append(dict(repo_id=ds['repo_id'],episodes=ds['episodes'],frames=n,videos=len(video_hashes),native_reader_rows=len(boundaries),checks_passed=True))
    print(json.dumps(receipts[-1]),flush=True)
for task,sets in seed_sets.items():assert len(sets)==3 and sets[0]==sets[1]==sets[2]
assert len(receipts)==(3 if args.task else 9) and sum(r['episodes'] for r in receipts)==(500 if args.task else 1500)
dump(root/(f'verification-{args.task}.json' if args.task else 'verification.json'),dict(status='checks_passed',datasets=receipts,episodes=sum(r['episodes'] for r in receipts),frames=sum(r['frames'] for r in receipts)))
