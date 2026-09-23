"""Extract a verified 50-episode L2R prefix without changing decoded video pixels."""
import argparse,json,shutil,subprocess
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ctr.uniform50_release import repack_video,image_statistics,write_manifest
from export_ctr_batch import refresh_statistics,sha,dump

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    for job in json.loads(args.config.read_text()):
        parent=Path(job['source']);out=Path(job['output']);assert not out.exists();out.mkdir(parents=True)
        # Verify the exact published parent manifest before selecting any rows.
        manifest=Path(job['parent_manifest']);assert sha(manifest)==job['parent_manifest_sha256']
        for line in manifest.read_text().splitlines():
            digest,name=line.split('  ',1)
            path=Path(job['parent_card']) if name=='README.md' else parent/name
            assert sha(path)==digest,(name,'parent checksum mismatch')
        info=json.load(open(parent/'meta/info.json'));retime=json.load(open(parent/'meta/retime_manifest.json'));assert info['total_episodes']==len(retime)==100
        chosen=retime[:50]
        for i,r in enumerate(chosen):
            assert r['episode_index']==i and r['normalized_u']==0
            for stage in r['timing']:
                if stage['independent']:assert stage['first']=='left' and stage['fraction']==1
        assert all(r['timing'][0]['first']=='right' for r in retime[50:])
        assert not any('mask' in k or k.endswith('_idle') for k in info['features'])
        tables=[]
        for path in sorted((parent/'data').rglob('*.parquet')):
            table=pq.read_table(path);selected=table.filter(pc.less(table['episode_index'],50))
            if not len(selected):continue
            dest=out/path.relative_to(parent);dest.parent.mkdir(parents=True,exist_ok=True);pq.write_table(selected,dest)
            assert pq.read_table(dest).equals(selected);tables.append(selected)
        total=sum(len(t) for t in tables);assert total==sum(r['frames'] for r in chosen)
        full=pa.concat_tables(tables);np.testing.assert_array_equal(np.asarray(full['index']),np.arange(total))
        episodes=pa.concat_tables([pq.read_table(p) for p in sorted((parent/'meta/episodes').rglob('*.parquet'))]);episodes=episodes.filter(pc.less(episodes['episode_index'],50));assert len(episodes)==50
        rows=episodes.to_pylist();video_jobs={};cameras=[k for k,v in info['features'].items() if v['dtype']=='video']
        for row in rows:
            for camera in cameras:
                key='videos/'+camera;rel=info['video_path'].format(video_key=camera,chunk_index=row[key+'/chunk_index'],file_index=row[key+'/file_index'])
                entry=video_jobs.setdefault(rel,dict(camera=camera,end=0.,start=row[key+'/from_timestamp']))
                entry['end']=max(entry['end'],row[key+'/to_timestamp']);entry['start']=min(entry['start'],row[key+'/from_timestamp'])
            row['meta/episodes/chunk_index']=0;row['meta/episodes/file_index']=0
        histograms={c:np.zeros((3,256),dtype=np.int64) for c in cameras};videos=[]
        for rel,jobvideo in video_jobs.items():
            assert abs(jobvideo['start'])<1e-8
            n=round(jobvideo['end']*info['fps']);hist,receipt=repack_video(parent/rel,out/rel,0,n,info['fps']);histograms[jobvideo['camera']]+=hist;videos.append(dict(path=rel,**receipt));print(json.dumps(dict(repo=job['repo_id'],video=rel,frames=n,pixels_identical=True)),flush=True)
        dest=out/'meta/episodes/chunk-000/file-000.parquet';dest.parent.mkdir(parents=True);pq.write_table(pa.Table.from_pylist(rows,schema=episodes.schema),dest)
        shutil.copyfile(parent/'meta/tasks.parquet',out/'meta/tasks.parquet')
        info.update(total_episodes=50,total_frames=total,splits={'train':'0:50'})
        for camera in cameras:info['features'][camera]['info']['video.pix_fmt']='yuv420p'
        dump(out/'meta/info.json',info);dump(out/'meta/retime_manifest.json',chosen)
        for slot in sorted({r['source_slot'] for r in chosen}):shutil.copytree(parent/'meta/source_programs'/f'slot-{slot:03d}',out/'meta/source_programs'/f'slot-{slot:03d}')
        stats=json.load(open(parent/'meta/stats.json'))
        for camera,hist in histograms.items():stats[camera]={k:v for k,v in image_statistics(hist).items() if not k.startswith('q')}
        dump(out/'meta/stats.json',stats);refresh_statistics(out,tables)
        # Recompute independently from the selected parent rows as a release gate.
        stats=json.load(open(out/'meta/stats.json'))
        for key in full.column_names:
            values=np.asarray(full[key].to_pylist(),dtype=np.float64).reshape(total,-1)
            for q in (.01,.1,.5,.9,.99):np.testing.assert_array_equal(stats[key][f'q{int(q*100):02d}'],np.quantile(values,q,axis=0,method='linear'))
        receipt=dict(success=True,repo_id=job['repo_id'],episodes=50,frames=total,parent_repo=job['parent_repo'],parent_revision=job['parent_revision'],parent_manifest_sha256=job['parent_manifest_sha256'],parent_episode_indices=list(range(50)),method='l2r',idle_mask=False,numeric_rows_identical=True,decoded_pixels_identical=True,videos=videos)
        dump(out/'meta/subset-receipt.json',receipt);dump(out/'.ctr-revision.json',dict(schema='ctr.generated_dataset_revision.v1',repo_id=job['repo_id'],revision=sha(out/'meta/retime_manifest.json'),revision_kind='sha256-scene-and-control-manifest',parent_revision=job['parent_revision'],episodes=50,frames=total))
        write_manifest(out);print(json.dumps(dict(repo=job['repo_id'],complete=True,episodes=50,frames=total)),flush=True)
if __name__=='__main__':main()
