"""Verify complete paired CTR datasets against the frozen qualified-scene recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pyarrow.parquet as pq
from export_ctr_batch import validate_commands,exact_stats


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--datasets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text());manifest=json.loads((args.collection/'manifest.json').read_text())
    complete=json.loads((args.collection/'complete.json').read_text());assert complete['success'] and len(manifest)==plan['slots']==50
    assert [r['slot'] for r in manifest]==list(range(50));seeds=[r['seed'] for r in manifest];assert len(set(seeds))==50
    attempts=json.loads((args.collection/'attempts.json').read_text());assert len({r['seed'] for r in attempts})==len(attempts)<=len(plan['candidate_seeds'])
    assert [r['seed'] for r in attempts]==plan['candidate_seeds'][:len(attempts)]
    assert [r['seed'] for r in attempts if r['status']=='accepted']==seeds
    args.output.mkdir(parents=True,exist_ok=False)
    review=[];raw={}
    for accepted in manifest:
        assert accepted['status']=='accepted'
        root=Path(accepted['path']);source=json.loads((root/'source/source.json').read_text())
        assert source['seed']==accepted['seed'] and source['success']
        assert np.isclose(np.linalg.norm(source['scan_offset']),.1,atol=1e-12)
        for spec in plan['slot_jobs'][accepted['slot']]['variants']:
            directory=root/spec['variant'];receipt=json.loads((directory/'result.json').read_text());assert receipt['success']
            assert receipt['control_hashes']==source['hashes']
            assert max(receipt['scan_translation_audit'][k] for k in ('max_orientation_error_deg','max_tilt_deg'))<=2
            assert all(max(v['max_rise_before_release_m'],v['max_drawup_before_release_m'])<=.008 for v in receipt['return_motion_audit'].values())
            with h5py.File(directory/'episode.hdf5') as h:
                validate_commands(h,root/'source/program.npz')
                idle=np.stack([h['retime/'+side+'_idle'][:] for side in ('left','right')],axis=1)
                assert h['observation/arm_active_mask'].dtype==np.float32
                assert np.array_equal(h['observation/arm_active_mask'][:],(~idle).astype(np.float32))
            raw[(accepted['slot'],spec['variant'])]=dict(path=directory,receipt=receipt,sha256=sha(directory/'episode.hdf5'))
            review.append(dict(task='scan_object_ctr',source=str(root/'source'),destination=str(directory)))
    # Reuse the established physics-clock/mask/barrier/camera/indicator audit.
    (args.output/'review').mkdir();(args.output/'review-plan.json').write_text(json.dumps(review,indent=2)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit_ctr_experts.py')),str(args.output),'--partial'],check=True)
    reports={}
    for method,group in plan['datasets'].items():
        root=args.datasets/method;receipt=json.loads((root/'meta/export-receipt.json').read_text());rows=json.loads((root/'meta/retime_manifest.json').read_text())
        info=json.loads((root/'meta/info.json').read_text());expected=group['episodes'];assert receipt['success'] and len(rows)==len(expected)==info['total_episodes']
        assert receipt['idle_mask']==(method=='ctr')
        for i,(actual,chosen) in enumerate(zip(rows,expected)):
            assert actual['episode_index']==i and actual['source_slot']==chosen['slot'] and actual['source_variant']==chosen['variant']
            assert actual['source_seed']==seeds[chosen['slot']]
            raw_row=raw[(chosen['slot'],chosen['variant'])]
            assert actual['source_hdf5_sha256']==raw_row['sha256'] and actual['control_hashes']==raw_row['receipt']['control_hashes']
            if chosen['u'] is not None:assert actual['normalized_u']==chosen['u']
        assert len({r['source_seed'] for r in rows})==50
        tables=[pq.read_table(f) for f in sorted((root/'data').rglob('*.parquet'))];assert sum(len(t) for t in tables)==info['total_frames']==receipt['frames']
        stats=json.loads((root/'meta/stats.json').read_text())
        for key,value in stats.items():
            if key.startswith('observation.images.'):
                assert not any(k.startswith('q') for k in value);continue
            values=np.concatenate([np.asarray(t[key].to_pylist(),dtype=np.float64) for t in tables])
            if values.ndim==1:values=values[:,None]
            correct=exact_stats(values)
            for name,expected_stat in correct.items():np.testing.assert_allclose(value[name],expected_stat,atol=1e-12,rtol=1e-12,err_msg=f'{method}/{key}/{name}')
        for table in tables:
            columns=table.column_names;assert 'retime.overlap' not in columns
            if method=='ctr':
                active=np.asarray(table['observation.arm_active_mask'].to_pylist(),dtype=np.float32)
                idle=np.stack([np.asarray(table['retime.'+side+'_idle'].to_pylist()).reshape(-1) for side in ('left','right')],axis=1)
                assert np.array_equal(active,(~idle).astype(np.float32))
            else:assert not any('mask' in k or k.endswith('_idle') for k in columns)
        files=0
        for line in (root/'SHA256SUMS').read_text().splitlines():
            digest,name=line.split('  ',1);assert sha(root/name)==digest,name;files+=1
        reports[method]=dict(episodes=len(rows),frames=info['total_frames'],verified_files=files,mask_enabled=method=='ctr',source_seeds=sorted({r['source_seed'] for r in rows}))
    result=dict(success=True,qualified_seeds=seeds,candidates_tried=len(attempts),raw_replays=len(raw),datasets=reports,plan_sha256=sha(args.plan),status='reported_audit_pending')
    (args.output/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
