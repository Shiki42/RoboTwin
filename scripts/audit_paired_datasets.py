"""Read-only physical-trajectory/camera/mask audit; writes only an audit receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

TASKS=('pick_dual_bottles','scan_object','place_dual_shoes')
CALIBRATION='bd7a9d918a775f3d59b70ca04d1a9f4151c9602557487947e2ee49d8a461b0e5'


def audit(source,required_slots=None):
    reports={}
    total_frames=0;episodes=0;max_mount_error=0.0
    for task in TASKS:
        manifest=source/task/'manifest.json'
        groups=json.loads(manifest.read_text()) if manifest.exists() else []
        if required_slots is not None and len(groups)!=required_slots:
            raise AssertionError(f'{task}: {len(groups)} != {required_slots} complete slots')
        reports[task]={'seeds':[g['seed'] for g in groups],'groups':len(groups),'episodes':5*len(groups)}
        for group in groups:
            with np.load(Path(group['path'])/'program.npz',allow_pickle=False) as data:
                program={k:data[k] for k in ('left','right','tail')}
            hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in program.items()}
            assert hashes==group['control_hashes']
            if task=='scan_object':
                assert set(program['left'][:,1]).issubset({0,1})
                assert set(program['right'][:,1]).issubset({0,1})
            for result in group['variants']:
                assert result['success'] and result['control_hashes']==hashes
                with h5py.File(Path(group['path'])/result['variant']/'episode.hdf5') as h:
                    assert h.attrs['wrist_camera_preset']=='centered_fovy90'
                    calibration=json.loads(h.attrs['wrist_camera_receipt'])
                    assert calibration['calibration_manifest_sha256']==CALIBRATION
                    for s in ('left','right'):
                        k=h[f'observation/{s}_camera/intrinsic_cv'][:]
                        np.testing.assert_allclose(k,np.broadcast_to([[120,0,160],[0,120,120],[0,0,1]],k.shape),atol=1e-3,rtol=0)
                        mounts=h[f'camera_validation/{s}_mount'][:]
                        error=float(np.max(np.abs(mounts-calibration['gripper_to_camera_matrix'])))
                        assert error<=2e-6
                        max_mount_error=max(max_mount_error,error)
                    steps=h['physics_step'][:];reasons=h['physics/reasons'][:];indices=h['physics/source_index'][:]
                    assert steps[0]==0 and steps[-1]==len(reasons)
                    assert np.all(np.diff(steps)>0) and np.all(np.diff(steps)<=10)
                    np.testing.assert_array_equal(steps[:-1],np.arange(len(steps)-1)*10)
                    assert np.isfinite(h['observation/state'][:]).all()
                    assert np.isfinite(h['joint_action/vector'][:]).all()
                    stored=np.column_stack([h['retime/'+s+'_idle'][:] for s in ('left','right')])
                    assert not np.any(stored.all(axis=1))
                    for frame,(a,b) in enumerate(zip(steps[:-1],steps[1:])):
                        expected=np.all((reasons[a:b]==1)|(reasons[a:b]==2),axis=0)
                        np.testing.assert_array_equal(stored[frame],expected)
                        assert bool(h['retime/overlap'][frame])==bool(np.any(np.all(reasons[a:b]==0,axis=1)))
                    barrier=[e['step'] for e in result['events'] if e['event']=='scan_barrier_release']
                    independent_end=barrier[0] if barrier else len(reasons)
                    if task=='scan_object':
                        assert len(barrier)==1
                        assert barrier[0]==max(e['step'] for e in result['events'] if e['event']=='arm_done')
                        assert not np.any(reasons[:independent_end]==2)
                    for a,s in enumerate(('left','right')):
                        active=reasons[:independent_end,a]==0
                        np.testing.assert_array_equal(indices[:independent_end,a][active],np.arange(len(program[s])))
                    owner=None
                    for event in result['events']:
                        if event['event']=='workspace_enter':assert owner is None;owner=event['arm']
                        elif event['event']=='workspace_exit':assert owner==event['arm'];owner=None
                    assert owner is None
                    total_frames+=len(steps);episodes+=1
    return dict(status='checks_passed',tasks=reports,episodes=episodes,observation_frames=total_frames,
                maximum_wrist_mount_error=max_mount_error,wrist_fovy_deg=90.0,
                calibration_manifest_sha256=CALIBRATION)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--require-slots',type=int);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=audit(args.source,args.require_slots)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
