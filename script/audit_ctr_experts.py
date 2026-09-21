"""Verify the 16 explicit CTR expert review episodes and write a checksum report."""
import json,hashlib,sys
from pathlib import Path
import h5py,numpy as np,cv2
if len(sys.argv)<2:
    raise SystemExit('usage: audit_ctr_experts.py REVIEW_ROOT [--partial]')
root=Path(sys.argv[1]).resolve()
plan=json.loads((root/'review-plan.json').read_text())
partial='--partial' in sys.argv
if partial: plan=[item for item in plan if (Path(item['destination'])/'result.json').exists()]
rows=[];head_intrinsics={}
for item in plan:
 p=Path(item['destination']);r=json.loads((p/'result.json').read_text())
 assert r['success']
 source=json.loads((Path(item['source'])/'source.json').read_text())
 assert r['control_hashes']==source['hashes']
 with h5py.File(p/'episode.hdf5') as h:
  steps=h['physics_step'][:];reasons=h['physics/reasons'][:];indices=h['physics/source_index'][:];stages=h['physics/stage_index'][:]
  assert len(reasons)==r['physics_steps']==steps[-1]
  assert np.all(np.diff(steps)>0) and steps[0]==0
  assert 'retime/overlap' not in h
  assert len(h['observation/state'])==len(steps)==r['training_frames']+1
  assert len(h['joint_action/vector'])==len(steps)
  expected=np.zeros((len(reasons),2),dtype=bool)
  for si,schedule in enumerate(r['stages']):
   begin=schedule['start_step'];delay=schedule['delay_steps'];dur=schedule['durations']
   delayed=1 if schedule['first']=='left' else 0
   if schedule['independent']:
    expected[begin:begin+delay,delayed]=True
   for ai,side in enumerate(('left','right')):
    selected=(stages==si)&(reasons[:,ai]==0)
    assert np.array_equal(indices[selected,ai],np.arange(dur[side]))
   span=max(dur[side]+(delay if side!=schedule['first'] and dur[side] else 0) for side in ('left','right'))
   if si+1<len(r['stages']):assert r['stages'][si+1]['start_step']==begin+span
  assert np.array_equal(reasons==1,expected)
  expected_masks=np.array([np.all(expected[a:b],axis=0) for a,b in zip(steps[:-1],steps[1:])])
  actual=np.stack([h['retime/left_idle'][:],h['retime/right_idle'][:]],axis=1)
  assert np.array_equal(actual,expected_masks)
  if not partial:
   assert h['observation/arm_active_mask'].dtype==np.float32
   assert np.array_equal(h['observation/arm_active_mask'][:],(~actual).astype(np.float32))
  assert not np.any(np.all(actual,axis=1))
  head=h['observation/head_camera/intrinsic_cv'][:]
  head_intrinsics.setdefault(item['task'],head[0])
  assert np.allclose(head,head_intrinsics[item['task']],atol=1e-5)
  camera_receipt=json.loads(h.attrs['wrist_camera_receipt'])
  for side in ('left','right'):
   assert np.allclose(h['camera_validation/'+side+'_fovy_deg'][:],90,atol=1e-5)
   assert len(h['camera_validation/'+side+'_mount'])==len(steps)
   assert np.allclose(h['camera_validation/'+side+'_mount'][:],camera_receipt['gripper_to_camera_matrix'],atol=2e-6,rtol=0)
  if item['task']=='scan_object_ctr':
   assert len(r['scan_success_steps'])==1
   assert r['scan_success_steps'][0]==r['stages'][2]['start_step']
   pose=np.array(r['stage_geometry'][0]['actors']['object'][:3])
   center=np.array([-.03,-.02,.95]);offset=np.array(source['scan_offset'])
   assert np.linalg.norm(pose-(center+offset))<.015,(pose,center+offset)
   roi=np.array(source['indicator_projection']['scan_indicator'])
   x0,y0=np.ceil(roi[0]+5).astype(int);x1,y1=np.floor(roi[1]-5).astype(int)
   initial=cv2.imdecode(h['observation/head_camera/rgb'][0],1)[y0:y1,x0:x1].astype(float)
   final=cv2.imdecode(h['observation/head_camera/rgb'][-1],1)[y0:y1,x0:x1].astype(float)
   assert np.mean(initial[:,:,2]>1.5*(initial[:,:,1]+1))>.5
   assert np.mean(final[:,:,1]>1.5*(final[:,:,2]+1))>.5
  else:
   assert r['minimum_clearance_m']>=.1
   poses=source['initial']
   for a,b in (('red','blue'),('yellow','green')):
    left=np.array(poses[a]['pose'][:3]);right=np.array(poses[b]['pose'][:3]);left[0]*=-1
    assert np.allclose(left,right,atol=1e-5)
  rows.append(dict(task=r['task'],variant=r['variant'],success=True,frames=r['training_frames'],
                   idle_counts=r['idle_counts'],minimum_clearance_m=r['minimum_clearance_m']))
 assert (p/'preview.mp4').stat().st_size>0
if not partial: assert len(rows)==16
manifest={str(p.relative_to(root/'review')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'review').glob('*/*/*') if p.is_file()}
report=dict(status='checks_passed_user_review_pending',episodes=rows,checks=['source hashes','fixed initial scenes','stage barriers','delay-only masks','no overlap field','New FOV per-frame geometry','scan geometry and light colors','mirrored block scenes','10cm clearance'],sha256=manifest)
(root/('review/partial-audit.json' if partial else 'review/audit.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(episodes=len(rows),passed=True,minimum_blocks_clearance_m=min((r['minimum_clearance_m'] for r in rows if r['minimum_clearance_m'] is not None),default=None))))
