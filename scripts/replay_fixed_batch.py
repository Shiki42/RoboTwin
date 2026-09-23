"""Replay one CPU-frozen 750-episode list; no seed substitution or retries."""
import os,sys,json,subprocess,shutil,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('CTR_BATCH_OUTPUT','/home/coder/share/ctr-paired-rt5mm-20260911-output'))
SOURCE=Path('/home/coder/share/ctr-paired-datasets-newfov-20260911-output')
TASKS=['pick_dual_bottles','scan_object','place_dual_shoes']
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def worker(job):
 os.chdir(ROOT);sys.path.insert(0,str(ROOT))
 import numpy as np,sapien,importlib
 from envs.robot.robot import Robot
 from collect_paired_datasets import config,replay
 from envs.paired_timing import Program,CandidateRejected
 Robot.set_planner=lambda *a,**kw:None
 cls=getattr(importlib.import_module('envs.'+job['task']+'_timed'),job['task']+'_timed')
 class Task(cls):
  def together_open_gripper(self,*a,**kw):pass
  def check_stable(self):return True,[]
 t=Task();cfg=config(job['task']);cfg['need_plan']=False;t.setup_demo(seed=job['seed'],now_ep_num=0,**cfg)
 ini=json.loads(Path(job['initial']).read_text())
 for name,v in ini.items():
  if name=='robot_qpos':continue
  a=getattr(t,name);assert a.config==v['config'],'asset identity mismatch';p=v['pose'];a.actor.set_pose(sapien.Pose(p[:3],p[3:]))
 q=np.array(ini['robot_qpos']);n=len(q)//2
 for side,qq in [('left',q[:n]),('right',q[n:])]:
  ent=getattr(t.robot,side+'_entity');ent.set_qpos(qq);ent.set_qvel(np.zeros_like(qq))
  for joint,v in zip(ent.get_active_joints(),qq):joint.set_drive_target(float(v));joint.set_drive_velocity_target(0)
 dest=Path(job['destination']);dest.mkdir(parents=True,exist_ok=False)
 try:
  p=Program.load(job['program']);assert p.hashes()==job['control_hashes']
  r=replay(t,p,job['task'],job['u'],dest/'episode.hdf5',ini)
 except CandidateRejected as e:
  save(dest/'failure.json',dict(job=job,reason=str(e)));return 2
 r.update(seed=job['seed'],slot=job['slot'],variant=job['variant'],source_commit=job['source_commit'],camera_config_sha256=job['camera_config_sha256'],host='evo-rl',account='coder',gpu_uuid='GPU-66e7fd65-e8bd-9792-e71c-3fe30fcd5ee5',source_program_sha256=job['program_sha256'])
 r['oidn_loaded_libraries']=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'OpenImageDenoise' in line))
 r['runtime']=sys.executable
 save(dest/'result.json',r);return 0
def main():
 if len(sys.argv)>1:return worker(json.loads((OUT/'jobs.json').read_text())[int(sys.argv[1])])
 jobs=json.loads((OUT/'jobs.json').read_text());start=time.time();success=0;failures=[]
 for i,j in enumerate(jobs):
  with (OUT/'logs'/f'{i:04d}.log').open('w') as f: code=subprocess.call([sys.executable,__file__,str(i)],stdout=f,stderr=subprocess.STDOUT)
  log=(OUT/'logs'/f'{i:04d}.log').read_text(errors='replace')
  if any(message in log for message in ['OIDN Error','[error]','unsupported device type','invalid handle']):code=70
  if code==0:success+=1
  else:failures.append(dict(index=i,exitcode=code,task=j['task'],seed=j['seed'],variant=j['variant']))
  state=dict(status='running',completed=i+1,total=len(jobs),success=success,failures=failures,elapsed_s=time.time()-start,last_job=j)
  save(OUT/'progress.json',state);print(json.dumps({'completed':i+1,'success':success,'failed':len(failures),'task':j['task'],'seed':j['seed'],'variant':j['variant'],'elapsed_s':state['elapsed_s']}),flush=True)
  if code not in [0,2]:state['status']='infrastructure_failed';save(OUT/'progress.json',state);return code
 state['status']='reported';save(OUT/'progress.json',state);print('COLLECTION_RESULT',success,'/',len(jobs),flush=True)
 (OUT/'collection.exit').write_text('0' if success==750 else '2')
 for task in TASKS:
  old=json.loads((SOURCE/task/'manifest.json').read_text());rows=[]
  for row in old:
   d=OUT/task/'episodes'/f"{row['slot']:03d}";results=[]
   for v in row['variants']:
    p=d/v['variant']/'result.json'
    if p.exists():results.append(json.loads(p.read_text()))
   if len(results)==5:rows.append(dict(row,path=str(d),variants=results))
  save(OUT/task/'manifest.json',rows)
  accepted={row['seed'] for row in rows}
  (OUT/task/'attempts.jsonl').write_text(''.join(json.dumps({'seed':row['seed'],'slot':row['slot'],'status':'accepted' if row['seed'] in accepted else 'rejected','mode':'fixed_control_replay'})+'\n' for row in old))
 if success!=750:return 2
 export_python='/home/coder/share/piperx-native-10-20260910/lerobot/.venv/bin/python'
 export_env=os.environ.copy();export_env['CUDA_VISIBLE_DEVICES']='';export_env['PYTHONPATH']='/home/coder/share/ctr-paired-datasets-newfov-20260911-output/export-deps:/home/coder/share/lerobot/src';export_env.pop('LD_PRELOAD',None)
 subprocess.run([export_python,str(ROOT/'scripts/export_paired_lerobot.py'),'--source',str(OUT),'--output',str(OUT/'lerobot'),'--slots','50'],check=True,env=export_env)
 subprocess.run([export_python,str(ROOT/'scripts/bundle_paired_metadata.py'),'--root',str(OUT/'lerobot')],check=True,env=export_env)
 save(OUT/'delivery-complete.json',dict(status='reported',episodes=750,datasets=12,lifecycle='audit pending'))
 return 0
if __name__=='__main__':raise SystemExit(main())
