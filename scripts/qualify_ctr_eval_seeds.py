"""Bounded native expert feasibility screening for frozen CTR evaluation seeds."""
import argparse,importlib,json,os,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from ctr_eval_seed_plan import accepted_list,ordered_prefix
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def worker(args,plan):
    os.chdir(ROOT)
    from collect_paired_datasets import config,setup
    from collect_ctr_experts import signature
    from envs.paired_timing import CandidateRejected
    task=getattr(importlib.import_module('envs.'+args.task),args.task)()
    job=plan['tasks'][args.task];assert args.seed in job['candidates'] and args.seed not in job['training_seeds']
    args.output.mkdir(parents=True,exist_ok=False);started=time.time();cfg=config(args.task);cfg.update(eval_mode=True,is_test=True,save_data=False)
    result=dict(task=args.task,seed=args.seed,source_commit=plan['source_commit'],native_success=False)
    try:
        setup(task,cfg,args.seed);result['initial']=signature(task)
        task.play_once()
        if not task.plan_success:raise CandidateRejected('expert planning failed')
        # Test native policy success independently of the expert's completion flag.
        task.return_complete=False;task.take_action_cnt=1
        task.eval_physics_steps=task.recorded_expert.timeline.step
        for _ in range(750):
            task.scene.step();task.eval_physics_steps+=1
            task.recorded_expert.safety.check(task.eval_physics_steps)
            if task.check_success():break
        if not task.check_success():raise CandidateRejected('native physical completion or stable return failed')
        assert not task.return_complete
        if args.task=='scan_object_ctr' and not task.scan_complete:raise CandidateRejected('scan was never completed')
        program=task.recorded_expert.program();program.save(args.output/'program.npz')
        result.update(status='accepted',native_success=True,source_control_hashes=program.hashes(),scan_complete=getattr(task,'scan_complete',None),evaluation=task.info['evaluation'],expert_completion_flag=task.return_complete,final=signature(task),physics_steps=task.eval_physics_steps,wrist_camera=task.wrist_camera_receipt)
    except CandidateRejected as error:
        result.update(status='rejected',reason=str(error),error_type=type(error).__name__)
    result['elapsed_s']=time.time()-started
    write(args.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial','final','wrist_camera','source_control_hashes')}),flush=True)
    if hasattr(task,'scene'):task.close_env()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task');p.add_argument('--seed',type=int);a=p.parse_args();plan=json.loads(a.plan.read_text())
    if a.task is not None:
        if a.seed is None:raise ValueError('worker needs seed')
        worker(a,plan);return
    a.output.mkdir(parents=True,exist_ok=False);started=time.time()
    results={name:{} for name in plan['tasks']};next_index={name:0 for name in plan['tasks']}
    target=plan['target'];committed=set();in_flight={}
    def receipt_row(path):
        r=json.loads(path.read_text())
        if r['status'] not in ('accepted','rejected'):raise ValueError('invalid worker outcome')
        return dict(seed=r['seed'],status=r['status'],reason=r.get('reason'),receipt=str(path),
                    receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_commit=r['source_commit'])
    for name,job in plan['tasks'].items():
        for saved in plan['resume_receipts'][name]:
            path=Path(saved['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==saved['sha256']
            row=receipt_row(path);assert row['seed'] not in results[name];results[name][row['seed']]=row
        count=len(results[name]);assert set(results[name])==set(job['candidates'][:count]);next_index[name]=count
    def prefix(name):return ordered_prefix(plan['tasks'][name]['candidates'],results[name],target)
    def selected(name):return accepted_list(prefix(name),plan['tasks'][name]['training_seeds'],target)
    def persist():
        progress={}
        for name,job in plan['tasks'].items():
            ordered=[results[name][seed] for seed in job['candidates'] if seed in results[name]]
            write(a.output/name/'attempts.json',ordered)
            chosen=selected(name);resolved=prefix(name)
            progress[name]=dict(accepted=len(chosen),attempted=len(ordered),resolved_prefix=len(resolved),active_workers=sum(n==name for n,_ in in_flight.values()))
            if len(chosen)==target and name not in committed:
                write(a.output/name/'seeds.json',dict(task=name,seeds=chosen,count=len(chosen),source_commit=plan['source_commit'],receipt_source_commits=sorted({r['source_commit'] for r in resolved}),scene_config=plan['scene_config'],training_seeds=job['training_seeds'],training_provenance=job['training_provenance'],selection='first100 native-expert-success seeds in candidate order, independent of worker completion order',candidate_limit=plan['candidate_limit'],candidates_examined=len(resolved),expert_success_rate=len(chosen)/len(resolved),protocol='expert play_once plus native released/supported/stationary success without expert completion flag',plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest()))
                committed.add(name)
        write(a.output/'progress.json',progress)
    def run_candidate(name,index):
        seed=plan['tasks'][name]['candidates'][index];dest=a.output/name/f'seed-{seed:04d}';dest.parent.mkdir(parents=True,exist_ok=True);log=dest.with_suffix('.log')
        command=[plan['runtime'],str(Path(__file__).resolve()),'--plan',str(a.plan.resolve()),'--output',str(dest),'--task',name,'--seed',str(seed)]
        with log.open('w') as stream:process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        if process.returncode:
            write(dest.parent/f'seed-{seed:04d}-infrastructure.json',dict(task=name,seed=seed,exit_code=process.returncode,log=str(log)))
            raise RuntimeError(f'qualification infrastructure failed: {name} seed{seed}; see {log}')
        return receipt_row(dest/'result.json')
    persist()
    with ThreadPoolExecutor(max_workers=sum(plan['workers'].values())) as executor:
        while True:
            for name,job in plan['tasks'].items():
                active=sum(n==name for n,_ in in_flight.values())
                while len(selected(name))<target and active<plan['workers'][name] and next_index[name]<len(job['candidates']):
                    index=next_index[name];next_index[name]+=1
                    in_flight[executor.submit(run_candidate,name,index)]=(name,index);active+=1
            if not in_flight:break
            finished,_=wait(in_flight,return_when=FIRST_COMPLETED)
            for future in finished:
                name,index=in_flight.pop(future);row=future.result();seed=row['seed'];assert seed not in results[name];results[name][seed]=row
                print(json.dumps(dict(task=name,seed=seed,outcome=row['status'],accepted=len(selected(name)),attempted=len(results[name]),elapsed_s=time.time()-started)),flush=True)
            persist()
    counts={name:len(selected(name)) for name in plan['tasks']}
    if any(v!=target for v in counts.values()):raise RuntimeError(f'bounded candidate pool exhausted: {counts}')
    write(a.output/'complete.json',dict(success=True,counts=counts,elapsed_s=time.time()-started,workers=plan['workers']))
if __name__=='__main__':main()
