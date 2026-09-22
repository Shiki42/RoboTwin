"""Execute an explicit task/seed/variant manifest with frozen per-scene controls."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from scan50_recipe import candidate_job

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def candidate(job, destination):
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'script'))
    from collect_ctr_experts import collect_episode
    from envs.paired_timing import CandidateRejected
    task_name=job['task']
    task=getattr(importlib.import_module('envs.'+task_name),task_name)()
    base=dict(task=task_name,seed=job['seed'],u=None,first=None,delay_fraction=None)
    start=time.time()
    try:
        source=destination/'source'
        collect_episode(SimpleNamespace(**base,output=source,source=None,variant='source'),task)
        results=[]
        for spec in job['variants']:
            args=dict(base,**{k:v for k,v in spec.items() if k!='variant'})
            receipt=collect_episode(SimpleNamespace(**args,output=destination/spec['variant'],source=source,variant=spec['variant']),task)
            results.append(receipt)
    except CandidateRejected as error:
        receipt=dict(status='rejected',slot=job['slot'],seed=job['seed'],reason=str(error),elapsed_s=time.time()-start)
        write(destination/'candidate.json',receipt);print(json.dumps(receipt),flush=True)
        return
    receipt=dict(status='accepted',slot=job['slot'],seed=job['seed'],path=str(destination),variants=results,elapsed_s=time.time()-start)
    write(destination/'candidate.json',receipt);print(json.dumps(dict(status='accepted',slot=job['slot'],seed=job['seed'],elapsed_s=receipt['elapsed_s'])),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-index',type=int)
    parser.add_argument('--slot',type=int)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    if args.seed_index is not None or args.slot is not None:
        if args.seed_index is None or args.slot is None:
            raise ValueError('worker requires both seed-index and slot')
        candidate(candidate_job(plan,args.slot,args.seed_index),args.output)
        return
    if plan['workers'] != 1 or len(plan['slot_jobs']) != plan['slots']:
        raise ValueError('invalid sequential slot plan')
    pool=plan['candidate_seeds']
    if len(pool)!=len(set(pool)) or len(pool)<plan['slots']:
        raise ValueError('candidate pool must be unique and cover requested slots')
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=plan['runtime'];manifest=[];attempts=[];started=time.time()
    for index,seed in enumerate(pool):
        if len(manifest)==plan['slots']:
            break
        slot=len(manifest);job=candidate_job(plan,slot,index)
        path=args.output/f'slot-{slot:03d}'/f'seed-{seed:04d}'
        path.parent.mkdir(parents=True,exist_ok=True)
        log=path.parent/f'seed-{seed:04d}.log'
        command=[runtime,str(Path(__file__).resolve()),'--plan',str(args.plan.resolve()),
                 '--output',str(path),'--seed-index',str(index),'--slot',str(slot)]
        with log.open('w') as stream:
            process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
        if process.returncode:
            attempts.append(dict(seed=seed,slot=slot,status='infrastructure_error',exit_code=process.returncode,log=str(log)))
            write(args.output/'attempts.json',attempts)
            raise RuntimeError(f'candidate infrastructure error; no retry: {log}, exit={process.returncode}')
        result=json.loads((path/'candidate.json').read_text())
        attempts.append(dict(seed=seed,slot=slot,status=result['status'],reason=result.get('reason'),path=str(path)))
        write(args.output/'attempts.json',attempts)
        if result['status']=='accepted':
            if len(result['variants'])!=len(job['variants']):
                raise ValueError('accepted source lacks required variants')
            manifest.append(result);write(args.output/'manifest.json',manifest)
        elif result['status']!='rejected':
            raise ValueError('unknown candidate outcome')
        print(json.dumps(dict(accepted=len(manifest),target=plan['slots'],attempts=len(attempts),
            slot=slot,seed=seed,outcome=result['status'],elapsed_s=time.time()-started)),flush=True)
    if len(manifest)!=plan['slots']:
        raise RuntimeError(f'frozen candidate pool exhausted: {len(manifest)}/{plan["slots"]} qualified')
    write(args.output/'complete.json',dict(success=True,slots=len(manifest),
        episodes=sum(len(r['variants']) for r in manifest),attempts=len(attempts),
        selected_seeds=[r['seed'] for r in manifest],elapsed_s=time.time()-started))

if __name__=='__main__':main()
