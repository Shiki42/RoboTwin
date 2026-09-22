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
    parser.add_argument('--candidate-index',type=int)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    if args.candidate_index is not None:
        job=plan['candidates'][args.candidate_index]
        candidate(job,args.output)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=plan['runtime'];manifest=[];started=time.time()
    def run_slot(slot):
        jobs=[(i,j) for i,j in enumerate(plan['candidates']) if j['slot']==slot]
        for index,job in jobs:
            path=args.output/f'slot-{slot:03d}'/f'seed-{job["seed"]:04d}'
            path.parent.mkdir(parents=True,exist_ok=True)
            log=path.parent/f'seed-{job["seed"]:04d}.log'
            command=[runtime,str(Path(__file__).resolve()),'--plan',str(args.plan.resolve()),'--output',str(path),'--candidate-index',str(index)]
            with log.open('w') as stream:
                process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
            if process.returncode:
                raise RuntimeError(f'candidate infrastructure error; no retry: {log}, exit={process.returncode}')
            result=json.loads((path/'candidate.json').read_text())
            if result['status']=='accepted':return result
        raise RuntimeError(f'finite candidate list exhausted for slot {slot}')
    # The manifest's fixed slot order makes failures stop before another source.
    if plan['workers'] != 1:
        raise ValueError('this paired collection requires one sequential worker')
    for slot in range(plan['slots']):
        result=run_slot(slot);manifest.append(result)
        write(args.output/'manifest.json',manifest)
        print(json.dumps(dict(accepted=len(manifest),target=plan['slots'],slot=result['slot'],seed=result['seed'],elapsed_s=time.time()-started)),flush=True)
    write(args.output/'complete.json',dict(success=True,slots=len(manifest),episodes=sum(len(r['variants']) for r in manifest),elapsed_s=time.time()-started))

if __name__=='__main__':main()
