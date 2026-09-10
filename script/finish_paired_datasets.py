"""Finalize completed task groups while the single GPU collector continues."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASKS=('pick_dual_bottles','scan_object','place_dual_shoes')
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--slots',type=int,default=50)
    args=p.parse_args()
    base=[sys.executable,str(ROOT/'export_paired_lerobot.py'),'--source',str(args.source),
          '--output',str(args.output),'--slots',str(args.slots)]
    for task in TASKS:
        manifest=args.source/task/'manifest.json'
        while not manifest.exists() or len(json.loads(manifest.read_text()))<args.slots:
            exit_path=args.source/'collection.exit'
            if exit_path.exists():
                raise RuntimeError(f'collector exited before {task} reached {args.slots} slots: {exit_path.read_text()}')
            time.sleep(15)
        subprocess.run(base+['--task',task],check=True)
    subprocess.run(base,check=True)
    subprocess.run([sys.executable,str(ROOT/'preview_paired_datasets.py'),'--source',str(args.source)],check=True)
    (args.source/'delivery-complete.json').write_text(json.dumps(dict(
        status='reported',tasks=3,datasets=12,episodes=15*args.slots,
        reports=str(args.source/'reports'),datasets_root=str(args.output)),indent=2)+'\n')
    print('DELIVERY_COMPLETE',flush=True)

if __name__=='__main__':main()
