"""Generate synchronized preview videos from accepted physical observations."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import h5py
import numpy as np

TASKS=('pick_dual_bottles','scan_object','place_dual_shoes')
CAMS=('head_camera','left_camera','right_camera')


def preview(directory,target):
    r=json.loads((directory/'result.json').read_text())
    with h5py.File(directory/'episode.hdf5') as h:
        n=len(h['joint_action/vector'])-1
        reasons=h['physics/reasons'][:]
        names=json.loads(h.attrs['reason_names'])
        steps=h['physics_step'][:]
        pipe=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','960x310','-r','25','-i','-','-an','-c:v','libx264','-threads','1',
            '-crf','20','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE)
        for i in range(n):
            tiles=[]
            for cam in CAMS:
                image=cv2.imdecode(h['observation/'+cam+'/rgb'][i],cv2.IMREAD_COLOR)
                tiles.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            frame=np.zeros((310,960,3),dtype=np.uint8);frame[:240]=np.concatenate(tiles,axis=1)
            why=reasons[steps[i]]
            texts=[f"{r['task']} | {r['variant']} | seed={r['seed']} | t={i/25:.2f}s | New FOVY90",
                   f"LEFT: {names[why[0]]}   RIGHT: {names[why[1]]}",
                   f"right-start offset={r['actual_offset_steps']*r['physics_dt_s']:.3f}s"]
            for line,text in enumerate(texts):
                cv2.putText(frame,text,(8,260+21*line),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
            pipe.stdin.write(frame.tobytes())
        pipe.stdin.close()
        if pipe.wait():raise RuntimeError('preview encoding failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    args=p.parse_args();out=args.source/'previews';out.mkdir(exist_ok=True)
    selected=[]
    for task in TASKS:
        groups=json.loads((args.source/task/'manifest.json').read_text())
        if not groups:continue
        for label in ('concurrent','left_first','right_first','uniform_0','uniform_1'):
            group=groups[len(groups)//2] if label.startswith('uniform') else groups[0]
            target=out/f'{task}-{label}.mp4'
            preview(Path(group['path'])/label,target)
            selected.append(dict(task=task,variant=label,seed=group['seed'],path=str(target)))
    (out/'index.json').write_text(json.dumps(selected,indent=2)+'\n')

if __name__=='__main__':main()
