"""Collect one explicit seed with timed native experts; no seed search/replay."""
import argparse
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['pick_dual_bottles', 'scan_object', 'place_dual_shoes'])
    p.add_argument('--right-start-offset-s', required=True, type=float)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    import yaml
    os.chdir(ROOT)
    cfg = yaml.safe_load((ROOT/'task_config/demo_clean.yml').read_text())
    embodiment = ROOT/'assets/embodiments/aloha-agilex'
    robot_cfg = yaml.safe_load((embodiment/'config.yml').read_text())
    cfg.update(task_name=args.task, task_config='timed_demo',
               left_robot_file=str(embodiment), right_robot_file=str(embodiment),
               left_embodiment_config=robot_cfg, right_embodiment_config=robot_cfg,
               dual_arm_embodied=True, need_plan=True, save_data=False,
               save_path=str(args.output.resolve()), save_freq=10,
               render_freq=0, right_start_offset_s=args.right_start_offset_s)
    args.output.mkdir(parents=True, exist_ok=False)
    name = args.task+'_timed'
    task = getattr(importlib.import_module('envs.'+name), name)()
    task.setup_demo(seed=args.seed, now_ep_num=0, **cfg)
    task.save_data = True
    info = task.play_once()
    success = bool(task.plan_success and task.check_success())
    print(json.dumps(dict(task=args.task, seed=args.seed, success=success)), flush=True)
    runtime_manifest = Path(sys.executable).parents[1]/'runtime_manifest.json'
    provenance = dict(host=socket.gethostname(), account=getpass.getuser(),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      python=sys.executable, exact_command=sys.argv,
                      renderer='sapien-default-raster',
                      gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).strip(),
                      runtime_manifest_sha256=hashlib.sha256(runtime_manifest.read_bytes()).hexdigest() if runtime_manifest.is_file() else None,
                      source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    (args.output/'result.json').write_text(json.dumps(dict(success=success, seed=args.seed,
                  task=args.task, provenance=provenance, **info), indent=2, default=str)+'\n')
    task.merge_pkl_to_hdf5_video()
    import h5py
    import numpy as np
    with h5py.File(args.output/'data/episode0.hdf5', 'a') as f:
        f.create_dataset('simulation_time_s', data=[r['time_s'] for r in info['timing']['frames']])
        f.attrs['right_start_offset_s'] = args.right_start_offset_s
    make_preview(args.output, info['timing']['frames'])
    task.close_env()
    if not success:
        raise RuntimeError('expert did not satisfy the native success criterion')


def make_preview(output, frames):
    """Stream synchronized three-camera video in simulation time, with phase labels."""
    import cv2
    import numpy as np
    import pickle
    fps = 25
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '960x300', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-crf', '20', str(output/'preview.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    index = 0
    cached = None
    for n in range(int(frames[-1]['time_s']*fps)+1):
        time_s = n/fps
        while index+1 < len(frames) and frames[index+1]['time_s'] <= time_s:
            index += 1
        if cached != index:
            with (output/'.cache/episode0'/f'{index}.pkl').open('rb') as f:
                obs = pickle.load(f)['observation']
            cameras = ['head_camera', 'left_camera', 'right_camera']
            tiles = [cv2.resize(obs[name]['rgb'], (320,240)) for name in cameras]
            canvas = np.zeros((300,960,3), dtype=np.uint8)
            canvas[:240] = np.concatenate(tiles, axis=1)
            cached = index
        frame = canvas.copy()
        phases = frames[index]['phases']
        cv2.putText(frame, f"t={time_s:.2f}s   LEFT: {phases['left']}", (10,263),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (240,240,240), 1)
        cv2.putText(frame, f"RIGHT: {phases['right']}", (10,287),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (240,240,240), 1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('preview encoding failed')


if __name__ == '__main__':
    main()
