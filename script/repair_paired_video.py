"""Re-encode one verified short video from its original ordered HDF5 frames."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import cv2
import h5py
from export_paired_lerobot import sha, dump, CAMS


def inspect(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(path)]))['streams'][0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--camera',choices=list(CAMS),required=True)
    p.add_argument('--diagnostics',type=Path,required=True)
    args=p.parse_args();root=args.dataset
    key='observation.images.'+CAMS[args.camera]
    videos=list((root/'videos'/key).rglob('*.mp4'))
    if len(videos)!=1:raise ValueError('expected exactly one consolidated video')
    video=videos[0];info=json.loads((root/'meta/info.json').read_text())
    before=inspect(video)
    if int(before['nb_frames'])>=info['total_frames']:raise ValueError('video is not short')
    rows=json.loads((root/'meta/retime_manifest.json').read_text())
    assert [r['episode_index'] for r in rows]==list(range(info['total_episodes']))
    args.diagnostics.mkdir(parents=True,exist_ok=False)
    backup=args.diagnostics/'original-short.mp4';temp=args.diagnostics/'reencoded.mp4'
    count=0
    proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','320x240',
        '-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18',
        '-g','25','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    try:
        for row in rows:
            assert sha(row['raw_hdf5'])==row['raw_hdf5_sha256']
            with h5py.File(row['raw_hdf5']) as h:
                images=h['observation/'+args.camera+'/rgb']
                assert len(images)-1==row['frames']
                for i in range(row['frames']):
                    bgr=cv2.imdecode(images[i],cv2.IMREAD_COLOR)
                    if bgr is None or bgr.shape!=(240,320,3):raise ValueError('invalid source image')
                    proc.stdin.write(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes());count+=1
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    finally:
        if proc.poll() is None:proc.terminate();proc.wait()
    after=inspect(temp);assert count==info['total_frames']==int(after['nb_frames'])
    old_sha=sha(video);new_sha=sha(temp)
    os.replace(video,backup);os.replace(temp,video)
    with (root/'SHA256SUMS').open('w') as f:
        for item in sorted(root.rglob('*')):
            if item.is_file() and item.name!='SHA256SUMS':f.write(f'{sha(item)}  {item.relative_to(root)}\n')
    receipt=dict(status='repaired_from_original_frames',dataset=str(root),camera=key,frames=count,
        original_sha256=old_sha,reencoded_sha256=new_sha,before=before,after=after,backup=str(backup),
        source='original HDF5 frames in exported episode order; final observation excluded',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    dump(args.diagnostics/'receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
