"""Bundle control-clock traces and refresh complete per-dataset checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def bundle(root):
    count=0
    for manifest in sorted(root.glob('*/meta/retime_manifest.json')):
        dataset=manifest.parent.parent
        entries=json.loads(manifest.read_text())
        for entry in entries:
            relative=Path('meta/retime_source_indices')/f'episode_{entry["episode_index"]:06d}.npz'
            target=dataset/relative;target.parent.mkdir(exist_ok=True)
            with h5py.File(entry['raw_hdf5']) as h:
                np.savez_compressed(target,sample_physics_step=h['physics_step'][:],
                    physics_source_index=h['physics/source_index'][:],physics_reason=h['physics/reasons'][:])
            entry['source_index_trace']=str(relative)
        temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(entries,indent=2)+'\n');temp.replace(manifest)
        (dataset/'meta/CONTROL_TRACE.md').write_text(
            '# Control-time trace semantics\n\n'
            'sample_physics_step maps observations to the 250 Hz physics clock.\n'
            'Each training action spans adjacent observation steps.\n'
            'physics_source_index contains the next frozen-control row for each arm;\n'
            'it advances only when physics_reason is active (0). After the scan\n'
            'barrier, active indices refer to the shared tail program. Negative\n'
            'indices mean no active source row. End indices may equal lane length.\n'
            'Reason codes are active=0, start_delay=1, finished=2, workspace_wait=3,\n'
            'scan_barrier=4, cooperative_hold=5, settle=6.\n'
            'These are control indices, not pixel origins: all videos are fresh\n'
            'synchronized physical renders with calibrated centered FOVY90 wrists.\n')
        sums=dataset/'SHA256SUMS';temporary=dataset/'SHA256SUMS.tmp'
        with temporary.open('w') as out:
            for path in sorted(dataset.rglob('*')):
                if path.is_file() and path not in (sums,temporary):
                    out.write(f'{sha(path)}  {path.relative_to(dataset)}\n')
        temporary.replace(sums)
        count+=len(entries)
        print(json.dumps(dict(dataset=dataset.name,bundled_episodes=len(entries))),flush=True)
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print('BUNDLED_EPISODES',bundle(args.root),flush=True)

if __name__=='__main__':main()
