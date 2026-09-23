from pathlib import Path
from collections import defaultdict
import json,sys
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,'/home/coder/share/robotwin-paired-merge-e751/script')
from export_paired_lerobot import sha,dump
root=Path('/home/coder/share/ctr-paired-merge-e751');dataset=root/'datasets/ctr-scan-object-200ep-uniform'
paths=sorted((dataset/'meta/episodes').rglob('*.parquet'));assert len(paths)==1
path=paths[0];before=sha(path);table=pq.read_table(path);rows=table.to_pylist();rows.sort(key=lambda r:r['episode_index'])
cursors=defaultdict(int);changes=[]
for row in rows:
    for camera in ('top','left_wrist','right_wrist'):
        prefix='videos/observation.images.'+camera;key=(camera,row[prefix+'/chunk_index'],row[prefix+'/file_index']);start=cursors[key];end=start+row['length']
        for suffix,value in [('from_timestamp',start/25),('to_timestamp',end/25)]:
            field=prefix+'/'+suffix
            if abs(row[field]-value)>1e-6:
                changes.append(dict(episode_index=row['episode_index'],field=field,before=row[field],after=value));row[field]=value
        cursors[key]=end
assert len(changes)==3 and {c['episode_index'] for c in changes}=={198,199} and all('left_wrist' in c['field'] for c in changes),changes
pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path)
receipt=dict(status='repaired',cause='E742 original exporter metadata retained a one-frame short episode after its video had been reconstructed',source_metadata_sha256=before,repaired_metadata_sha256=sha(path),changes=changes,rule='cumulative exact episode frame counts per unchanged physical video',images_changed=False,numeric_actions_changed=False)
dump(dataset/'meta/timestamp_repair.json',receipt);dump(root/'timestamp-repair-receipt.json',receipt)
card=dataset/'README.md';card.write_text(card.read_text()+'\n## Timestamp repair in this merge\n\nThe repaired E742 left-wrist video contained all frames, but its historical metadata retained a one-frame-short episode. Three timestamp fields in merged episodes198–199 were rebuilt from exact cumulative frame counts; images and numeric actions are unchanged. See [timestamp_repair.json](meta/timestamp_repair.json).\n')
with (dataset/'SHA256SUMS').open('w') as f:
    for p in sorted(dataset.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':f.write(f'{sha(p)}  {p.relative_to(dataset)}\n')
print(json.dumps(receipt))
