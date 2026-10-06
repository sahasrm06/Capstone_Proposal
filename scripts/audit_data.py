"""Audit real files and reproduce predefined group and chronological allocations."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];raw=ROOT/'data/raw';out=ROOT/'results';out.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
for item in manifest['files']:
    h=hashlib.sha256()
    with (raw/item['filename']).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    assert h.hexdigest()==item['sha256'],f"Checksum mismatch: {item['filename']}"
d=pd.read_csv(raw/'dataset.csv');mapping=json.loads((raw/'mapping.json').read_text())
assert len(d)==440274 and len(d.columns)==78
assert set(d['class'].unique())=={0,1}
assert set(d['inspection_type'].unique())=={0,1,2,3,4}
features=[c for c in d if c.startswith('inspection_feat')]
mapped=set(sum(mapping.values(),[]))
assert len(features)==70 and mapped.issubset(set(features))
unmapped=sorted(set(features)-mapped)
assert unmapped==['inspection_feat14','inspection_feat15','inspection_feat35','inspection_feat36','inspection_feat37']
dt=pd.to_datetime(d['timestamp'],utc=True);d['day']=dt.dt.normalize()
days=sorted(d['day'].unique());assert len(days)==101
g=d.groupby('timestamp').agg(y=('class','max'),n=('class','size'),dominant=('inspection_type',lambda a:a.mode().min()))
g.index=pd.to_datetime(g.index,utc=True);g=g.sort_index();assert len(g)==39742
summary={'rows':len(d),'columns':list(d.columns[:-1]),'groups':len(g),'positive_groups':int(g.y.sum()),'days':len(days),
    'class':{str(k):int(v) for k,v in d['class'].value_counts().items()},'nulls':int(d.isna().sum().sum()),'unmapped_features':unmapped,
    'duplicates':int(d.drop(columns=['Unnamed: 0','day']).duplicated().sum()),'range':[str(g.index.min()),str(g.index.max())]}
for a,b,label in [(0,40,'train'),(40,60,'validation'),(60,101,'test'),(0,50,'early'),(50,101,'late')]:
    s=g[(g.index.normalize()>=days[a])&(g.index.normalize()<=days[b-1])]
    summary[label]={'groups':len(s),'positives':int(s.y.sum()),'start':str(days[a]),'end':str(days[b-1])}
(out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
pd.crosstab(g.dominant,g.y).to_csv(out/'dominant_type_counts.csv')
print(json.dumps({k:v for k,v in summary.items() if k!='columns'},indent=2))
