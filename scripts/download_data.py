"""Retrieve original public source files and enforce the immutable manifest."""
from pathlib import Path
import hashlib, json, urllib.request, os
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
target=ROOT/'data/raw';target.mkdir(exist_ok=True)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
for item in manifest['files']:
    dest=target/item['filename']
    if dest.exists():
        if digest(dest)!=item['sha256']:raise ValueError(f'Conflicting local file: {dest.name}')
        print(f'Already verified: {dest.name}');continue
    partial=dest.with_suffix(dest.suffix+'.part')
    try:
        with urllib.request.urlopen(item['url'],timeout=180) as r,partial.open('wb') as f:
            for chunk in iter(lambda:r.read(1048576),b''):f.write(chunk)
        if partial.stat().st_size!=item['bytes'] or digest(partial)!=item['sha256']:
            raise ValueError(f'Source checksum or size mismatch: {dest.name}')
        os.replace(partial,dest)
    finally:
        partial.unlink(missing_ok=True)
    print(f'Downloaded and verified: {dest.name}')
