"""Lossless bounded outputs over Modal's supported generator API.

One function invocation, no public upload or persistent storage. Chunks stay
below the SDK's blob threshold. This transport was chosen before any blob-host
request/denial; it does not bypass a denied network destination.
"""
import hashlib
from pathlib import Path

CHUNK_BYTES=512*1024
MAX_BYTES=128*1024**2
MAX_FILES=128  # 12 frames x 9 diagnostics + sidecars; still a finite set.
SUFFIXES={'.json','.npz','.png'}


def emit_artifacts(directory, *, suffixes=SUFFIXES, max_bytes=MAX_BYTES):
    files=sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in suffixes)
    if len(files)>MAX_FILES or sum(p.stat().st_size for p in files)>max_bytes:
        raise ValueError('artifact set exceeds bounded transfer')
    for path in files:
        sha=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(CHUNK_BYTES),b''):sha.update(block)
        yield {'kind':'file','name':path.name,'bytes':path.stat().st_size,'sha256':sha.hexdigest()}
        offset=0
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(CHUNK_BYTES),b''):
                yield {'kind':'chunk','offset':offset,'data':block}
                offset+=len(block)
        yield {'kind':'file_end'}
    yield {'kind':'complete','files':len(files)}


def receive_artifacts(events,directory, *, suffixes=SUFFIXES, max_bytes=MAX_BYTES):
    directory.mkdir(parents=True,exist_ok=False)
    active=None;handle=None;count=0;total=0;completed=False;names=set()
    try:
        for event in events:
            if completed:raise ValueError('trailing artifact data')
            kind=event.get('kind')
            if kind=='file':
                name=event['name'];size=event['bytes']
                if (active is not None or Path(name).name!=name or Path(name).suffix not in suffixes
                    or name in names or not isinstance(size,int) or size<0 or total+size>max_bytes or count>=MAX_FILES):
                    raise ValueError('invalid artifact header')
                names.add(name);total+=size;active=event;offset=0;sha=hashlib.sha256()
                partial=directory/(name+'.part');handle=partial.open('xb')
            elif kind=='chunk':
                data=event['data']
                if (active is None or not isinstance(data,bytes) or len(data)>CHUNK_BYTES
                    or event['offset']!=offset or offset+len(data)>active['bytes']):
                    raise ValueError('invalid artifact chunk')
                handle.write(data);sha.update(data);offset+=len(data)
            elif kind=='file_end':
                if active is None or offset!=active['bytes'] or sha.hexdigest()!=active['sha256']:
                    raise ValueError('incomplete or corrupt artifact')
                handle.close();handle=None;partial.rename(directory/active['name']);active=None;count+=1
            elif kind=='complete':
                if active is not None or event['files']!=count:raise ValueError('incomplete artifact set')
                completed=True
            else:raise ValueError('unexpected artifact event')
        if not completed:raise ValueError('artifact stream interrupted; partial files preserved')
    finally:
        if handle is not None:handle.close()
