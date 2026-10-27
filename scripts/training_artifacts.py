"""Bounded private transport for original reconstruction stage artifacts."""
from pathlib import Path
import stat
import zipfile

MAX_BYTES=512*1024**2
MAX_FILES=2048

def pack(source,destination,max_bytes=MAX_BYTES):
    paths=sorted(source.rglob('*'));files=[p for p in paths if p.is_file()]
    if any(p.is_symlink() for p in paths) or len(files)>MAX_FILES or sum(p.stat().st_size for p in files)>max_bytes:
        raise ValueError('unsafe or oversized stage artifacts')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:z.write(p,p.relative_to(source).as_posix())

def unpack(source,destination,max_bytes=MAX_BYTES):
    with zipfile.ZipFile(source) as z:
        entries=z.infolist();names=set();total=0
        if len(entries)>MAX_FILES:raise ValueError('too many stage files')
        for info in entries:
            p=Path(info.filename);mode=info.external_attr>>16
            if (p.is_absolute() or '..' in p.parts or '\\' in info.filename or info.filename in names
                or stat.S_ISLNK(mode) or info.flag_bits&1):raise ValueError('unsafe stage archive entry')
            names.add(info.filename);total+=info.file_size
            if total>max_bytes:raise ValueError('stage archive exceeds decompressed bound')
        destination.mkdir(parents=True,exist_ok=False)
        for info in entries:
            out=destination/info.filename
            if info.is_dir():out.mkdir(parents=True,exist_ok=True)
            else:
                out.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,out.open('xb') as dst:
                    for b in iter(lambda:src.read(1024**2),b''):dst.write(b)
