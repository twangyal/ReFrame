"""Fetch only pinned sample members from the author's public ZIP; no retries.

The manifest hashes were measured from ZIP CRC-verified author downloads, not
provided as authenticated upstream signatures. Asset publication rights remain
unverified. TLS verification is never disabled.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request
import zipfile


class ConfirmationForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}; self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs['action']; self.active = True
        if tag == 'input' and self.active and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form': self.active = False


def download_url(file_id):
    url = 'https://drive.google.com/uc?' + urllib.parse.urlencode({'export':'download','id':file_id})
    with urllib.request.urlopen(url, timeout=30) as response:
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return response.url
        html = response.read(131073)
    if len(html) > 131072: raise ValueError('unexpected download confirmation size')
    form = ConfirmationForm(); form.feed(html.decode('utf-8'))
    if (form.action != 'https://drive.usercontent.google.com/download'
            or form.fields.get('id') != file_id):
        raise ValueError('unexpected public download confirmation; inspect manually')
    return form.action + '?' + urllib.parse.urlencode(form.fields)


class RemoteZipReader(io.RawIOBase):
    """Bounded, exact HTTP ranges; never silently download the full archive."""
    def __init__(self, url, total):
        self.url = url; self.total = total; self.pos = 0; self.start = 0; self.cache = b''

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.total+offset
        if not 0 <= self.pos <= self.total: raise ValueError('seek outside archive')
        return self.pos

    def read(self, size=-1):
        if size < 0: size = self.total-self.pos
        size = min(size, self.total-self.pos)
        if not size: return b''
        if size > 16*1024**2: raise ValueError('single range exceeds 16 MiB limit')
        if not self.start <= self.pos or self.pos+size > self.start+len(self.cache):
            end = min(self.total-1, max(self.pos+size-1,self.pos+16*1024**2-1))
            request = urllib.request.Request(self.url, headers={'Range':f'bytes={self.pos}-{end}'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {self.pos}-{end}/{self.total}':
                    raise ValueError('server did not honor exact bounded range')
                self.start = self.pos; self.cache = response.read(end-self.pos+2)
                if len(self.cache) != end-self.pos+1: raise ValueError('range length mismatch')
        result = self.cache[self.pos-self.start:self.pos-self.start+size]
        self.pos += len(result)
        return result


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2), b''): sha.update(block)
    return sha.hexdigest()


def extract_verified(archive, members, destination):
    for member in members:
        rel = PurePosixPath(member['archive_path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe archive path')
        out = destination/Path(*rel.parts)
        if not out.resolve().is_relative_to(destination.resolve()): raise ValueError('path escapes destination')
        if out.exists():
            if out.stat().st_size != member['bytes'] or digest(out) != member['sha256']:
                raise ValueError('existing asset differs; refusing overwrite')
            continue
        info = archive.getinfo(member['archive_path'])
        if info.file_size != member['bytes']: raise ValueError('archive member size changed')
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_suffix(out.suffix+'.part')
        with archive.open(info) as src, partial.open('xb') as dst:
            for block in iter(lambda:src.read(1024**2), b''): dst.write(block)
        if partial.stat().st_size != member['bytes'] or digest(partial) != member['sha256']:
            raise ValueError('download digest differs; partial preserved for inspection')
        partial.rename(out)
        print(f'Verified {member["archive_path"]} ({member["bytes"]} bytes)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); manifest = json.loads(args.manifest.read_text())
    if sum(m['bytes'] for m in manifest['members']) > 300*1024**2:
        raise ValueError('sample exceeds bounded extraction size')
    with zipfile.ZipFile(RemoteZipReader(download_url(manifest['file_id']), manifest['archive_total_bytes'])) as archive:
        extract_verified(archive, manifest['members'], args.out)


if __name__ == '__main__': main()
