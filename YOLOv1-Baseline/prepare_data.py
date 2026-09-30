"""Download public VOC archives using torchvision's official URLs/checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
from protocol import ROOT as BASE_DIR

# URLs and MD5 values from torchvision's official VOC dataset implementation.
DATASET_YEAR_DICT = {
    '2007': {'url':'https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
             'filename':'VOCtrainval_06-Nov-2007.tar','md5':'c52e279531787c972589f7e41ab4ae64'},
    '2007-test': {'url':'https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
                  'filename':'VOCtest_06-Nov-2007.tar','md5':'b6e924de25625d8de591ea690078ad9f'},
    '2012': {'url':'https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
             'filename':'VOCtrainval_11-May-2012.tar','md5':'6cd6e144f989b92b3379bac3b3de84fd'}
}


def digest(path, algorithm='md5'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, destination, md5):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and digest(destination) == md5:
        return
    part = destination.with_suffix(destination.suffix + '.part')
    start = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-'} if start else {})
    with urllib.request.urlopen(request, timeout=60) as source:
        append = start > 0 and source.status == 206
        with part.open('ab' if append else 'wb') as out:
            n = start if append else 0
            mark = n // (100*1024*1024)
            while block := source.read(1024*1024):
                out.write(block)
                n += len(block)
                if n // (100*1024*1024) > mark:
                    mark = n // (100*1024*1024)
                    print(f'{destination.name}: {n / 1024**2:.0f} MiB', flush=True)
    if digest(part) != md5:
        raise RuntimeError(f'Checksum mismatch: {part}; remove this .part before retry')
    part.replace(destination)


def prepare(key):
    info = DATASET_YEAR_DICT[key]
    archive = BASE_DIR / 'data' / 'archives' / info['filename']
    marker = archive.with_suffix('.verified.json')
    if marker.exists():
        print(f'{key}: already verified/extracted', flush=True)
        return
    print(f'Downloading {info["url"]}', flush=True)
    download(info['url'], archive, info['md5'])
    root = (BASE_DIR / 'data').resolve()
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.name.startswith('VOCdevkit/'):
                continue
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError(f'Unsafe archive member: {member.name}')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source:
                    content = source.read()
                if target.exists():
                    if target.read_bytes() != content:
                        raise RuntimeError(f'Existing file differs, preserved: {target}')
                else:
                    target.write_bytes(content)
    marker.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'{key}: checksum and extraction OK', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['2007-test', '2012'], choices=['2007', '2007-test', '2012'])
    for key in p.parse_args().datasets:
        prepare(key)
