#!/usr/bin/env python3
"""Download and verify an exact synthetic demo bundle, or install a local copy."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()

def safe_member(member):
    path=PurePosixPath(member.name)
    return not path.is_absolute() and '..' not in path.parts and (member.isfile() or member.isdir())

def verify_folder(folder):
    manifest=json.loads((folder/'ASSET_MANIFEST.json').read_text())
    for row in manifest['files']:
        relative=PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        file=folder/relative
        if not file.is_file() or file.stat().st_size!=row['bytes'] or checksum(file)!=row['sha256']:
            raise ValueError('Asset verification failed: '+str(relative))
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['mini','full'],default='mini')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--archive',type=Path,help='Use an already downloaded release asset')
    a=p.parse_args()
    record=json.loads((ROOT/'assets/demo_assets.json').read_text())['profiles'][a.profile]
    target=(a.output_dir or ROOT/'data'/a.profile).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        manifest=verify_folder(target)
        if manifest['profile']!=a.profile:raise SystemExit('Existing folder is a different profile')
        marker=target/'.bundle.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('sha256')!=record['sha256']:
            raise SystemExit('Existing folder is not the selected release bundle; use a fresh output directory')
        print(target/'sources.json');return
    archive=a.archive
    if archive is None:
        cache=ROOT/'data/downloads';cache.mkdir(parents=True,exist_ok=True);archive=cache/record['file']
        if not archive.exists() or checksum(archive)!=record['sha256']:
            part=archive.with_suffix(archive.suffix+'.part')
            print('Downloading',record['url'],flush=True)
            with urllib.request.urlopen(record['url'],timeout=60) as src,part.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
            if checksum(part)!=record['sha256']:raise SystemExit('Download checksum mismatch')
            part.replace(archive)
    if archive.stat().st_size!=record['bytes'] or checksum(archive)!=record['sha256']:
        raise SystemExit('Archive does not match the selected release profile')
    with tempfile.TemporaryDirectory(prefix='.demo-',dir=target.parent) as tmp:
        folder=Path(tmp)
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers()
            if not all(safe_member(m) for m in members):raise SystemExit('Unsafe archive member')
            tar.extractall(folder,members=members)
        verify_folder(folder)
        (folder/'.bundle.json').write_text(json.dumps({'sha256':record['sha256'],'release':record['url']})+'\n')
        folder.rename(target)
    print(target/'sources.json')

if __name__=='__main__':main()
