#!/usr/bin/env python3
"""Download the pinned Open Model Zoo CPU model, verifying both files."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = 'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/person-detection-0202/FP16-INT8/'
FILES = {
    'person-detection-0202.xml': '5945dc0778915c41d3d5cbb77d4780dbb1db2a4a91a0f223c653f84e3a0a23be5e69651249549633825e426c5e939f67',
    'person-detection-0202.bin': 'f3c64f38b325f6054daa42132aa382a539c22df0082e73444a0af0bbd8786a36b455492e79e46fb8773baf8bf4597f17',
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('models/cpu-person'))
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        target = a.output_dir / name
        if target.exists():
            if hashlib.sha384(target.read_bytes()).hexdigest() != digest:
                raise SystemExit(f'Existing file has the wrong checksum: {target}; choose a fresh directory')
            continue
        temporary = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(BASE + name, timeout=60) as src, temporary.open('wb') as dst:
            while chunk := src.read(1024 * 1024): dst.write(chunk)
        if hashlib.sha384(temporary.read_bytes()).hexdigest() != digest:
            raise SystemExit(f'Download checksum mismatch: {temporary}')
        temporary.replace(target)
    model = a.output_dir / 'person-detection-0202.xml'
    model.with_suffix('.provenance.json').write_text(json.dumps({
        'model': 'Open Model Zoo person-detection-0202 FP16-INT8',
        'source': BASE, 'checksums_sha384': FILES,
        'license_reference': 'https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE',
        'input_shape': [1, 3, 512, 512], 'preprocessing': 'BGR raw pixels, direct resize; no /255',
        'note': 'Different model from YOLOv8s. Deployment quality needs validation.'
    }, indent=2) + '\n')
    print(model.resolve())

if __name__ == '__main__': main()
