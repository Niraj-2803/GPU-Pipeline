#!/usr/bin/env python3
"""Exit nonzero if a run fails the selected throughput, latency or metadata checks."""
import argparse
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camex_pipeline.common import validate_frame

def failures(summary, minimum_fps, processing_ms, frame_age_ms=None, seconds=None):
    errors = []
    native = summary.get('native', {})
    if summary.get('error') or summary.get('validation_errors') or summary.get('metadata_dropped'):
        errors.append('runtime, schema or metadata-drop error')
    if native.get('interrupted') or native.get('native_exit_code', 0): errors.append('run interrupted or native process failed')
    if not summary.get('per_camera') or any(c['fps'] < minimum_fps for c in summary['per_camera']):
        errors.append('one or more cameras missed the FPS target')
    latency = summary.get('p95_processing_ms')
    if latency is None or not 0 <= latency <= processing_ms: errors.append('processing P95 limit missed')
    if frame_age_ms is not None:
        age = native.get('p95_ms')
        if age is None or not 0 <= age <= frame_age_ms: errors.append('GPU pipeline frame-age P95 limit missed')
    if seconds is not None and summary.get('seconds', 0) < seconds - .05: errors.append('run did not complete the requested duration')
    return errors

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--min-fps', type=float, default=9.8)
    p.add_argument('--max-processing-ms', type=float, default=300)
    p.add_argument('--max-frame-age-ms', type=float)
    p.add_argument('--seconds', type=float)
    p.add_argument('--metadata', action='store_true')
    a = p.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    errors = failures(summary, a.min_fps, a.max_processing_ms, a.max_frame_age_ms, a.seconds)
    if a.metadata:
        counts = collections.Counter()
        with (a.run / 'metadata.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line); validate_frame(event); counts[event['camera_id']] += 1
        if dict(counts) != {c['camera_id']: c['frames'] for c in summary['per_camera']}:
            errors.append('metadata counts differ from summary')
    print(json.dumps({'pass': not errors, 'errors': errors, 'minimum_fps': summary.get('minimum_fps'),
                      'p95_processing_ms': summary.get('p95_processing_ms')}, indent=2))
    return 2 if errors else 0

if __name__ == '__main__': raise SystemExit(main())
