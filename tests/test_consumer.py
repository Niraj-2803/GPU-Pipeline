import json
from pathlib import Path
import subprocess
import sys

CONSUMER = Path(__file__).resolve().parents[1] / 'examples/consume_metadata.py'

def test_consumer_reads_complete_frame(tmp_path):
    file = tmp_path / 'metadata.jsonl'
    event = {'schema_version': '1.0', 'run_id': 'test', 'camera_id': 'C01',
             'frame_id': 1, 'tracking_epoch': 0, 'objects': [{'track_id': '7'}]}
    file.write_text(json.dumps(event) + '\n')
    result = subprocess.run([sys.executable, str(CONSUMER), str(file)],
                            text=True, capture_output=True, timeout=3)
    assert result.returncode == 0
    assert 'C01 1 1 detections 1 tracked' in result.stdout

def test_nonfollowing_consumer_rejects_truncated_line(tmp_path):
    file = tmp_path / 'metadata.jsonl'
    file.write_text('{"schema_version":')
    result = subprocess.run([sys.executable, str(CONSUMER), str(file)],
                            text=True, capture_output=True, timeout=3)
    assert result.returncode != 0
    assert 'Incomplete final metadata line' in result.stderr
