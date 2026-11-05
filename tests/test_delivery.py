import importlib.util
import json
from pathlib import Path
import tarfile
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_archive_rejects_paths_and_links():
    helper=module(ROOT/'tools/fetch_demo.py')
    assert helper.safe_member(tarfile.TarInfo('video/C01.mp4'))
    assert not helper.safe_member(tarfile.TarInfo('../outside'))
    assert not helper.safe_member(tarfile.TarInfo('/absolute'))
    link=tarfile.TarInfo('link');link.type=tarfile.SYMTYPE
    assert not helper.safe_member(link)

def test_result_gate_rejects_slow_source_even_with_normal_exit():
    helper=module(ROOT/'tools/check_result.py')
    summary={'per_camera':[{'fps':10},{'fps':7}], 'p95_processing_ms':30,'seconds':30,'native':{}}
    assert helper.failures(summary,9.8,300,seconds=30)

def test_result_gate_checks_gpu_age_and_duration():
    helper=module(ROOT/'tools/check_result.py')
    summary={'per_camera':[{'fps':10}], 'p95_processing_ms':30,'seconds':30,'native':{'p95_ms':400}}
    assert helper.failures(summary,9.8,300,300,600)

def test_result_gate_passes_clean_completed_case():
    helper=module(ROOT/'tools/check_result.py')
    summary={'per_camera':[{'fps':10}], 'p95_processing_ms':30,'seconds':600,'native':{'p95_ms':150}}
    assert not helper.failures(summary,9.8,300,300,600)

def test_adapter_uses_native_pixels_and_epoch():
    helper=module(ROOT/'examples/camex_adapter.py')
    event={'schema_version':'1.0','type':'frame','run_id':'r','camera_id':'C01','frame_id':1,'pts_ns':1,
           'source_pts_ns':1,'timestamp_domain':'pipeline','tracking_epoch':2,'width':1920,'height':1080,
           'objects':[{'bbox_xyxy':[200,100,320,480],'track_id':'99999999999','confidence':-1}]}
    result=helper.to_projection_requests(event)[0]
    assert result['footpoint_native_px']==[260,480]
    assert result['track_key']==['r','C01',2,'99999999999']
    assert result['confidence'] is None
