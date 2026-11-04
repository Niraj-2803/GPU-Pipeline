import json
import numpy as np
import pytest

from camex_pipeline.common import JsonlWriter, cuda_ordinal, load_sources, redact_uri, validate_frame
from camex_pipeline.postprocess import decode_person_output, letterbox_geometry

def config(tmp_path, rows=None):
    video=tmp_path/'test.mp4';video.touch()
    rows=rows or [{'camera_id':'front-door','uri':'test.mp4','width':1920,'height':1080,'fps':10}]
    p=tmp_path/'sources.json';p.write_text(json.dumps({'sources':rows}))
    return p

def test_source_expansion(tmp_path):
    sources=load_sources(config(tmp_path),3)
    assert [s['camera_id'] for s in sources]==['front-door','front-door_r2','front-door_r3']
    assert len({s['stream_index'] for s in sources})==3

@pytest.mark.parametrize('identifier',['bad/name','bad\nname','','x'*65])
def test_invalid_camera(tmp_path,identifier):
    with pytest.raises(ValueError):
        load_sources(config(tmp_path,[{'camera_id':identifier,'uri':'test.mp4','width':1920,'height':1080,'fps':10}]))

@pytest.mark.parametrize('count',[0,-1,401])
def test_bad_count(tmp_path,count):
    with pytest.raises(ValueError):load_sources(config(tmp_path),count)

def test_missing_video(tmp_path):
    with pytest.raises(ValueError):
        load_sources(config(tmp_path,[{'camera_id':'C01','uri':'missing.mp4','width':1920,'height':1080,'fps':10}]))

def test_rtsp_redaction():
    assert redact_uri('rtsp://alice:secret@127.0.0.1:8554/camera?token=secret')=='rtsp://127.0.0.1:8554/camera'

@pytest.mark.parametrize('width,height,expected',[(1920,1080,(640,360,0,140)),(2560,1440,(640,360,0,140)),(1080,1920,(360,640,140,0))])
def test_letterbox(width,height,expected):
    assert letterbox_geometry(width,height)==expected

def test_person_filter_nms_and_scaling():
    rows=np.array([[100,160,200,300,.9,.1],[102,162,202,302,.8,.1],[300,200,350,250,.7,.8],[0,0,640,640,.01,.0]],dtype=np.float32)
    boxes,scores=decode_person_output(rows,1920,1080)
    assert len(boxes)==1
    np.testing.assert_allclose(boxes[0],[300,60,600,480])
    np.testing.assert_allclose(scores,[.9])

def test_outside_and_nan():
    rows=np.array([[-20,-30,700,700,.8,.1],[np.nan,10,100,100,.9,.1]])
    boxes,_=decode_person_output(rows,1920,1080)
    np.testing.assert_allclose(boxes,[[0,0,1920,1080]])

def test_empty():
    boxes,scores=decode_person_output(np.empty((0,6)),1920,1080)
    assert boxes.shape==(0,4) and scores.shape==(0,)

def frame():
    return {'schema_version':'1.0','type':'frame','run_id':'test','camera_id':'C01','frame_id':1,'pts_ns':1,'width':1920,'height':1080,
            'objects':[{'bbox_xyxy':[0,0,100,100],'track_id':'5000000000'}]}

def test_valid_metadata():validate_frame(frame())

def test_ids_are_not_floats():
    event=frame();event['objects'][0]['track_id']=1.1
    with pytest.raises(ValueError):validate_frame(event)

def test_native_coordinates_are_bounded():
    event=frame();event['objects'][0]['bbox_xyxy'][2]=2000
    with pytest.raises(ValueError):validate_frame(event)

def test_writer_drains(tmp_path):
    p=tmp_path/'events.jsonl';w=JsonlWriter(p)
    for i in range(10):w.push({'id':i})
    w.close()
    assert len(p.read_text().splitlines())==10 and w.dropped==0

def test_cuda_mapping_uses_pci_order(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('camex_pipeline.common.subprocess.run',lambda *args,**kwargs:SimpleNamespace(stdout='0, 00000000:0B:00.0\n1, 00000000:0A:00.0\n'))
    assert cuda_ordinal(0)==1
    assert cuda_ordinal(1)==0
