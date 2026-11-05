import threading
from types import SimpleNamespace
import numpy as np
from camex_pipeline.postprocess import decode_person_output,decode_ssd_output
from camex_pipeline.cpu import CpuPipeline

def test_small_detector_maps_back_to_native_pixels():
    rows=np.array([[50,80,100,150,.9,.9]],dtype=np.float32)
    boxes,_=decode_person_output(rows,1920,1080,size=320)
    np.testing.assert_allclose(boxes,[[300,60,600,480]])

def test_ssd_maps_normalized_boxes_and_filters_labels():
    output=np.array([[[[0,0,.9,.1,.2,.3,.4],[0,1,.8,0,0,1,1],[-1,0,0,0,0,0,0]]]])
    boxes,scores=decode_ssd_output(output,1920,1080,.5,0)
    np.testing.assert_allclose(boxes,[[192,216,576,432]])
    np.testing.assert_allclose(scores,[.9])

def test_batched_callback_preserves_source_routing():
    calls=[]
    pipeline=SimpleNamespace(states=[{'busy':True},{'busy':True}],lock=threading.Lock(),
        _finish_one=lambda output,i,p:calls.append((i,p,output.copy())),stop=threading.Event())
    outputs=np.array([[[1,2,3,4,.9,.9]],[[5,6,7,8,.8,.8]]])
    req=SimpleNamespace(get_output_tensor=lambda _:SimpleNamespace(data=outputs))
    CpuPipeline._inference_done(pipeline,req,([(1,'second'),(0,'first')],None))
    assert [(c[0],c[1]) for c in calls]==[(1,'second'),(0,'first')]
    np.testing.assert_array_equal(calls[0][2],outputs[0])
    assert not any(s['busy'] for s in pipeline.states)

def test_batch_callback_error_releases_all_sources():
    def fail(*_):raise ValueError('test failure')
    pipeline=SimpleNamespace(states=[{'busy':True},{'busy':True}],lock=threading.Lock(),
        _finish_one=fail,stop=threading.Event())
    req=SimpleNamespace(get_output_tensor=lambda _:SimpleNamespace(data=np.zeros((2,1,6))))
    CpuPipeline._inference_done(pipeline,req,([(0,None),(1,None)],None))
    assert pipeline.stop.is_set() and 'test failure' in pipeline.error
    assert not any(s['busy'] for s in pipeline.states)
