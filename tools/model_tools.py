"""Shared calibration/evaluation frame preparation. No identity data is used."""
import sys
import pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import av
import numpy as np
from camex_pipeline.postprocess import letterbox_geometry

def tensor_at(source, seconds, size=640):
    with av.open(source['uri']) as container:
        stream=container.streams.video[0]
        stream.codec_context.thread_count=1
        container.seek(int(seconds/float(stream.time_base)),stream=stream)
        for frame in container.decode(stream):
            if float(frame.pts*frame.time_base)+.001<seconds:
                continue
            w,h,x,y=letterbox_geometry(frame.width,frame.height,size)
            resized=frame.reformat(width=w,height=h,format='rgb24').to_ndarray()
            image=np.zeros((size,size,3),np.uint8);image[y:y+h,x:x+w]=resized
            return np.ascontiguousarray(image.transpose(2,0,1)[None],dtype=np.float32)/255.
    raise ValueError('No frame at requested timestamp')

def match_boxes(reference,candidate,threshold=.5):
    matched=0; used=set()
    for a in reference:
        best,best_i=0,None
        for i,b in enumerate(candidate):
            if i in used:continue
            inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
            inter=float(inter[0]*inter[1]);area=lambda c:float((c[2]-c[0])*(c[3]-c[1]))
            overlap=inter/max(1e-9,area(a)+area(b)-inter)
            if overlap>best:best,best_i=overlap,i
        if best_i is not None and best>=threshold:
            matched+=1;used.add(best_i)
    return matched
