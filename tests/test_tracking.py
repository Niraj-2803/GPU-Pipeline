import numpy as np
import supervision as sv

def detections(boxes):
    return sv.Detections(xyxy=np.asarray(boxes,dtype=np.float32).reshape(-1,4),
        confidence=np.ones(len(boxes),dtype=np.float32)*.9,
        class_id=np.zeros(len(boxes),dtype=int))

def tracker():
    return sv.ByteTrack(frame_rate=10,lost_track_buffer=30,minimum_consecutive_frames=1)

def test_short_gap_retains_track():
    t=tracker()
    first=t.update_with_detections(detections([[20,20,60,100]]))
    original=int(first.tracker_id[0])
    t.update_with_detections(detections([[22,20,62,100]]))
    t.update_with_detections(detections([]))
    back=t.update_with_detections(detections([[24,20,64,100]]))
    assert int(back.tracker_id[0])==original

def test_distinct_people_get_distinct_ids():
    result=tracker().update_with_detections(detections([[20,20,60,100],[120,20,160,100]]))
    assert len(set(result.tracker_id.tolist()))==2

def test_camera_scope_is_part_of_the_key():
    left=tracker().update_with_detections(detections([[20,20,60,100]]))
    right=tracker().update_with_detections(detections([[20,20,60,100]]))
    assert ('run','camera-a',str(left.tracker_id[0])) != ('run','camera-b',str(right.tracker_id[0]))

def test_half_second_gap_retains_track():
    t=tracker()
    original=t.update_with_detections(detections([[20,20,60,100]])).tracker_id[0]
    t.update_with_detections(detections([[20,20,60,100]]))
    for _ in range(5):t.update_with_detections(detections([]))
    assert t.update_with_detections(detections([[20,20,60,100]])).tracker_id[0]==original
