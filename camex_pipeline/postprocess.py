import numpy as np

def letterbox_geometry(width, height, size=640):
    ratio = min(size / width, size / height)
    w, h = int(round(width * ratio)), int(round(height * ratio))
    return w, h, (size - w) // 2, (size - h) // 2

def decode_person_output(output, width, height, confidence=.25, iou_threshold=.45, topk=300):
    rows = np.asarray(output, dtype=np.float32).reshape(-1, 6)
    good = np.isfinite(rows).all(axis=1) & (rows[:,4] >= confidence) & (rows[:,4] >= rows[:,5])
    rows = rows[good]
    if not len(rows):
        return np.empty((0,4), np.float32), np.empty(0, np.float32)
    boxes, scores = rows[:,:4].copy(), rows[:,4].copy()
    boxes = np.clip(boxes, 0, 640)
    keep = (boxes[:,2]-boxes[:,0] >= 1) & (boxes[:,3]-boxes[:,1] >= 1)
    boxes, scores = boxes[keep], scores[keep]
    order = np.argsort(-scores, kind='stable')
    picked = []
    while len(order) and len(picked) < topk:
        i = order[0]
        picked.append(i)
        other = order[1:]
        a = boxes[i]
        inter = np.maximum(0, np.minimum(a[2:], boxes[other,2:]) - np.maximum(a[:2], boxes[other,:2]))
        intersection = inter[:,0] * inter[:,1]
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (boxes[other,2]-boxes[other,0]) * (boxes[other,3]-boxes[other,1])
        iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
        order = other[iou <= iou_threshold]
    boxes, scores = boxes[picked], scores[picked]
    w, h, left, top = letterbox_geometry(width, height)
    boxes[:,[0,2]] = (boxes[:,[0,2]] - left) * width / w
    boxes[:,[1,3]] = (boxes[:,[1,3]] - top) * height / h
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, width)
    boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, height)
    valid = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
    return boxes[valid], scores[valid]
