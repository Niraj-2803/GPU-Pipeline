#!/usr/bin/env python3
"""Optional stock YOLOv8s export. Validate any new export before deployment."""
import argparse
import hashlib
import json
import pathlib
import os
import torch
from torch import nn

class PersonOutput(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,image):
        raw=self.model(image)
        if isinstance(raw,tuple):raw=raw[0]
        raw=raw.transpose(1,2)
        boxes=raw[:,:,:4]
        xyxy=torch.cat((boxes[:,:,:2]-boxes[:,:,2:4]/2,boxes[:,:,:2]+boxes[:,:,2:4]/2),dim=2)
        scores=raw[:,:,4:]
        return torch.cat((xyxy,scores[:,:,:1],scores.max(dim=2,keepdim=True).values),dim=2)

p=argparse.ArgumentParser()
p.add_argument('--weights',default='yolov8s.pt')
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
os.environ['YOLO_CONFIG_DIR']=str(a.output.resolve().parent/'.ultralytics-export-settings')
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect
model=YOLO(a.weights).model.cpu().float().eval()
model.fuse()
for module in model.modules():
    if isinstance(module,Detect):
        module.export=True;module.format='onnx';module.dynamic=False
    elif isinstance(module,C2f):module.forward=module.forward_split
a.output.parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(PersonOutput(model),torch.zeros(1,3,640,640),str(a.output),opset_version=17,
                  input_names=['input'],output_names=['output'],dynamic_axes={'input':{0:'batch'},'output':{0:'batch'}})
import onnx
onnx.checker.check_model(onnx.load(a.output))
a.output.with_suffix('.export.json').write_text(json.dumps({'model_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
    'source_weights':a.weights,'box_format':'xyxy','output_columns':['x1','y1','x2','y2','person_score','max_class_score']},indent=2))
print(a.output)
