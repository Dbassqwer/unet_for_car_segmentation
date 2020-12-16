# 计算一次训练中保存的各个epoch参数对应的IoU

import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
from car_IoU import IoU
from predict_car import predict

def calculate():
    
    IoUs = {}
    model_path = glob.glob('saved_models/12150446/*.pth')
    model_path.sort()
    print(model_path)
    for path in model_path:
        with torch.cuda.device(1):
            predict(in_channel=3,model_path=path, data_path='../dataset/', light=True)
        IoUs[path] = IoU()
    return IoUs

if __name__ == "__main__":
    IoUs = calculate()
    print(IoUs)

# {'saved_models/12150446/best_model.pth': 0.9769231172929452, 
# 'saved_models/12150446/epoch_005.pth': 0.9760529328443692, 
# 'saved_models/12150446/epoch_010.pth': 0.9768443524287183, 
# 'saved_models/12150446/epoch_015.pth': 0.9769231172929452}



