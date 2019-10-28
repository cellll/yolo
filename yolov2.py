from darkflow.net.build import TFNet
import cv2
import time
import os
import subprocess

#o = subprocess.getoutput('/home/test/yolo2/gpu.sh')
#available_gpu = o.split('output_gpu::')[1].split(':')[1]
#os.environ['CUDA_VISIBLE_DEVICES']=available_gpu
#os.environ['CUDA_VISIBLE_DEVICES']='0'


class YOLOV2_XIILAB:
    
    def __init__(self):
        options = {"model": "/root/object_detection/yolo2/cfg/yolov2-tiny.cfg", 
                   "load": "/root/object_detection/yolo2/weights/yolov2-tiny.weights", 
                   "threshold": float(os.environ['YOLO_THRESHOLD']),
                   "gpu" : float(os.environ['GPU_MEMORY_FRACTION'])}
        self.tfnet = TFNet(options)

    def append_to_list(self, nparr):
 
        self.tfnet.append_to_list(nparr)

    def inference(self):

        result = self.tfnet.detection()
        
        return result
