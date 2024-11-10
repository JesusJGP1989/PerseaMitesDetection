import torch
import utils
import os

from numba import jit



display = utils.notebook_init()  # checks

print('version %s' % (torch.__version__))


#@jit#(target="cuda")
#classify/train.py --model yolov5s-cls.pt --data $DATASET_NAME --epochs 20 --batch-size 64 --imgsz 224 --lr0 0.001 --pretrained weights/yolov5s-cls.pt
python classify/train.py --model yolov5s-cls.pt --data C:\Users\Carlos\OneDrive\Documentos\Mastr\Tesis\Repo\SW2\yolov5\Persea-mites-detection2-4 --epochs 10 --batch-size 64 --imgsz 224 --lr0 0.001 --pretrained weights/yolov5s-cls.pt


# input image
#python classify/predict.py --weights C:\Users\Carlos\Downloads\yolov5\runs\train-cls\exp18\weights\best.pt --source C:\Users\Carlos\Downloads\yolov5\Persea-mites-detection2-4\train\Persea-mite\14968281886_46d02e13d6_b_jpg.rf.a1c3f2e2ee0914cea2b62cbc5435f493.jpg