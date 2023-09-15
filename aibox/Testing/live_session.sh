#!/bin/bash

python HPC/optivist/ultralytics/detect.py --weights HPC/optivist/ultralytics/runs/train/exp8/weights/best.pt --source 0 --data 
../OptiVisT/aibox/datasets/cocohands/cocohands.yaml --view-img
