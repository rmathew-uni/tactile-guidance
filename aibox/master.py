"""
This script is using code from the following sources:
- YOLOv5 🚀 by Ultralytics, AGPL-3.0 license, https://github.com/ultralytics/yolov5
- StrongSORT MOT, https://github.com/dyhBUPT/StrongSORT, https://pypi.org/project/strongsort/
- Youtube Tutorial "Simple YOLOv8 Object Detection & Tracking with StrongSORT & ByteTrack" by Nicolai Nielsen, https://www.youtube.com/watch?v=oDALtKbprHg
- https://github.com/zenjieli/Yolov5StrongSORT/blob/master/track.py, original: https://github.com/mikel-brostrom/yolo_tracking/commit/9fec03ddba453959f03ab59bffc36669ae2e932a
"""

# region Setup

# System
import os
import requests
import platform
import sys
from pathlib import Path
import itertools
import time
import numpy as np

# Use the project file packages instead of the conda packages, i.e. add to system path for import
file = Path(__file__).resolve()
root = file.parents[0]
sys.path.append(str(root) + '/yolov5')
sys.path.append(str(root) + '/strongsort')
sys.path.append(str(root) + '/MiDaS')

# Object tracking
import torch
from labels import coco_labels # COCO labels dictionary
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from strongsort.strong_sort import StrongSORT # there is also a pip install, but it has multiple errors

# DE
from MiDaS.midas.model_loader import default_models, load_model
from MiDaS.run import create_side_by_side, process

# Navigation
from bracelet import navigate_hand, connect_belt
import controller

# Utility
import keyboard
from playsound import playsound
import threading

# endregion

# region Helpers

def playstart():
    file = 'resources/sound/beginning.mp3' # ROOT
    playsound(str(file))


def play_start():
    play_start_thread = threading.Thread(target=playstart, name='play_start')
    play_start_thread.start()

# endregion

if __name__ == '__main__':

    #check_requirements(requirements='../requirements.txt', exclude=('tensorboard', 'thop'))
    
    weights_obj = 'yolov5s.pt'  # Object model weights path
    weights_hand = 'hand.pt' # Hands model weights path
    weights_tracker = 'osnet_x0_25_market1501.pt' # ReID weights path
    depth_estimator = 'midas_v21_small_256' # depth estimator model type (weights are loaded automatically!), 
                                      # e.g.'midas_v21_small_256', ('dpt_levit_224', 'dpt_swin2_tiny_256',) 'dpt_large_384'
    source = '1' # image/video path or camera source (0 = webcam, 1 = external, ...)
    mock_navigate = False # Navigate without the bracelet using only print commands
    belt_controller = None
    run_object_tracker = True
    run_depth_estimator = True

    print(f'\nLOADING CAMERA AND BRACELET')

    # Check camera connection
    try:
        source = str(source)
        print('Camera connection successful')
    except:
        print('Cannot access selected source. Aborting.')
        sys.exit()

    # Check bracelet connection
    if not mock_navigate:
        connection_check, belt_controller = controller.connect_belt()
        if connection_check:
            print('Bracelet connection successful.')
        else:
            print('Error connecting bracelet. Aborting.')
            sys.exit()

    try:
        bracelet_controller = controller.BraceletController(weights_obj=weights_obj,  # model_obj path or triton URL # ROOT
                        weights_hand=weights_hand,  # model_obj path or triton URL # ROOT
                        weights_tracker=weights_tracker, # ROOT
                        depth_estimator=depth_estimator,
                        source=source,  # file/dir/URL/glob/screen/0(webcam) # ROOT
                        iou_thres=0.45,  # NMS IOU threshold
                        max_det=1000,  # maximum detections per image
                        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                        view_img=True,  # show results
                        save_txt=False,  # save results to *.txtm)
                        imgsz=(640, 640),  # inference size (height, width)
                        conf_thres=0.7,  # confidence threshold
                        save_conf=False,  # save confidences in --save-txt labels
                        save_crop=False,  # save cropped prediction boxes
                        nosave=True,  # do not save images/videos
                        classes_obj=[1,39,40,41,45,46,47,58,74],  # filter by class /  check coco.yaml file or coco_labels variable in this script
                        classes_hand=[0,1], 
                        class_hand_nav=[80,81],
                        agnostic_nms=False,  # class-agnostic NMS
                        augment=False,  # augmented inference
                        visualize=False,  # visualize features
                        update=False,  # update all models
                        project='runs/detect',  # save results to project/name # ROOT
                        name='video',  # save results to project/name
                        exist_ok=False,  # existing project/name ok, do not increment
                        line_thickness=3,  # bounding box thickness (pixels)
                        hide_labels=False,  # hide labels
                        hide_conf=False,  # hide confidences
                        half=False,  # use FP16 half-precision inference
                        dnn=False,  # use OpenCV DNN for ONNX inference
                        vid_stride=1,  # video frame-rate stride_obj
                        manual_entry=False, # True means you will control the exp manually versus the standard automatic running
                        run_object_tracker=run_object_tracker,
                        run_depth_estimator=run_depth_estimator,
                        mock_navigate=mock_navigate,
                        belt_controller=belt_controller,
                        target_objs = ['bottle' for _ in range(5)]) # debugging
        
        bracelet_controller.run()

    except KeyboardInterrupt:
        controller.close_app(belt_controller)
    
    # In the end, kill everything
    controller.close_app(belt_controller)