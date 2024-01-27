# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import keyboard

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from bracelet import navigate_hand

obj_name_dict = {
0: "person",
1: "bicycle",
2: "car",
3: "motorcycle",
4: "airplane",
5: "bus",
6: "train",
7: "truck",
8: "boat",
9: "traffic light",
10: "fire hydrant",
11: "stop sign",
12: "parking meter",
13: "bench",
14: "bird",
15: "cat",
16: "dog",
17: "horse",
18: "sheep",
19: "cow",
20: "elephant",
21: "bear",
22: "zebra",
23: "giraffe",
24: "backpack",
25: "umbrella",
26: "handbag",
27: "tie",
28: "suitcase",
29: "frisbee",
30: "skis",
31: "snowboard",
32: "sports ball",
33: "kite",
34: "baseball bat",
35: "baseball glove",
36: "skateboard",
37: "surfboard",
38: "tennis racket",
39: "bottle",
40: "wine glass",
41: "cup",
42: "fork",
43: "knife",
44: "spoon",
45: "bowl",
46: "banana",
47: "apple",
48: "sandwich",
49: "orange",
50: "broccoli",
51: "carrot",
52: "hot dog",
53: "pizza",
54: "donut",
55: "cake",
56: "chair",
57: "couch",
58: "potted plant",
59: "bed",
60: "dining table",
61: "toilet",
62: "tv",
63: "laptop",
64: "mouse",
65: "remote",
66: "keyboard",
67: "cell phone",
68: "microwave",
69: "oven",
70: "toaster",
71: "sink",
72: "refrigerator",
73: "book",
74: "clock",
75: "vase",
76: "scissors",
77: "teddy bear",
78: "hair drier",
79: "toothbrush"
}


@smart_inference_mode()
def run(
        weights_obj=ROOT / 'yolov5s.pt',  # model_obj path or triton URL
        weights_hand=ROOT / 'yolov5s.pt',  # model_obj path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webca
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txtm)
        #data_obj=ROOT / 'coco.yaml',  # dataset.yaml path
        #data_hand=ROOT / 'data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.7,  # confidence threshold
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes_obj=[1,39,40,41,45,46,47,58,74],  # filter by class: --class 0, or --class 0 2 3 / check coco.yaml file - person class is 0
        classes_hand=[0,1],
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride_obj
    
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model_obj
    device = select_device(device)
    model_obj = DetectMultiBackend(weights_obj, device=device, dnn=dnn, fp16=half)
    model_hand = DetectMultiBackend(weights_hand, device=device, dnn=dnn, fp16=half)
    stride_obj, names_obj, pt_obj = model_obj.stride, model_obj.names, model_obj.pt
    stride_hand, names_hand, pt_hand = model_hand.stride, model_hand.names, model_hand.pt
    imgsz = check_img_size(imgsz, s=stride_obj)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride_obj, auto=True, vid_stride=vid_stride)
        bs = len(dataset)

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model_obj.warmup(imgsz=(1 if pt_obj or model_obj.triton else bs, 3, *imgsz))  # warmup
    model_hand.warmup(imgsz=(1 if pt_hand or model_hand.triton else bs, 3, *imgsz))  # warmup

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # Milad s
    bboxs_hands = []  # Initialize a list to store bounding boxes for hands
    bboxs_objs = [] # Initialize a list to store bounding boxes for objects

    horizontal_in, vertical_in = False, False
    target_entered = False
    #target_obj = 0
    check = 1
    check_dur = 0

    # Milad e
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im_obj = torch.from_numpy(im).to(model_obj.device)
            im_obj = im_obj.half() if model_obj.fp16 else im_obj.float()  # uint8 to fp16/32
            im_obj /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im_obj.shape) == 3:
                im_obj = im_obj[None]  # expand for batch dim

            im_hand = torch.from_numpy(im).to(model_hand.device)
            im_hand = im_hand.half() if model_hand.fp16 else im_hand.float()  # uint8 to fp16/32
            im_hand /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im_hand.shape) == 3:
                im_hand = im_hand[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred_obj = model_obj(im_obj, augment=augment, visualize=visualize)
            pred_hand = model_hand(im_hand, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred_obj = non_max_suppression(pred_obj, conf_thres, iou_thres, classes_obj, agnostic_nms, max_det=max_det)
            pred_hand = non_max_suppression(pred_hand, conf_thres, iou_thres, classes_hand, agnostic_nms, max_det=max_det)

        annotators_list = []

        if webcam:  # batch_size >= 1
            p, im0, frame = path[0], im0s[0].copy(), dataset.count
            s += f'{0}: '

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names_obj))

        # Process hand predictions
        for i, det in enumerate(pred_hand):  # per image
            curr_labels = names_hand
            i = 0
            seen += 1
            '''if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_obj))'''
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {curr_labels[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Milad s
                    # Collect bounding box information
                    bbox = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()

                    bboxs_hands.append({
                        "class": int(cls),
                        "label": curr_labels[int(cls)],
                        "confidence": conf,
                        "bbox": bbox
                    })
                    # Milad e
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (curr_labels[c] if hide_conf else f'{curr_labels[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names_obj[c] / f'{p.stem}.jpg', BGR=True)
        
        # Process object predictions
        for i, det in enumerate(pred_obj):  # per image
            curr_labels = names_obj
            i = 0
            seen += 1
            '''if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_obj))'''
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {curr_labels[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Milad s
                    # Collect bounding box information
                    bbox = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()

                    bboxs_objs.append({
                        "class": int(cls),
                        "label": curr_labels[int(cls)],
                        "confidence": conf,
                        "bbox": bbox
                    })
                    # Milad e
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(f'{txt_path}.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (curr_labels[c] if hide_conf else f'{curr_labels[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names_obj[c] / f'{p.stem}.jpg', BGR=True)

        # Stream results
        im0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond


        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Hand navigation loop
        # After passing target object class hand is navigated in each frame until grasping command is sent
        if target_entered == False:
            user_in = "n"
            while user_in == "n":
                print("These are the available objects:")
                print(obj_name_dict)
                target_obj_verb = input('Enter the object you want to target: ')

                if target_obj_verb in obj_name_dict.values():
                    user_in = input("Selected object is " + target_obj_verb + ". Correct? [y,n]")
                else:
                    print(f'The object {target_obj_verb} is not in the list of available targets. Please reselect.')

            target_entered = True
            grasp = False
            horizontal_in, horizontal_out = False, False
            vertical_in, vertical_out = False, False
        elif target_entered:
            pass

        # Navigate the hand based on information from last frame and current frame detections
        horizontal_out, vertical_out, grasp, check, check_dur = navigate_hand(bboxs_hands,bboxs_objs,target_obj_verb, classes_hand, horizontal_in, vertical_in, grasp,check, check_dur)

        # Exit the loop if hand and object aligned horizontally and vertically and grasp signal was sent
        if horizontal_out and vertical_out and grasp:
            target_entered = False

        #horizontal_in, vertical_in = False, False

        # Set values of the inputs for the next loop iteration
        if horizontal_out:
           horizontal_in = True
        if vertical_out:
           vertical_in = True

        # Clear bbox_info after applying navigation logic for the current frame
        bboxs_hands = []
        bboxs_objs = []


# def main(weights_obj, weights_hand, source):
#     '''
#     Function that navigates hand toward target object based on input from object and hand detectors.
#     Input:
#     • weights_obj - weights for the object detector model
#     • weights_hand - weights for the hand detector model
#     • source - source of the visual data for which navigation will be applied; for default system camera type 0 or "0"
#     '''
#     #check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#     run(weights_obj=weights_obj, weights_hand= weights_hand, source=source)


if __name__ == '__main__':
    weights_obj = 'aibox/yolov5s.pt'  # Object model weights path
    weights_hand = 'aibox/hand.pt'# Hands model weights path
    source = '0'  # Input image path
    # Add other parameters as needed

    run(weights_obj=weights_obj, weights_hand=weights_hand, source=source)
