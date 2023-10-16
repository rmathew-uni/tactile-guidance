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
        conf_thres=0.25,  # confidence threshold
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes_obj=67,  # filter by class: --class 0, or --class 0 2 3 / check coco.yaml file - person class is 0
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
    #model_obj = DetectMultiBackend(weights_obj, device=device, dnn=dnn, data=data_obj, fp16=half)
    #model_hand = DetectMultiBackend(weights_hand, device=device, dnn=dnn, data=data_hand, fp16=half)
    model_obj = DetectMultiBackend(weights_obj, device=device, dnn=dnn, fp16=half)
    model_hand = DetectMultiBackend(weights_hand, device=device, dnn=dnn, fp16=half)
    stride_obj, names_obj, pt_obj = model_obj.stride, model_obj.names, model_obj.pt
    stride_hand, names_hand, pt_hand = model_hand.stride, model_hand.names, model_hand.pt
    imgsz = check_img_size(imgsz, s=stride_obj)  # check image size
    #input(data_hand)

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride_obj, auto=True, vid_stride=vid_stride)
        bs = len(dataset)

    # Run inference
    model_obj.warmup(imgsz=(1 if pt_obj or model_obj.triton else bs, 3, *imgsz))  # warmup
    model_hand.warmup(imgsz=(1 if pt_hand or model_hand.triton else bs, 3, *imgsz))  # warmup

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # Milad s
    bbox_info = []  # Initialize a list to store bounding boxs

    horizontal_in, vertical_in = False, False
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


        pred =  pred_hand + pred_obj
        #pred = pred_hand

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

        # Process predictions
        for i, det in enumerate(pred):  # per image
            if i==0:
                curr_labels = names_hand
            else:
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
                    bbox_info.append({
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
        #for annotator in annotators_list:
        #    im0 += annotator.result()
        im0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond


            #Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)
        # Milad s
        # Print bounding box information
        for idx, bbox_dict in enumerate(bbox_info):
            print(f"Label: {bbox_dict['label']}, Bbox: {bbox_dict['bbox']}")
        # Milad e
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        horizontal_out, vertical_out = navigate_hand(bbox_info,classes_obj,classes_hand,horizontal_in,vertical_in)

        #horizontal_in, vertical_in = False, False

        if horizontal_out:
           horizontal_in = True
        if vertical_out:
           vertical_in = True

        bbox_info = []


def main(weights_obj, weights_hand, source, target_obj):
    #check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(weights_obj=weights_obj, weights_hand= weights_hand, source=source, classes_obj=target_obj)


# if __name__ == '__main__':
#     weights = 'yolov5s.pt'  # Model weights path
#     source = '0'  # Input image path
#     # Add other parameters as needed
#
#     main(weights, source)
