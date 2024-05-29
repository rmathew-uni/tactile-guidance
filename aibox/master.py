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

# Navigation
from bracelet import navigate_hand, connect_belt

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


def close_app(controller):
    print("Application will be closed")
    cv2.destroyAllWindows()
    # As far as I understood, all threads are properly closed by releasing their locks before being stopped
    threads = threading.enumerate()
    for thread in threads:
        thread._tstate_lock = None
        thread._stop()
    controller.disconnect_belt() if controller else None
    sys.exit()


def xyxy_to_xywh(bb):
    x1, y1, x2, y2 = bb
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    x = x1 + w//2
    y = y1 + h//2
    return [x, y, w, h]

# endregion

@smart_inference_mode()
def run(
        weights_obj='yolov5s.pt',  # model_obj path or triton URL # ROOT
        weights_hand='hand.pt',  # model_obj path or triton URL # ROOT
        weights_tracker='osnet_x0_25_market1501.pt', # ROOT
        source='data/images',  # file/dir/URL/glob/screen/0(webcam) # ROOT
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txtm)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.7,  # confidence threshold
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
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
):

    # region main setup

    # Experiment setup
    if not manual_entry:
        target_objs = ['apple','banana','potted plant','bicycle','cup','clock','wine glass']
        target_objs = ['bottle' for _ in range(5)] # debugging
        obj_index = 0
        print(f'The experiment will be run automatically. The selected target objects, in sequence, are:\n{target_objs}')
    else:
        print('The experiment will be run manually. You will enter the desired target for each run yourself.')

    horizontal_in, vertical_in = False, False
    target_entered = False
    play_start()  # play welcome sound

    # Configure saving
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    if is_url and is_file:
        source = check_file(source)  # download

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    if save_img:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load object detection models
    print(f'\nLOADING OBJECT DETECTORS')
    device = select_device(device)
    model_obj = DetectMultiBackend(weights_obj, device=device, dnn=dnn, fp16=half)
    model_hand = DetectMultiBackend(weights_hand, device=device, dnn=dnn, fp16=half)

    stride_obj, names_obj, pt_obj = model_obj.stride, model_obj.names, model_obj.pt
    stride_hand, names_hand, pt_hand = model_hand.stride, model_hand.names, model_hand.pt
    imgsz = check_img_size(imgsz, s=stride_obj) # check image size
    dt = (Profile(), Profile(), Profile())

    # Load data stream
    bs = 1  # batch_size
    view_img = check_imshow(warn=True)
    try:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride_obj, auto=True, vid_stride=vid_stride)
    except AssertionError:
        while True:
            change_camera = input(f'Failed to open camera with index {source}. Do you want to continue with webcam? (Y/N)')
            if change_camera == 'Y':
                source = '0'
                dataset = LoadStreams(source, img_size=imgsz, stride=stride_obj, auto=True, vid_stride=vid_stride)
                break
            elif change_camera == 'N':
                exit()
    bs = len(dataset)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Create combined label dictionary
    index_add = len(names_obj)
    labels_hand_adj = {key + index_add: value for key, value in names_hand.items()}
    master_label = names_obj | labels_hand_adj

    # Load tracker model
    print(f'LOADING OBJECT TRACKER')
    tracker = StrongSORT(
            model_weights=weights_tracker, 
            device=device,
            fp16=False,
            max_dist=0.5,          # The matching threshold. Samples with larger distance are considered an invalid match
            max_iou_distance=0.7,  # Gating threshold. Associations with cost larger than this value are disregarded.
            max_age=70,            # Maximum number of missed misses (prediction calls, i.e. frames I think) before a track is deleted
            n_init=1,              # Number of frames that a track remains in initialization phase --> if 0, track is confirmed on first detection
            nn_budget=100,         # Maximum size of the appearance descriptors gallery
            mc_lambda=0.995,       # matching with both appearance (1 - MC_LAMBDA) and motion cost
            ema_alpha=0.9          # updates  appearance  state in  an exponential moving average manner
            )

    # Warmup models
    model_obj.warmup(imgsz=(1 if pt_obj or model_obj.triton else bs, 3, *imgsz))
    model_hand.warmup(imgsz=(1 if pt_hand or model_hand.triton else bs, 3, *imgsz))
    tracker.model.warmup()

    # endregion


    # region main tracking

    print(f'\nSTARTING MAIN LOOP')

    # Initialize vars for tracking
    prev_frames = None
    curr_frames = None
    fpss = []
    outputs = []

    # Data processing: Iterate over each frame of the live stream
    for frame, (path, im, im0s, vid_cap, _) in enumerate(dataset):

        # Start timer for FPS measure
        start = time.perf_counter()

        # Setup saving and visualization
        p, im0 = Path(path[0]), im0s[0].copy() # idx 0 is for first source (and we only have one source)
        save_path = str(save_dir / p.name)  # im.jpg
        annotator = Annotator(im0, line_width=line_thickness, example=str(names_obj))

        # Image pre-processing
        with dt[0]:
            image = torch.from_numpy(im).to(model_obj.device)
            image = image.half() if model_obj.fp16 else image.float()  # uint8 to fp16/32
            image /= 255  # 0 - 255 to 0.0 - 1.0
            if len(image.shape) == 3:
                image = image[None]  # expand for batch dim

        # Object detection inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred_target = model_obj(image, augment=augment, visualize=visualize)
            pred_hand = model_hand(image, augment=augment, visualize=visualize)

        # Non-maximal supression
        with dt[2]:
            pred_target = non_max_suppression(pred_target, conf_thres, iou_thres, classes_obj, agnostic_nms, max_det=max_det) # list containing one tensor (n,6)
            pred_hand = non_max_suppression(pred_hand, conf_thres, iou_thres, classes_hand, agnostic_nms, max_det=max_det) # list containing one tensor (n,6)

        for hand in pred_hand[0]:
            if len(hand):
                hand[5] += index_add # assign correct classID by adding len(coco_labels)

        # Camera motion compensation for tracker (ECC)
        curr_frames = im0
        tracker.tracker.camera_update(prev_frames, curr_frames)
        
        # Initialize/clear detections
        xywhs = torch.empty(0,4)
        confs = torch.empty(0)
        clss = torch.empty(0)

        # Process object detections
        preds = torch.cat((pred_target[0], pred_hand[0]), dim=0)
        if len(preds) > 0:
            preds[:, :4] = scale_boxes(im.shape[2:], preds[:, :4], im0.shape).round()
            xywhs = xyxy2xywh(preds[:, :4])
            confs = preds[:, 4]
            clss = preds[:, 5]

        # Generate tracker outputs for navigation
        outputs = tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0) # returns xyxys again

        # Get FPS
        end = time.perf_counter()
        runtime = end - start
        fps = 1 / runtime
        fpss.append(fps)
        #print(f'Frame: {frame}, Average FPS: {int(np.mean(fpss))}, Device: {device}')
        prev_frames = curr_frames

        # region visualization
        # Write results
        for *xyxy, obj_id, cls, conf in outputs:
            id, cls = int(obj_id), int(cls)
            if save_img or save_crop or view_img:
                label = None if hide_labels else (f'ID: {id} {master_label[cls]}' if hide_conf else (f'ID: {id} {master_label[cls]} {conf:.2f}'))
                annotator.box_label(xyxy, label, color=colors(cls, True))

        # Display results
        im0 = annotator.result()
        if view_img:
            #cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # for resizing
            cv2.putText(im0, f'FPS: {int(fps)}, Avg: {int(np.mean(fpss))}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 1)
            cv2.imshow(str(p), im0)
            #cv2.resizeWindow(str(p), im0.shape[1]//2, im0.shape[0]//2) # for resizing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save results
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[0] != save_path:  # new video
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0)
        
        # endregion

        # Convert BBs after display
        if len(outputs) > 0:
            for det in range(len(outputs)):
                outputs[det, :4] = xyxy_to_xywh(outputs[det, :4])

        # endregion


        # region main navigation
        # Get the target object class
        if not target_entered:
            if manual_entry:
                user_in = "n"
                while user_in == "n":
                    print("These are the available objects:")
                    print(coco_labels)
                    target_obj_verb = input('Enter the object you want to target: ')

                    if target_obj_verb in coco_labels.values():
                        user_in = input("Selected object is " + target_obj_verb + ". Correct? [y,n]")
                        class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                        file = f'resources/sound/{target_obj_verb}.mp3'
                        playsound(str(file))
                        # Start trial time measure (end in navigate_hand(...))
                    else:
                        print(f'The object {target_obj_verb} is not in the list of available targets. Please reselect.')
            else:
                target_obj_verb = target_objs[obj_index]
                class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                file = f'resources/sound/{target_obj_verb}.mp3'
                playsound(str(file))
                # Start trial time measure (end in navigate_hand(...))

            target_entered = True

        # Navigate the hand based on information from last frame and current frame detections
        if frame == 0: # Initialize navigation
            horizonal, vertical, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
            navigate_hand(belt_controller, outputs, class_target_obj, class_hand_nav)

        horizonal, vertical, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
            navigate_hand(belt_controller,
                          outputs,
                          class_target_obj,
                          class_hand_nav,
                          horizonal,
                          vertical,
                          grasp,
                          obj_seen_prev,
                          search,
                          count_searching,
                          count_see_object,
                          jitter_guard,
                          navigating)
    
        # Exit the loop if hand and object aligned horizontally and vertically and grasp signal was sent
        if grasp:
            if manual_entry and ((obj_index+1)<=len(target_objs)):
                obj_index += 1
            target_entered = False
        
        # endregion
        

if __name__ == '__main__':

    #check_requirements(requirements='../requirements.txt', exclude=('tensorboard', 'thop'))
    weights_obj = 'yolov5s.pt'  # Object model weights path
    weights_hand = 'hand.pt' # Hands model weights path
    weights_tracker = 'osnet_x0_25_market1501.pt' # ReID weights path
    source = '0' # image/video path or camera source (0 = webcam, 1 = external, ...)
    mock_navigate = True # Navigate without the bracelet using only print commands
    belt_controller = None

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
        connection_check, belt_controller = connect_belt()
        if connection_check:
            print('Bracelet connection successful.')
        else:
            print('Error connecting bracelet. Aborting.')
            sys.exit()

    try:
        run(weights_obj=weights_obj, weights_hand=weights_hand, weights_tracker=weights_tracker, source=source, nosave=True)
        close_app(belt_controller)
    except KeyboardInterrupt:
        close_app(belt_controller)
    
    # In the end, kill everything
    close_app(belt_controller)
