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

# Configure relative path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # AIBox folder
sys.path.append(str(ROOT) + '/yolov5')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#os.chdir(ROOT) # change path to ROOT

# Object tracking
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from strongsort.strong_sort import StrongSORT # there is also a pip install, but it has multiple errors
from deep_sort_realtime.deepsort_tracker import DeepSort

# Navigation
from bracelet import navigate_hand, mock_navigate_hand, connect_belt

# Utility
import keyboard
from playsound import playsound
import threading

# COCO labels dictionary
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

# endregion

# region Helpers

def playstart():
    file = ROOT / f'resources/sound/beginning.mp3'
    #playsound(str(file))


def play_start():
    play_start_thread = threading.Thread(target=playstart, name='play_start')
    play_start_thread.start()


def hist_equalization(im):
    im_eq = np.squeeze(np.transpose(im, (3,2,1,0)))
    im = cv2.cvtColor(im_eq, cv2.COLOR_RGB2Lab)
    #configure CLAHE
    clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))
    #0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    im[:,:,0] = clahe.apply(im[:,:,0])
    im = cv2.cvtColor(im, cv2.COLOR_Lab2RGB)
    im = np.transpose(im, (2,0,1))
    im = np.expand_dims(im, axis=0)
    return im


def resize_image(im):
    # Resizing to Hx640 (training on 640x640 images)
    aspect_ratio = im.shape[1] / im.shape[0] # width / height
    im = cv2.resize(im, dsize=(640, int(640/aspect_ratio))) # width, height
    return im


def preprocess_detections(detections, im, im0, labels, count, idx_shift=None):
    # Rescale boxes from img_size to im0 size
    detections[:, :4] = scale_boxes(im.shape[2:], detections[:, :4], im0.shape).round()

    if idx_shift is not None:
        for k, (*xyxy, conf, cls) in enumerate(reversed(detections)):
            detections[-k-1][-1] = cls + idx_shift

    # Print results
    for c in detections[:, 5].unique():
        n = (detections[:, 5] == c).sum()  # detections per class
        count += f"{n} {labels[int(c)]}{'s' * (n > 1)}, "  # add to string
    
    return detections, count


def load_tracker(model_type, weights, device):
    if model_type == 'strongsort':
        model = StrongSORT(
            model_weights=weights, 
            device=device,
            fp16=False,
            max_dist=0.2,          # The matching threshold. Samples with larger distance are considered an invalid match
            max_iou_distance=0.7,  # Gating threshold. Associations with cost larger than this value are disregarded.
            max_age=70,            # Maximum number of missed misses (prediction calls, i.e. frames I think) before a track is deleted
            n_init=3,              # Number of frames that a track remains in initialization phase
            nn_budget=100,         # Maximum size of the appearance descriptors gallery)
            mc_lambda=0.995,       # matching with both appearance (1 - MC_LAMBDA) and motion cost
            ema_alpha=0.9          # updates  appearance  state in  an exponential moving average manner
            )         
    else:
        raise NotImplementedError("I noticed that StrongSORT is using the same code as DeepSORT, i.e. use model_type = 'strongsort'.")
        model = DeepSort(
            max_iou_distance=0.7,
            max_age=30,
            n_init=3,
            nms_max_overlap=1,
            max_cosine_distance=0.2,
            nn_budget=None,
            gating_only_position=False,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
    return model


def update_deepsort(tracker, xywhs, confs, clss, frame):
    # Convert tensors to Python lists
    xywhs_list = xywhs.tolist()
    confs_list = confs.tolist()
    clss_list = clss.tolist()

    # Create raw detections list
    raw_detections = []
    for i in range(len(xywhs_list)):
        detection = (xywhs_list[i], confs_list[i], clss_list[i])
        raw_detections.append(detection)

    tracks = tracker.update_tracks(raw_detections, frame=frame)

    outputs = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltwh()  # alternative formats: original_ltwh, to_tlwh(), to_ltwh(), to_tlbr(), to_ltrb()
        left, top, width, height = bbox
        center_x = int(left + width / 2)
        center_y = int(top + height / 2)
        bbox = [center_x, center_y, width, height]
        if not isinstance(bbox, list):
            bbox = bbox.tolist() # only necessary if above calculations are not saved in a list already
        tracked_id = track.track_id
        class_id = track.det_class
        confidence = track.det_conf
        detection = bbox + [int(tracked_id), int(class_id), confidence]  # Combine bbox, class_id, and confidence
        outputs.append(detection)
    
    # Currently does not work, detections for "lost" tracks are saved, but confidence is NoneType
    return outputs


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
        weights_obj=ROOT / 'yolov5s.pt',  # model_obj path or triton URL
        weights_hand=ROOT / 'hand.pt',  # model_obj path or triton URL
        weights_tracker= ROOT / 'osnet_x0_25_market1501.pt',
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
        classes_obj=[1,39,40,41,45,46,47,58,74],  # filter by class /  check coco.yaml file or obj_name_dict variable in this script
        classes_hand=[0,1], 
        class_hand_nav=[80,81],
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
        manual_entry=False, # True means you will control the exp manually versus the standard automatic running
        mock_navigate=True # True means that navigation will be conducted only via print commands without connecting the bracelet
):

    # region main setup
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

    # Experiment setup
    if manual_entry == False:
        target_objs = ['apple','banana','potted plant','bicycle','cup','clock','wine glass']
        target_objs = ['apple' for i in range(20)] # debugging
        obj_index = 0
        gave_command = False
        print(f'The experiment will be run automatically. The selected target objects, in sequence, are:\n{target_objs}')
    else:
        print('The experiment will be run manually. You will enter the desired target for each run yourself.')

    horizontal_in, vertical_in = False, False
    target_entered = False
    play_start()  # play welcome sound

    # Configure flags
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

    # Load object detection models
    device = select_device(device)
    model_obj = DetectMultiBackend(weights_obj, device=device, dnn=dnn, fp16=half)
    model_hand = DetectMultiBackend(weights_hand, device=device, dnn=dnn, fp16=half)

    stride_obj, names_obj, pt_obj = model_obj.stride, model_obj.names, model_obj.pt
    stride_hand, names_hand, pt_hand = model_hand.stride, model_hand.names, model_hand.pt
    imgsz = check_img_size(imgsz, s=stride_obj)  # check image size
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # Dataloader
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
    model_type='strongsort'
    tracker = load_tracker(model_type=model_type, weights=weights_tracker, device=device)

    # Warmup models
    model_obj.warmup(imgsz=(1 if pt_obj or model_obj.triton else bs, 3, *imgsz))
    model_hand.warmup(imgsz=(1 if pt_hand or model_hand.triton else bs, 3, *imgsz))
    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()

    # Initialize vars for tracking
    prev_frames = None
    curr_frames = None
    outputs = []

    # endregion

    # Process the whole dataset / stream
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

        # Start timer for fps measure
        start = time.perf_counter()

        # Image pre-processing
        with dt[0]:
            image = torch.from_numpy(im).to(model_obj.device)
            image = image.half() if model_obj.fp16 else image.float()  # uint8 to fp16/32
            image /= 255  # 0 - 255 to 0.0 - 1.0
            if len(image.shape) == 3:
                image = image[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred_obj = model_obj(image, augment=augment, visualize=visualize)
            pred_hand = model_hand(image, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred_obj = non_max_suppression(pred_obj, conf_thres, iou_thres, classes_obj, agnostic_nms, max_det=max_det)
            pred_hand = non_max_suppression(pred_hand, conf_thres, iou_thres, classes_hand, agnostic_nms, max_det=max_det)

        # region main object tracking
        # Process predictions
        for i, (hand,object) in enumerate(itertools.zip_longest(pred_hand,pred_obj)):  # per image
            seen += 1

            # i always equals 0 (per frame there is just one prediction object, because just one source)
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_obj))

            curr_frames = im0

            # Camera motion compensation for hand tracker (ECC)
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:
                    tracker.tracker.camera_update(prev_frames, curr_frames)

            # Pre-process detections
            if len(hand):
                hand, s = preprocess_detections(hand, im, im0, master_label, s, index_add)
            if len(object):
                object, s = preprocess_detections(object, im, im0, master_label, s)

            if len(hand) or len(object):
                # Track hands and objects
                xywhs_hand = xyxy2xywh(hand[:, 0:4])
                confs_hand = hand[:, 4]
                clss_hand = hand[:, 5]
                xywhs_obj = xyxy2xywh(object[:, 0:4])
                confs_obj = object[:, 4]
                clss_obj = object[:, 5]
                # Concatenate tracked hands and objects and hands and update tracker
                xywhs = torch.cat((xywhs_hand, xywhs_obj), dim=0)
                confs = torch.cat((confs_hand, confs_obj), dim=0)
                clss = torch.cat((clss_hand, clss_obj), dim=0)

                if model_type == 'strongsort':
                    outputs = tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                elif model_type == 'deepsort':
                    outputs = update_deepsort(tracker, xywhs, confs, clss, im0)
                    print(f'Detections:\n{outputs}')
                else:
                    print('Tracker was not correctly initialized.')

                # Write results to annotator (visualization)
                for *xyxy, obj_id, cls, conf in outputs:
                    id = int(obj_id)
                    c = int(cls)

                    # add BBs to annotator here
                    if save_img or save_crop or view_img:  # Add bbox to image
                        label = None if hide_labels else (f'ID: {id} {master_label[c]}' if hide_conf else (f'ID: {id} {master_label[c]} {conf:.2f}'))
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / txt_file_name / master_label[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

            # Get FPS
            end = time.perf_counter()
            runtime = end - start
            fps = 1 / runtime
            prev_frames = curr_frames
            # Save (running mean) FPS

        # Stream results
        im0 = annotator.result()
        if view_img:
            #cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # for resizing
            cv2.putText(im0, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 1)
            cv2.imshow(str(p), im0)
            #cv2.resizeWindow(str(p), im0.shape[1]//2, im0.shape[0]//2) # for resizing

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
        #LOGGER.info(f"{s}{'' if len(hand) or len(object) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # endregion


        # region main navigation loop
        # After passing target object class hand is navigated in each frame until grasping command is sent
        if len(outputs) > 0:
            for det in range(len(outputs)):
                outputs[det, :4] = xyxy_to_xywh(outputs[det, :4])

        if manual_entry == True:
            if target_entered == False:
                user_in = "n"
                while user_in == "n":
                    print("These are the available objects:")
                    print(obj_name_dict)
                    target_obj_verb = input('Enter the object you want to target: ')

                    if target_obj_verb in obj_name_dict.values():
                        user_in = input("Selected object is " + target_obj_verb + ". Correct? [y,n]")
                        class_target_obj = next(key for key, value in obj_name_dict.items() if value == target_obj_verb)
                        file = ROOT / f'resources/sound/{target_obj_verb}.mp3'
                        #playsound(str(file))
                        # Start trial time measure (end in navigate_hand(...))
                    else:
                        print(f'The object {target_obj_verb} is not in the list of available targets. Please reselect.')

                target_entered = True
                grasp = False
                horizontal_in, horizontal_out = False, False
                vertical_in, vertical_out = False, False
                obj_seen_prev, search, navigating = False, False, False
                count_searching, count_see_object, jitter_guard = 0,0,0

            elif target_entered:
                pass

            # Navigate the hand based on information from last frame and current frame detections
            if not mock_navigate:
                horizontal_out, vertical_out, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
                    navigate_hand(belt_controller, outputs, class_target_obj, class_hand_nav, horizontal_in, vertical_in, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating)
            else:
                horizontal_out, vertical_out, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
                    mock_navigate_hand(outputs, class_target_obj, class_hand_nav, horizontal_in, vertical_in, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating)
            # Exit the loop if hand and object aligned horizontally and vertically and grasp signal was sent
            if grasp:
                target_entered = False

            # Set values of the inputs for the next loop iteration
            if horizontal_out:
                horizontal_in = True
            if vertical_out:
                vertical_in = True

        else:
            target_obj_verb = target_objs[obj_index]
            class_target_obj = next(key for key, value in obj_name_dict.items() if value == target_obj_verb)

            if gave_command == False:
                file = ROOT / f'resources/sound/{target_obj_verb}.mp3'
                #playsound(str(file))
                grasp = False
                horizontal_in, horizontal_out = False, False
                vertical_in, vertical_out = False, False
                gave_command = True
                obj_seen_prev, search, navigating = False, False, False
                count_searching, count_see_object, jitter_guard = 0,0,0

            # Navigate the hand based on information from last frame and current frame detections
            if not mock_navigate:
                horizontal_out, vertical_out, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
                    navigate_hand(belt_controller, outputs, class_target_obj, class_hand_nav, horizontal_in, vertical_in, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating)
            else:
                horizontal_out, vertical_out, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
                    mock_navigate_hand(outputs, class_target_obj, class_hand_nav, horizontal_in, vertical_in, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating)

            if grasp and ((obj_index+1)<=len(target_objs)):
                #gave_command = False
                obj_index += 1

            if obj_index == len(target_objs):
                file = ROOT / f'resources/sound/ending.mp3'
                #playsound(str(file))
                print('Experiment Completed')
                break

            # Exit the loop if hand and object aligned horizontally and vertically and grasp signal was sent
            if horizontal_out and vertical_out and grasp:
                gave_command = False

            # Set values of the inputs for the next loop iteration
            if horizontal_out:
                horizontal_in = True
            if vertical_out:
                vertical_in = True
        
        # endregion

        

if __name__ == '__main__':
    '''
    Function that navigates hand toward target object based on input from object and hand detectors.
    '''
    #check_requirements(requirements=ROOT / '../requirements.txt', exclude=('tensorboard', 'thop'))

    weights_obj = 'yolov5s.pt'  # Object model weights path
    weights_hand = 'hand.pt' # Hands model weights path
    weights_tracker = 'osnet_x0_25_market1501.pt' # ReID weights path
    source = '1' # image/video path or camera source (0 = webcam, 1 = external, ...)
    mock_navigate = True # Navigate without the bracelet using only print commands

    run(weights_obj=weights_obj, weights_hand=weights_hand, weights_tracker=weights_tracker, source=source, mock_navigate=mock_navigate)
