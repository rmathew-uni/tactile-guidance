# System
import os
import platform
import sys
from pathlib import Path
import itertools
import time
import numpy as np
import requests
from playsound import playsound
import threading

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

def get_midas_weights(model_type):

    path = f'./MiDaS/weights/{model_type}.pt'

    # Download weights if not available
    if not os.path.exists(path):
        print("File does not exist. Downloading weights...")

        # Get version from model type
        if 'v21' in model_type:
            version = 'v2_1'
        elif model_type == 'dpt_large_384' or model_type == 'dpt_hybrid_384':
            version = 'v3'
        else:
            print('Fallback to latest version V3.1 (May 2024).')
            version = 'v3_1'
        
        # Create and download from URL
        url = f'https://github.com/isl-org/MiDaS/releases/download/{version}/{model_type}.pt'
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                file.write(response.content)
            print("Weights downloaded successfully!")
        else:
            print("Failed to download weights file. Status code:", response.status_code)
    else:
        print("Weights already exists!")

    weights = f'./MiDaS/weights/{model_type}.pt'

    return weights


def get_depth(image, transform, device, model, model_type, net_w, net_h, vis=False, sides=False, bbs=None):
    """
    Depth Estimation with MiDaS.
    """
    if image.max() > 1:
        img = np.flip(image, 2)  # in [0, 255] (flip required to get RGB)
        img = img/255
    img_resized = transform({"image": img})["image"]

    depth = process(device, model, model_type, img_resized, (net_w, net_h),
                            image.shape[1::-1], False, True) # webcam: (720,1280)

    if vis:
        original_image_bgr = np.flip(image, 2) if sides else None
        content = create_side_by_side(original_image_bgr, depth, False)
        cv2.imshow('MiDaS Depth Estimation', content/255)

    if bbs is not None:
        outputs = []
        for bb in bbs:
            x,y,x2,y2 = [int(coord) for coord in bb[:4]]
            roi = depth[y:y2, x:x2]
            mean_depth = np.mean(roi)
            median_depth = np.median(roi) # probably works better
            print(f'Mean depth: {mean_depth}, Median depth: {median_depth}')
            bb = np.append(bb, mean_depth)
            outputs.append(bb)

    return depth, np.array(outputs)

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


class AutoAssign:

    def __init__(self, **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)


class BraceletController(AutoAssign):

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

    def load_object_detector(self):
        
        print(f'\nLOADING OBJECT DETECTORS')
        
        self.device = select_device(self.device)
        self.model_obj = DetectMultiBackend(self.weights_obj, device=self.device, dnn=self.dnn, fp16=self.half)
        self.model_hand = DetectMultiBackend(self.weights_hand, device=self.device, dnn=self.dnn, fp16=self.half)

        self.stride_obj, self.names_obj, self.pt_obj = self.model_obj.stride, self.model_obj.names, self.model_obj.pt
        self.stride_hand, self.names_hand, self.pt_hand = self.model_hand.stride, self.model_hand.names, self.model_hand.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride_obj) # check image size
        self.dt = (Profile(), Profile(), Profile())

        print(f'\nOBJECT DETECTORS LOADED SUCCESFULLY')


    def load_object_tracker(self):

        print(f'\nLOADING OBJECT TRACKER')
        
        self.tracker = StrongSORT(
                model_weights=self.weights_tracker, 
                device=self.device,
                fp16=False,
                max_dist=0.5,          # The matching threshold. Samples with larger distance are considered an invalid match
                max_iou_distance=0.7,  # Gating threshold. Associations with cost larger than this value are disregarded.
                max_age=70,            # Maximum number of missed misses (prediction calls, i.e. frames I think) before a track is deleted
                n_init=1,              # Number of frames that a track remains in initialization phase --> if 0, track is confirmed on first detection
                nn_budget=100,         # Maximum size of the appearance descriptors gallery
                mc_lambda=0.995,       # matching with both appearance (1 - MC_LAMBDA) and motion cost
                ema_alpha=0.9          # updates  appearance  state in  an exponential moving average manner
                )
        
        print(f'\nOBJECT TRACKER LOADED SUCCESFULLY')


    def load_depth_estimator(self):
        
        print(f'\nLOADING DEPTH ESTIMATOR')

        self.weights_DE = get_midas_weights(self.depth_estimator)
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device,
                                                                        self.weights_DE,
                                                                        self.depth_estimator,
                                                                        optimize=False,
                                                                        height=640,
                                                                        square=False)

        print(f'\nDEPTH ESTIMATOR LOADED SUCCESFULLY')
        

    def warmup_model(self, model, type='detector'):

        print(f'\nWARMING UP MODEL...')

        if type == 'detector':
            model.warmup(imgsz=(1 if self.pt_obj or self.model_obj.triton else self.bs, 3, *self.imgsz))
        
        if type == 'tracker':
            model.warmup()

    def experiment_loop(self, save_dir, save_img, index_add, vid_path, vid_writer):

        print(f'\nSTARTING MAIN LOOP')

        # Initialize vars for tracking
        prev_frames = None
        curr_frames = None
        fpss = []
        outputs = []

        # Data processing: Iterate over each frame of the live stream
        for frame, (path, im, im0s, vid_cap, _) in enumerate(self.dataset):

            # Start timer for FPS measure
            start = time.perf_counter()

            # Setup saving and visualization
            p, im0 = Path(path[0]), im0s[0].copy() # idx 0 is for first source (and we only have one source)
            save_path = str(save_dir / p.name)  # im.jpg
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names_obj))

            # Image pre-processing
            with self.dt[0]:
                image = torch.from_numpy(im).to(self.model_obj.device)
                image = image.half() if self.model_obj.fp16 else image.float()  # uint8 to fp16/32
                image /= 255  # 0 - 255 to 0.0 - 1.0
                if len(image.shape) == 3:
                    image = image[None]  # expand for batch dim

            # Object detection inference
            with self.dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred_target = self.model_obj(image, augment=self.augment, visualize=visualize)
                pred_hand = self.model_hand(image, augment=self.augment, visualize=visualize)

            # Non-maximal supression
            with self.dt[2]:
                pred_target = non_max_suppression(pred_target, self.conf_thres, self.iou_thres, self.classes_obj, self.agnostic_nms, max_det=self.max_det) # list containing one tensor (n,6)
                pred_hand = non_max_suppression(pred_hand, self.conf_thres, self.iou_thres, self.classes_hand, self.agnostic_nms, max_det=self.max_det) # list containing one tensor (n,6)

            for hand in pred_hand[0]:
                if len(hand):
                    hand[5] += index_add # assign correct classID by adding len(coco_labels)

            # Camera motion compensation for tracker (ECC)
            if self.run_object_tracker:
                curr_frames = im0
                self.tracker.tracker.camera_update(prev_frames, curr_frames)
            
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
            if self.run_object_tracker:
                outputs = self.tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0) # returns xyxys again
            else:
                outputs = np.array(preds)
                outputs = np.insert(outputs, 4, -1, axis=1) # insert track_id placeholder
                outputs[:, [5, 6]] = outputs[:, [6, 5]] # switch cls and conf columns for alignment with tracker

            # Depth estimation
            depth_img, outputs = get_depth(im0, self.transform, self.device, self.model, self.depth_estimator, self.net_w, self.net_h, vis=False, bbs=outputs)

            # Get FPS
            end = time.perf_counter()
            runtime = end - start
            fps = 1 / runtime
            fpss.append(fps)
            prev_frames = curr_frames

            # region visualization
            # Write results
            for *xyxy, obj_id, cls, conf, depth in outputs:
                id, cls = int(obj_id), int(cls)
                if save_img or self.save_crop or self.view_img:
                    label = None if self.hide_labels else (f'ID: {id} {self.master_label[cls]}' if self.hide_conf else (f'ID: {id} {self.master_label[cls]} {conf:.2f} {depth:.2f}'))
                    annotator.box_label(xyxy, label, color=colors(cls, True))

            # Display results
            im0 = annotator.result()
            if self.view_img:
                #cv2.namedWindow("AIBox", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # for resizing
                cv2.putText(im0, f'FPS: {int(fps)}, Avg: {int(np.mean(fpss))}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 1)
                #cv2.imshow("AIBox", im0) # original image only
                side_by_side = create_side_by_side(im0, depth_img, False) # original image & depth side-by-side
                cv2.imshow("AIBox & Depth", side_by_side)
                #cv2.resizeWindow("AIBox", im0.shape[1]//2, im0.shape[0]//2) # for resizing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save results
            if save_img:
                if self.dataset.mode == 'image':
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
                    outputs[det, :4] = xyxy2xywh(outputs[det, :4])

            # endregion

            # region main navigation
            # Get the target object class
            if not self.target_entered:
                if self.manual_entry:
                    user_in = "n"
                    while user_in == "n":
                        print("These are the available objects:")
                        print(coco_labels)
                        target_obj_verb = input('Enter the object you want to target: ')

                        if target_obj_verb in coco_labels.values():
                            user_in = input("Selected object is " + target_obj_verb + ". Correct? [y,n]")
                            class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                            file = f'resources/sound/{target_obj_verb}.mp3'
                            #playsound(str(file))
                            # Start trial time measure (end in navigate_hand(...))
                        else:
                            print(f'The object {target_obj_verb} is not in the list of available targets. Please reselect.')
                else:
                    target_obj_verb = self.target_objs[self.obj_index]
                    class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                    file = f'resources/sound/{target_obj_verb}.mp3'
                    #playsound(str(file))
                    # Start trial time measure (end in navigate_hand(...))

                self.target_entered = True

            # Navigate the hand based on information from last frame and current frame detections
            if frame == 0: # Initialize navigation
                horizonal, vertical, dist, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
                navigate_hand(self.belt_controller,
                              outputs,
                              class_target_obj,
                              self.class_hand_nav)

            horizonal, vertical, dist, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating = \
                navigate_hand(self.belt_controller,
                            outputs,
                            class_target_obj,
                            self.class_hand_nav,
                            horizonal,
                            vertical,
                            dist,
                            grasp,
                            obj_seen_prev,
                            search,
                            count_searching,
                            count_see_object,
                            jitter_guard,
                            navigating)
        
            # Exit the loop if hand and object aligned horizontally and vertically and grasp signal was sent
            if grasp:
                if self.manual_entry and ((self.obj_index+1)<=len(self.target_objs)):
                    self.obj_index += 1
                self.target_entered = False


    @smart_inference_mode()
    def run(self):

        # region main setup

        # Experiment setup
        if not self.manual_entry:
            target_objs = self.target_objs
            self.obj_index = 0
            print(f'The experiment will be run automatically. The selected target objects, in sequence, are:\n{target_objs}')
        else:
            print('The experiment will be run manually. You will enter the desired target for each run yourself.')

        horizontal_in, vertical_in = False, False
        self.target_entered = False
        #play_start()  # play welcome sound

        # Configure saving
        source = self.source

        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')

        if is_url and is_file:
            source = check_file(source)  # download

        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        if save_img:
            (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load object detection models
        self.load_object_detector()

        # Load data stream
        self.bs = 1  # batch_size
        view_img = check_imshow(warn=True)
        try:
            self.dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride_obj, auto=True, vid_stride=self.vid_stride)
        except AssertionError:
            while True:
                change_camera = input(f'Failed to open camera with index {source}. Do you want to continue with webcam? (Y/N)')
                if change_camera == 'Y':
                    source = '0'
                    self.dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride_obj, auto=True, vid_stride=self.vid_stride)
                    break
                elif change_camera == 'N':
                    exit()
        self.bs = len(self.dataset)
        vid_path, vid_writer = [None] * self.bs, [None] * self.bs

        # Create combined label dictionary
        index_add = len(self.names_obj)
        labels_hand_adj = {key + index_add: value for key, value in self.names_hand.items()}
        self.master_label = self.names_obj | labels_hand_adj

        # Load tracker model
        if self.run_object_tracker:
            self.load_object_tracker()
        else:
            print('SKIPPING OBJECT TRACKER INITIALIZATION')

        # Load depth estimator
        if self.run_depth_estimator:
            self.load_depth_estimator()
        else:
            print('SKIPPING DEPTH ESTIMATOR INITIALIZATION')

        # Warmup models
        self.warmup_model(self.model_obj)
        self.warmup_model(self.model_hand)
        self.warmup_model(self.tracker.model,'tracker')

        # endregion

        # Start experiment loop
        self.experiment_loop(save_dir, save_img, index_add, vid_path, vid_writer)

if __name__ == '__main__':

    mock_navigate = True
    belt_controller = None

    # Check bracelet connection
    if not mock_navigate:
        connection_check, belt_controller = connect_belt()
        if connection_check:
            print('Bracelet connection successful.')
        else:
            print('Error connecting bracelet. Aborting.')
            sys.exit()

    bracelet_controller = BraceletController(weights_obj='yolov5s.pt',  # model_obj path or triton URL # ROOT
                            weights_hand='hand.pt',  # model_obj path or triton URL # ROOT
                            weights_tracker='osnet_x0_25_market1501.pt', # ROOT
                            depth_estimator='midas_v21_small_256',
                            source='1',  # file/dir/URL/glob/screen/0(webcam) # ROOT
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
                            run_object_tracker=True,
                            run_depth_estimator=True,
                            mock_navigate=True,
                            belt_controller=belt_controller,
                            target_objs = ['bottle' for _ in range(5)]) # debugging

    try:
        bracelet_controller.run()
    except KeyboardInterrupt:
        close_app(belt_controller)
        
    # In the end, kill everything
    close_app(belt_controller)