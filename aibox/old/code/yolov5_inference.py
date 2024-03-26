import torch
import sys
import cv2
import torch
from PIL import Image
from pathlib import Path

# Model
# https://pytorch.org/hub/ultralytics_yolov5/
# To load the Yolov5 model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

Parent_directory = str(Path(__file__).parent)
model_directlry = "/runs/train/exp6/weights/best.pt"

model_path = Parent_directory + model_directlry

# Loading the model that is trained on costume dataset
model = torch.hub.load('', 'custom',
                       path=model_path,
                       source='local')  # local repo


# Images
# reading the images with PIL
im1 = Image.open('data/images/bus.jpg')  # PIL image

# reading the image with cv2
# im2 = cv2.imread('data/images/bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model([im1], size=640) # batch of images

# Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save()










