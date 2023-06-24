from pycocotools.coco import COCO
import requests
import sys

# instantiate COCO specifying the annotations json path
coco = COCO('cocodataset/annotations/instances_val2017.json')

fruits_list = [['banana'], ['apple'], ['orange']]

# looping through fruits_list and downloading the image to the path (cocodataset/coco_fruits)
for fruit in fruits_list:
    # index of category (e.g banana is 52)
    catIds = coco.getCatIds(catNms=fruit)
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    for im in images:
        img_data = requests.get(im['coco_url']).content
        with open('cocodataset/coco_fruits/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)