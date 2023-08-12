from pycocotools.coco import COCO
import requests
import sys

# instantiate COCO specifying the annotations json path
coco_train = COCO('./datasets/instances_train2017.json')
coco_val = COCO('./datasets/instances_val2017.json')

obj_list = [['potted plant'], ['clock'], ['bicycle'], ['wine glass'],['bottle'],
            ['apple'],['orange'],['banana'],['bowl'],['cell phone'],['spoon'],
            ['zebra'],['knife'],['fork'],['car'],['motorcycle'],['cup'],['pizza'],
            ['scissors'],['toothbrush']]

# looping through fruits_list and downloading the image to the path (cocodataset/coco_fruits)
for obj in obj_list:
    # index of category (e.g banana is 52)
    catIds_train = coco_train.getCatIds(catNms=obj)
    catIds_val = coco_val.getCatIds(catNms=obj)
    # Get the corresponding image ids and images using loadImgs
    imgIds_train = coco_train.getImgIds(catIds=catIds_train)
    images_train = coco_train.loadImgs(imgIds_train)
    imgIds_val = coco_val.getImgIds(catIds=catIds_val)
    images_val = coco_val.loadImgs(imgIds_val)

    # Save the images into a local folder
    for im in images_train:
        img_data = requests.get(im['coco_url']).content
        with open('./datasets/coco_subset/train/images/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)
        
    for im in images_val:
        img_data = requests.get(im['coco_url']).content
        with open('./datasets/coco_subset/val/images' + im['file_name'], 'wb') as handler:
            handler.write(img_data)