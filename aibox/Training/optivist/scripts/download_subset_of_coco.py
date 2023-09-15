from pycocotools.coco import COCO
import requests
import sys
from tqdm import tqdm

# instantiate COCO specifying the annotations json path
coco = COCO('/share/neurobiopsychologie/datasets/coco_subset_optivist/instances_train2017.json')

items_list = [['potted plant'], ['clock'], ['bicycle'], ['wine glass'],['bottle'],['apple'],['orange'],['banana'],['bowl'],['cell phone'],['spoon'],['zebra'],['knife'],['fork'],['car'],['motorcycle'],['cup'],['pizza'],['scissors'],['toothbrush']]
# looping through items_list and downloading the image to the path 
for item in items_list:
    # index of category (e.g banana is 52)
    catIds = coco.getCatIds(catNms=item)
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    for im in tqdm(images):
        img_data = requests.get(im['coco_url']).content
        with open('/share/neurobiopsychologie/datasets/coco_subset_optivist/train/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)