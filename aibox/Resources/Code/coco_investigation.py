import json
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

# path for annotation instances json file
path = "cocodataset/annotations/instances_val2017.json"
f = open(path)
anns = json.load(f)
print(anns.keys())

ann_file = "cocodataset/annotations/instances_val2017.json"
coco = COCO(ann_file)

# Get list of category_ids, depending on the category (e.g. banana is 52)
category_ids = coco.getCatIds(['banana'])
print(category_ids)


# Get list of image_ids which contain banana (52)
image_ids = coco.getImgIds(catIds=[52])
# example of image_ids
print(image_ids[0:5])


image_id = 6012
# Get all banana annotations for image 000000006012.jpg
annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=[52])
# print(len(annotation_ids))


# annotation loaded into list
anns = coco.loadAnns(annotation_ids)

# accessing the bounding boxes
for ann in anns:
    print(ann['bbox'])

# Visualization 
# image folder
images_path = "cocodataset/images/val2017/"
image_name = str(image_id).zfill(12) + ".jpg"  # Image names are 12 characters long
image = Image.open(images_path + image_name)
print(image.size)
fig, ax = plt.subplots()

# Draw boxes and add label to each box
for ann in anns:
    box = ann['bbox']
    bb = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor="blue", facecolor="none")
    ax.add_patch(bb)

    # Add red dot to the center
    center_x = box[0] + box[2] / 2  # x+(width/2)
    center_y = box[1] + box[3] / 2  # y+(height/2)
    ax.scatter(center_x, center_y, color='red')

ax.imshow(image)
plt.show()

