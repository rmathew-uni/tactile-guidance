import csv
import json
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import shutil

split = 'val'

path = f"/share/neurobiopsychologie/datasets/egohands/{split}_hands.json"
f = open(path)
anns = json.load(f)
print(anns.keys())

# path for jason annotation
ann_file = f"/share/neurobiopsychologie/datasets/egohands/{split}_hands.json"
coco = COCO(ann_file)

# Get list of category_ids, cased on category
category_ids = coco.getCatIds(['hands','myleft','myright','yourleft','yourright'])

category_ids.sort()

# print(category_ids)

# write dictionary of category_ids as key which contain image_id as values
dict_all = {}
image_names = []

for idx in category_ids:

    image_ids = coco.getImgIds(catIds=idx)

    with open(path) as f:
        data = json.load(f)

        for id in image_ids:
            for image_id in data['images']:
                if image_id['id'] == id:
                    image_names.append(image_id['file_name'])

    dict_all[idx] = list(zip(image_names,image_ids))

category_index = 20

dest_image_path = f"/share/neurobiopsychologie/datasets/egohands/{split}_new/"

# Access the bounding boxes and add rows to the CSV list
for category_idx in dict_all:

    for i, (image_name, image_id) in enumerate(dict_all[category_idx]):

        # Create a list to store the CSV rows
        csv_rows = []
        # Get all annotations for image image_idx
        annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=category_idx)
        anns = coco.loadAnns(annotation_ids)

        images_path = f"/share/neurobiopsychologie/datasets/egohands/{split}/"
        image_name = str(image_name)
        image = Image.open(images_path + image_name)

        # shape of the image
        w = image.size[0]
        h = image.size[1]
        # going through bbox annotations
        for ann in anns:
            bbox = ann['bbox']
            center_x = bbox[0] + bbox[2] / 2  # x+(width/2)
            center_y = bbox[1] + bbox[3] / 2  # y+(height/2)
            # division to w, h of image to bring number between 0,1
            bbox_yolo = [center_x/w,
                         center_y/h,
                         bbox[2]/w,
                         bbox[3]/h]

            csv_rows.append([category_index, bbox_yolo[0], bbox_yolo[1], bbox_yolo[2], bbox_yolo[3]])

            # Define the CSV file path
            new_image_name = 'EH' + str(image_id + 3840)

            csv_file_path = f'/share/neurobiopsychologie/datasets/egohands/csv_{split}/{new_image_name}.csv'

            # Write the CSV file
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['category_type', 'category_index', 'image_idx', 'bbox'])  # Write header row
                writer.writerows(csv_rows)  # Write data rows

            print(f"CSV file '{csv_file_path}' has been generated.")

            new_image_path = dest_image_path + new_image_name + ".jpg"

            shutil.copy(images_path + image_name, new_image_path)

            print(f"image file '{new_image_path}' has been generated.")

    category_index+=1

