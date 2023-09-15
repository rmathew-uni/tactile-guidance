import csv
import json
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from tqdm import tqdm

path = "/share/neurobiopsychologie/datasets/coco_subset_optivist/instances_train2017.json"
f = open(path)
anns = json.load(f)
print(anns.keys())

# path for jason annotation
ann_file = "/share/neurobiopsychologie/datasets/coco_subset_optivist/instances_train2017.json"
coco = COCO(ann_file)

# Get list of category_ids, cased on category
category_ids = coco.getCatIds(['potted plant', 'clock', 'bicycle', 'wine glass','bottle','apple','orange','banana','bowl','cell phone','spoon','zebra','knife','fork','car','motorcycle','cup','pizza','scissors','toothbrush'])

category_ids.sort()

# print(category_ids)

# write dictionary of category_ids as key which contain image_id as values
dict_all = {}
for idx in category_ids:
    image_ids = coco.getImgIds(catIds=idx)
    dict_all[idx] = image_ids

category_index = 0

# Access the bounding boxes and add rows to the CSV list
for category_idx in dict_all:

    for i, image_idx in enumerate(dict_all[category_idx]):
        # Create a list to store the CSV rows
        csv_rows = []
        # Get all annotations for image image_idx
        annotation_ids = coco.getAnnIds(imgIds=image_idx, catIds=category_idx)
        anns = coco.loadAnns(annotation_ids)

        images_path = "/share/neurobiopsychologie/datasets/coco_subset_optivist/train/"
        image_name = str(image_idx).zfill(12) + ".jpg"  # Image names are 12 characters long
        image = Image.open(images_path + image_name)

        # shape of the image
        w = image.size[0]
        h = image.size[1]
        print(f"Image width & height: {w, h}")

        # going through bbox annotations
        for ann in anns:
            bbox = ann['bbox']
            print(f"BBox: {bbox}")
            print(f"BBox[0]: {bbox[0]}")
            print(f"BBox[1]: {bbox[1]}")
            print(f"BBox[2]: {bbox[2]}")
            print(f"BBox[3]: {bbox[3]}")
            center_x = bbox[0] + bbox[2] / 2  # x+(width/2)
            center_y = bbox[1] + bbox[3] / 2  # y+(height/2)
            # division to w, h of image to bring number between 0,1
            bbox_yolo = [center_x/w,
                         center_y/h,
                         bbox[2]/w,
                         bbox[3]/h]

            csv_rows.append([category_index, bbox_yolo[0], bbox_yolo[1], bbox_yolo[2], bbox_yolo[3]])

            # Define the CSV file path
            image_name = str(image_idx).zfill(12)  # Image names are 12 characters long
            csv_file_path = f'/share/neurobiopsychologie/datasets/coco_subset_optivist/csv_train/{image_name}.csv'

            # Write the CSV file
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['category_type', 'category_index', 'image_idx', 'bbox'])  # Write header row
                writer.writerows(csv_rows)  # Write data rows

            print(f"CSV file '{csv_file_path}' has been generated.")

    category_index+=1
