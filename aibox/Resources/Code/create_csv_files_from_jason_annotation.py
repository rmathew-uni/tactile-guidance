import csv
import json
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

path = "cocodataset/annotations/instances_val2017.json"
f = open(path)
anns = json.load(f)
print(anns.keys())

# path for jason annotation
ann_file = "cocodataset/annotations/instances_val2017.json"
coco = COCO(ann_file)

# Get list of category_ids, cased on category
category_ids = coco.getCatIds(['banana', 'apple', 'orange'])
# print(category_ids)

# write dictionary of category_ids as key which contain image_id as values
dict_all = {}
for idx in category_ids:
    image_ids = coco.getImgIds(catIds=idx)
    dict_all[idx] = image_ids


# Access the bounding boxes and add rows to the CSV list
for category_idx in dict_all:

    for i, image_idx in enumerate(dict_all[category_idx]):
        # Create a list to store the CSV rows
        csv_rows = []
        # Get all annotations for image image_idx
        annotation_ids = coco.getAnnIds(imgIds=image_idx, catIds=category_idx)
        anns = coco.loadAnns(annotation_ids)

        images_path = "cocodataset/images/val2017/"
        image_name = str(image_idx).zfill(12) + ".jpg"  # Image names are 12 characters long
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

            # change idx
            if category_idx == 52: # banana
                category_index = 0
            elif category_idx == 53: # apple
                category_index = 1
            elif category_idx == 55: # oranges
                category_index = 2


            csv_rows.append([category_index, bbox_yolo[0], bbox_yolo[1], bbox_yolo[2], bbox_yolo[3]])

            # Define the CSV file path
            image_name = str(image_idx).zfill(12)  # Image names are 12 characters long
            csv_file_path = f'csv/{image_name}.csv'

            # Write the CSV file
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['category_type', 'category_index', 'image_idx', 'bbox'])  # Write header row
                writer.writerows(csv_rows)  # Write data rows

            print(f"CSV file '{csv_file_path}' has been generated.")

