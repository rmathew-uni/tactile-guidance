# This is script to rescale images to the same target size.

import os
from PIL import Image

# target image size
target_w, target_h = 640, 480

# images input and output paths
input_path = '/share/neurobiopsychologie/datasets/egohands/val'
output_path = '/share/neurobiopsychologie/datasets/egohands/val_img_rescaled/'

print(f"Starting to process {len(os.listdir(input_path))} images")

# iterate through images
counter = 0 # just for the sake of process progress clarity

for image_path in os.listdir(input_path):
    # read image
    image = Image.open(input_path + '/' + image_path)

    # resize image to target dimensions
    image = image.resize((target_w, target_h))

    # save resized image
    image.save(output_path + '/' + image_path)

    # process progress (print out after every 10% of all images processed)
    counter += 1
    if 100*counter/len(os.listdir(input_path)) % 10 == 0:
        print(f"Processed {counter} out of {len(os.listdir(input_path))} images")