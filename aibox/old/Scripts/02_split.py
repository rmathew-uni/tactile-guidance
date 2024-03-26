import os
import shutil
import random

# Define the paths to your dataset folders
dataset_folder = '/Users/florian/Documents/Studium/NBP/Projects/OptiVisT/AIBox/Dataset/Datasets/COCO_balanced/'
images_folder = os.path.join(dataset_folder, 'images/')
labels_folder = os.path.join(dataset_folder, 'labels/')

# Define the split percentages
train_percent = 0.9  # 70% for training
val_percent = 0.05  # 15% for validation
test_percent = 0.05  # 15% for testing

# Create the train, val, and test folders
train_folder = os.path.join(dataset_folder, 'train/')
val_folder = os.path.join(dataset_folder, 'val/')
test_folder = os.path.join(dataset_folder, 'test/')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

os.makedirs(os.path.join(train_folder, 'images/'), exist_ok=True)
os.makedirs(os.path.join(train_folder, 'labels/'), exist_ok=True)
os.makedirs(os.path.join(val_folder, 'images/'), exist_ok=True)
os.makedirs(os.path.join(val_folder, 'labels/'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'images/'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'labels/'), exist_ok=True)

# Get the list of image files in the dataset
image_files = [file for file in images_folder if file.endswith('.jpg')]
random.shuffle(image_files)

# Calculate the number of images for each split
total_images = len(image_files)
num_train = int(total_images * train_percent)
num_val = int(total_images * val_percent)
num_test = total_images - num_train - num_val

# Move images and corresponding labels to the respective split folders
for i, image_file in enumerate(image_files):

    print(f'Processing file {i}/{len(image_files)}...')

    label_file = os.path.splitext(image_file)[0] + '.txt'

    if i < num_train:
        destination_folder = train_folder
    elif i < num_train + num_val:
        destination_folder = val_folder
    else:
        destination_folder = test_folder

    # Move the image file
    shutil.move(os.path.join(images_folder, image_file), os.path.join(destination_folder, 'images/'))

    # Move the label file
    shutil.move(os.path.join(labels_folder, label_file), os.path.join(destination_folder, 'labels/'))

print(f"Split dataset into {num_train} training samples, {num_val} validation samples, and {num_test} test samples.")
