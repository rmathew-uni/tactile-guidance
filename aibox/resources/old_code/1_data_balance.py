import os
import random
import shutil

# Define your dataset directory containing label and image files
dataset_dir = '/Users/florian/Documents/Studium/NBP/Projects/OptiVisT/AIBox/Dataset/Datasets/EgoHands_augmented/'

# Define the desired classes and instances per class
desired_classes = ['0', '1', '2', '3'] # EH: myleft, myright, yourleft, yourright
instances_per_class = {'0': 5000, '1': 5000, '2': 5000, '3': 5000}
#desired_classes = ['58', '74', '1', '40', '39', '49', '45', '22', '43', '2', '3', '41', '76', '79'] # coco
#instances_per_class = {'58': 5000, '74': 5000, '1': 5000, '40': 5000, '39': 5000, '49': 5000, '45': 5000, '22': 5000, '43': 5000, '2': 5000, '3': 5000, '41': 5000, '76': 5000, '79': 5000}

# Create a dictionary to keep track of instances per class
class_instances_count = {class_name: 0 for class_name in desired_classes}

# Create new folders to store selected images and labels
img_dir = os.listdir(os.path.join(dataset_dir, 'images/'))
output_image_dir = os.path.join(dataset_dir, 'images_selection')
output_label_dir = os.path.join(dataset_dir, 'labels_selection')

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

if len(os.listdir(output_image_dir)) != 0:
    print(len(os.listdir(output_image_dir)))
    print(os.listdir(output_image_dir))
    print('It seems there already are images selected. You might want to empty the images and label selection folders before subsetting to your specified number of instances.')

# Iterate through the label files
label_files = [file for file in os.listdir(os.path.join(dataset_dir, 'labels')) if file.endswith('.txt')]
random.shuffle(label_files)  # Shuffle to select randomly
for label_file in label_files:
    label_path = os.path.join(dataset_dir, 'labels/', label_file)
    image_path = os.path.join(dataset_dir, 'images/', label_file).replace('.txt', '.jpg')

    print(f'The classes have the following number of instances: {class_instances_count}')

    # Read the label file and parse bounding box information
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_name, *bbox_values = line.strip().split()
            if class_name in desired_classes and class_instances_count[class_name] < instances_per_class[class_name]:
                class_instances_count[class_name] += 1
                if label_file.replace('.txt', '.jpg') in img_dir:
                    # Move the selected image and label to new folders
                    #shutil.move(image_path, os.path.join(output_image_dir, os.path.basename(image_path)))
                    #shutil.move(label_path, os.path.join(output_label_dir, os.path.basename(label_path)))
                    img_dir.remove(label_file.replace('.txt', '.jpg'))

    # Check if we have reached the desired number of instances for all classes
    if all(count >= instances_per_class[class_name] for class_name, count in class_instances_count.items()):
        break