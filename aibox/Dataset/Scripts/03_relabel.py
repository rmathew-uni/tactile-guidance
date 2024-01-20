import os

# Define your dataset directory containing label and image files
dataset_dir = '/Users/florian/Documents/Studium/NBP/Projects/OptiVisT/AIBox/Dataset/Datasets/cocohands_balanced/'
folders = ['test/', 'train/', 'val/']

for split in folders:

    # Define the path to your labels folder
    labels_folder = os.path.join(dataset_dir, split, 'labels/')

    # Mapping for class updates: hand labels are 0,1,2,3 -> all coco labels to 4-17
    class_mapping = {
        1: 4, 
        2: 5, 
        3: 6, 
        22: 7, 
        39: 8, 
        40: 9, 
        41: 10, 
        43: 11, 
        45: 12, 
        49: 13, 
        58: 14, 
        74: 15, 
        76: 16, 
        79: 17
    }

    # Iterate through label files and update classes
    print(labels_folder)
    label_files = [file for file in os.listdir(labels_folder) if file.endswith('.txt')]
    print(label_files)
    for label_file in label_files:
        label_path = os.path.join(labels_folder, label_file)

        if "frame" not in label_file:
            # If "frame" is not in the filename, update the classes
            with open(label_path, 'r') as old_label_file:
                lines = old_label_file.readlines()
                print(f"Read old lines: {lines}")
                print()

            updated_lines = []
            for line in lines:
                class_id, *rest = line.split()
                class_id = int(class_id)
                if class_id in class_mapping:
                    new_class_id = str(class_mapping[class_id])
                    updated_lines.append(new_class_id + ' ' + ' '.join(rest) + '\n')

            with open(label_path, 'w') as new_label_file:
                new_label_file.writelines(updated_lines)
                print(f"Old class ID: {class_id} -> new class ID: {new_class_id}")

print("Class labels updated in label files.")