import torch
from emnist import extract_test_samples
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from rescale import (load_and_preprocess_image, int_labels_to_emnist_format)

def test(cnn):
    '''Calculates the accuracy of the CNN on the test data'''
    cnn.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            #images, labels = images.cuda(), labels.cuda()
            print(images)
            for i in range(len(images)):
                plt.subplot(5, 8, i + 1)
                plt.imshow((images[i].numpy().transpose([1, 2, 0])+1)/2)
                plt.xticks([])
                plt.yticks([])
            plt.show()            
            test_output = cnn.forward(images)
            print(test_output)
            pred_y = torch.max(test_output, 1)[1]
            print(f"Prediction: {pred_y}, label: {labels}")
            correct += (pred_y == labels).sum()
    #accuracy = correct / 400 # Our test data has 40,000 images
    accuracy = 100 * (correct/len(pred_y))
    print('Test Data Accuracy: {0:.2f}'.format(accuracy))
    return accuracy

def load_images(path, targets):

    images = []

    for target in targets:
        image = load_and_preprocess_image(path + target + '.jpg')
        images.append(image)
    test_images = np.stack(images)

    return test_images

if __name__ == '__main__':
    # Load EMNIST training dataset
    test_images, test_labels = extract_test_samples('digits')
    print(type(test_labels))
    print(test_images.shape)

    # Select an index of the image you want to visualize
    index = 0  # Change this index to see different images

    # Get the image and the corresponding label
    image = test_images[index]
    label = test_labels[index]

    # Visualize the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')  # Hide the axis
    #plt.show()

    #image_path = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/four.jpg' 
    image_path = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/'
    targets = ['zero', 'one', 'two', 'four']
    # Create a 3D array with shape (1, 28, 28)
    #image_array = np.expand_dims(image_array, axis=0)  # Add a new axis at the front
    test_labels = [0, 1, 2, 4]
    # test_images, test_labels = np.expand_dims(load_and_preprocess_image(image_path), axis=0), int_labels_to_emnist_format([4])[0]
    test_images, test_labels = load_images(image_path, targets), int_labels_to_emnist_format(test_labels)

    print(test_images.shape)
    print(type(test_labels))

    # Select an index of the image you want to visualize
    index = 0  # Change this index to see different images

    # Get the image and the corresponding label
    image = test_images[index]
    label = test_labels[index]

    # Visualize the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')  # Hide the axis
    #plt.show()
    #test_images = torch.tensor((test_images/255-0.5).reshape(40000, 1, 28, 28))
    test_images = torch.tensor((test_images/255-0.5).reshape(len(test_labels), 1, 28, 28))
    test_data = list(zip(test_images.float(), test_labels.astype('int64')))

    # Get the image and the corresponding label
    image = test_images.float()[index][index]
    label = test_labels[index]

    # Visualize the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')  # Hide the axis
    plt.show()

    #test_images, test_labels = extract_test_samples('digits')
    #test_images = torch.tensor((test_images/255-0.5).reshape(len(test_labels), 1, 28, 28))
    #test_data = list(zip(test_images.float(), test_labels.astype('int64')))

    # Load and test CNN
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=min(10000,len(test_labels)), shuffle=False)
    cnn = torch.load('torch_emnistcnn_checkpoint.pt', map_location=torch.device("cpu"))
    #cnn.cuda()
    test(cnn)