import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from model import SiameseModel

def predict_similarity(input1, input2):
    # Load the model
    model = SiameseModel()
    model = model.double()
    model.eval()
    model_path = 'C:/Users/Felicia/Bracelet/tactile-guidance/Shape_detection/Siamese/modelSiamese.pth'

    # Check if model checkpoint exists
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print('Loaded model checkpoint from:', model_path)
    else:
        print('Model checkpoint not found at:', model_path)
        return

    # Assuming input1 and input2 are already tensors processed similarly to how your training data was processed
    output1, output2 = model(input1, input2)
    
    # Compute Euclidean distance
    euclidean_distance = F.pairwise_distance(output1, output2)

    print(f"Euclidean Distance between inputs: {euclidean_distance.item()}")

    # Plot the inputs and the distance
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(input1.squeeze().numpy(), cmap='gray')
    plt.title('Input 1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(input2.squeeze().numpy(), cmap='gray')
    plt.title('Input 2')
    plt.axis('off')

    plt.suptitle(f'Euclidean Distance: {euclidean_distance.item():.2f}')
    plt.show()

# Example of how to call this function:
# Assuming input1 and input2 are preprocessed images loaded into PyTorch tensors
input1 = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/9.jpg'
print(input1.shape) #harus jadi numpy dulu 
input2 = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/nine.jpg'
print(input2.shape)
#predict_similarity(input1, input2)
