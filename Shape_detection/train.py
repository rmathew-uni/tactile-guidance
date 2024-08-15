import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from emnist import extract_training_samples
import torch_cnn
import itertools
import matplotlib.pyplot as plt
import math

def train(num_epochs, cnn, train_loader, valid_loader, optimiser, loss_func, scheduler=None):
    '''Trains the CNN with given parameters'''
    valid_accuracy = []
    for epoch in range(num_epochs):
        cnn.train()
        for (images, labels) in train_loader:
            #images, labels = images.cuda(), labels.cuda()
            output = cnn.forward(images)
            loss = loss_func(output, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        if scheduler is not None:
            scheduler.step()

        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        valid_accuracy.append(valid(cnn, valid_loader))
    return valid_accuracy

def valid(cnn, valid_loader):
    '''Calculates the accuracy of the CNN on the validation data'''
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            # Check sizes of batches
            print(f"Batch size: {images.size(0)}")
            #images, labels = images.cuda(), labels.cuda()
            test_output = cnn.forward(images)
            _, predicted = torch.max(test_output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Debugging print statements
        print(f"Total samples processed: {total}")
        if total == 0:
            raise ValueError("No samples processed in validation. Check validation data and loader.")

    accuracy = 100*(correct / total)
    print('Validation Data Accuracy: {:.2f}'.format(accuracy))
    return accuracy

def train_plot_params(epochs, train_data, valid_data):
    '''Plots accuracy on validation data for all combinations of given parameters'''
    # Take subset of data to save time
    train_data = train_data[:50000]
    # Parameters
    fmaps1s = [40]
    fmaps2s = [160]
    denses = [200]
    dropouts = [0.4]
    batch_sizes = [5]

    # Train and save accuracies
    plots = []
    for params in list(itertools.product(fmaps1s, fmaps2s, denses, dropouts, batch_sizes)):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params[4], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=10000, shuffle=False)
        cnn = torch_cnn.EMNISTCNN(*params[:4])#.cuda()

        loss_func = nn.CrossEntropyLoss()
        optimiser = optim.SGD(cnn.parameters(), lr=0.02 * math.sqrt(params[4]))
        scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

        plots.append((train(epochs, cnn, train_loader, valid_loader, optimiser, loss_func, scheduler=scheduler), *params))
    
    # Plot accuracies
    for plot in plots:
        plt.plot(range(epochs), plot[0], label=str(plot[1:])[1:-1])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.title('Validation accuracy of fmaps1, fmaps2, dense, dropout, batch_size')
    plt.savefig('validation_plot.png')

def train_save(epochs, train_data, valid_data):
    '''Train the network and save it'''
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1000, shuffle=False)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(valid_data)}")

    # Initialize and move model to GPU
    cnn = torch_cnn.EMNISTCNN(40, 160, 200, 0.4)#.cuda()
    
    # Validate before training to check if validation data is correctly loaded
    valid(cnn, valid_loader)

    # Define loss function, optimizer, and scheduler
    loss_func = nn.CrossEntropyLoss()
    optimiser = optim.SGD(cnn.parameters(), lr=0.045)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.5)

    # Train the model
    train(epochs, cnn, train_loader, valid_loader, optimiser, loss_func, scheduler=scheduler)
    
    # Save the trained model
    torch.save(cnn.state_dict(), 'torch_emnistcnn_letters.pt')

if __name__ == '__main__':
    epochs = 60
    # Load EMNIST letters dataset
    images, labels = extract_training_samples('letters')
    print("Shape of images:", images.shape)

    # Convert labels to integers (0-25 for letters A-Z)
    labels = np.array(labels) - 1  # Adjust labels to be zero-indexed if needed
    
    # Ensure you split the data properly
    train_images = torch.tensor((images[:120000] / 255 - 0.5).reshape(-1, 1, 28, 28).astype(np.float32))
    train_labels = torch.tensor(labels[:120000], dtype=torch.int64)
    valid_images = torch.tensor((images[120000:] / 255 - 0.5).reshape(-1, 1, 28, 28).astype(np.float32))
    valid_labels = torch.tensor(labels[120000:], dtype=torch.int64)

    train_data = list(zip(train_images, train_labels))
    valid_data = list(zip(valid_images, valid_labels))

    train_save(epochs, train_data, valid_data)
    #train_plot_params(epochs, train_data, valid_data)
