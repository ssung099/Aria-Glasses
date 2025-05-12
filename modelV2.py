import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pathlib
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, cache_file=None):
        """
        Args:
            root_dir (string): Directory with all the spectrograms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir

        self.cache_file = cache_file
        self.data = []

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize all images to 64x64
            transforms.ToTensor(),       # Convert to tensor and normalize to [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize: mean=0.5, std=0.5
        ])

        for audio_source in self.root_dir.iterdir():
            # print(audio_source)
            for tensor_file in audio_source.iterdir():
                filename = tensor_file.name
                angle = int(filename.split('_tensor')[0])
                if angle < 0:
                    angle = angle + 360
                
                angle = angle / 360

                tensor_data = torch.load(tensor_file, weights_only=True)
                self.data.append((tensor_data, angle))
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram_tensor, angle = self.data[idx]
        return spectrogram_tensor, angle

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)  # Input channels: 14, Output: 32
        self.pool = nn.MaxPool2d(2, 2)  # Pooling halves dimensions
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Input: 32, Output: 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Input: 64, Output: 128
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjusted for 8x8 final feature map size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Output: 1 (angle prediction)

    def forward(self, x):
        x = x.squeeze(2)
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + Pool
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))  # Fully connected 1
        x = F.relu(self.fc2(x))  # Fully connected 2
        x = self.fc3(x)  # Output layer
        return x

def plot_cdf(errors):
    """Plot the CDF of the errors."""
    # Sort the errors
    sorted_errors = np.sort(errors)

    # Compute the CDF
    cdf = np.arange(1, len(errors) + 1) / len(errors)

    # Plot the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_errors, cdf, marker='.', linestyle='none')
    plt.title('CDF of Model Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data = pathlib.Path("Input")
    dataset = SpectrogramDataset(data)

    train_indices, test_indices = train_test_split(range(len(dataset.data)), test_size=0.2)
    train_subset = torch.utils.data.Subset(dataset.data, train_indices)
    test_subset = torch.utils.data.Subset(dataset.data, test_indices)

    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=True, num_workers=2)
    
    net = Net()
    net.load_state_dict(torch.load("model.pth", weights_only=True))

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()   

    num_epochs = 30
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in trainloader:
            labels = labels.float()
            optimizer.zero_grad()

            outputs = net(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
        # exit(0)
    print('Finished Training')

    errors = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            labels = labels.float()
            outputs = net(inputs).view(-1)
            predicted_angles = outputs
            total += labels.size(0)
            correct += (predicted_angles == labels).sum().item()
            
            # Multiply error by 360 to get the degree error
            degree_errors = np.abs(predicted_angles.numpy() - labels.numpy()) * 360
            errors.extend(degree_errors)
            # print(f"Prediction: {predicted_angles}\n Correct: {labels}")

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    torch.save(net.state_dict(), "model.pth")

    plot_cdf(errors)