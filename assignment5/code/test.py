import torch
import torch.nn as nn 
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import shutil
from PIL import Image
import os


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 75 * 75, 128)  
        self.dropout = nn.Dropout(0.5)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # Forward pass through layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.dropout(x) 
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    

class TestCNN:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.testdata = datasets.ImageFolder(root='code/testdata', transform=self.transform)

        # Load model and initialize it
        self.model = CNN()  # Create an instance of CNN model
        self.model.load_state_dict(torch.load('code/model.pth'), strict=False)
        self.model.eval()  # Set model to evaluation mode

        self.criterion = nn.CrossEntropyLoss()

    def test(self):
        total_correct = 0
        total_samples = 0

        test_loader = DataLoader(self.testdata, batch_size=1, shuffle=False)

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = 100 * total_correct / total_samples
        return accuracy

    def main(self):
        # Call test method to get accuracy
        test_accuracy = self.test()
        print(f'Test Accuracy: {test_accuracy:.2f}%')

if __name__ == "__main__":
    tester = TestCNN()
    tester.main()