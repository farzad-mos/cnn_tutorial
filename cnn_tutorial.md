# CNN Tutorial for ADAS

## What is a CNN?
A Convolutional Neural Network (CNN) is a class of deep neural networks, most commonly applied to analyzing visual imagery.

## ADAS Use Cases:
- Object Detection (e.g., pedestrians, vehicles)
- Lane Detection
- Traffic Sign Recognition

## CNN Architecture:
1. Convolutional Layers – Feature extraction using kernels
2. Activation Function – Typically ReLU
3. Pooling Layers – Dimensionality reduction
4. Fully Connected Layers – Classification or regression output

## Sample Python Code:
```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # e.g., 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```