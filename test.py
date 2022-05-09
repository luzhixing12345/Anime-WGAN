import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="dataset",
    train=True,
    download=False,
    transform=ToTensor(),
)

a = DataLoader(training_data, batch_size=32, shuffle=True)
print(len(a))
