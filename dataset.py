import torch
import os
import numpy
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128), antialias=False),
    transforms.Grayscale()
    ])

class ImageDataset(Dataset):
    def __init__(self, left, right, transform=None):
        self.transform = transform
        self.data = []
        self.left = left
        self.right = right

        for file in os.listdir(left):
            self.data.append("{left}/{file}".format(left=left, file=file))

        for file in os.listdir(right):
            self.data.append("{right}/{file}".format(right=right, file=file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = numpy.array(Image.open(self.data[idx]))
        if self.transform:
            image = self.transform(image)

        image = image.reshape(-1)

        label = torch.tensor([1., 0.], dtype=torch.float32)
        if self.right in self.data[idx]:
            label = torch.tensor([0., 1.], dtype=torch.float32)

        return image, label