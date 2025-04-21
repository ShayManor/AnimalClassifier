import os

import torch
from PIL import Image
from pandas import read_csv
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    def __init__(self, data, image_dir, transform):
        self.df = read_csv(data)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df['path'][idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor((self.df['label'][idx]), dtype=torch.long)
        return image, label

