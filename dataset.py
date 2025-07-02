import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import transforms
import ast

class TrafficLightDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.img_dir, row['file'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(int(row['class']))
        points = torch.tensor(ast.literal_eval(row['points']), dtype=torch.float)
        return {'image': image, 'mode': label, 'points': points}

    def __len__(self):
        return len(self.data)