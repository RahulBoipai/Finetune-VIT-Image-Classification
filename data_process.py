import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


class CreateDataset(Dataset):
    """Create a dataset for the given data"""
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {}
        self.id_label = {}

        suffix = '_train' if train else '_test'
        print(root_dir, suffix)
        i=0
        for label in os.listdir(root_dir):

            if not label.endswith(suffix):
                continue
            print(label)
            label_name = label.split('_')[0]
            self.id_label[i] = label_name
            self.label_map[label_name] = i
            label_idx = self.label_map[label_name]
            label_dir = os.path.join(root_dir, label)
            i+=1

            for image_name in os.listdir(label_dir):
                if image_name.endswith('.JPEG'):
                    image_path = os.path.join(label_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(label_idx)
                    
    def label_map(self, label):
        return self.label_map[label]

    def id_map(self, id):
        return self.id_label[id]

    def __len__(self):
            return len(self.images)

    def __getitem__(self, idx):
            image_path = self.images[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label