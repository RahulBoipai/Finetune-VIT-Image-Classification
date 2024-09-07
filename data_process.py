import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from tqdm.auto import tqdm


class CreateDataset(Dataset):
    """Create a dataset for the given data"""
    def __init__(self, root_dir, transform=None, train=True, label_id=None, id_label=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_id = label_id if label_id else {}
        self.id_label = id_label if id_label else {}

        suffix = '_train' if train else '_test'
        print(root_dir, suffix)
        i=0
        for label in tqdm(os.listdir(root_dir)):

            if not label.endswith(suffix):
                continue
            print(label)
            label_name = label.split('_')[0]
            #create a mapping for the label
            if train:
                self.id_label[i] = label_name
                self.label_id[label_name] = i
                i+=1
                
            label_idx = self.label_id[label_name]
            label_dir = os.path.join(root_dir, label)
            

            for image_name in os.listdir(label_dir):
                if image_name.endswith('.JPEG'):
                    image_path = os.path.join(label_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(label_idx)
                    
    def get_mapping(self):
        return self.label_id, self.id_label
                    
    def label_map(self, label):
        return self.label_id[label]

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