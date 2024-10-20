import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self.default_transform()

        # Load metadata
        self.data = self.load_metadata()
        self.create_class_mapping()
        
    def load_metadata(self):
        split_file = os.path.join(self.root_dir, f'{self.split}.txt')
        data = pd.read_csv(split_file, sep=' ', names=['filename', 'label'])
        data['filepath'] = data['filename'].apply(lambda x: os.path.join(self.root_dir + '/' + self.split, self.split, x))
        data['class_name'] = data['filename'].apply(self.extract_class_name)
        return data

    def extract_class_name(self, filename):
        # Extract class name from filename
        parts = filename.split('_')
        return ' '.join(parts[:-2])
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')
        
        # Ensure the image is a PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.transform:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = self.transform(image)
        
        label = self.data.iloc[idx]['label']
        
        return image, label
    
    def create_class_mapping(self):
        class_mapping = {}
        for _, row in self.data.iterrows():
            label = row['label']
            class_name = row['class_name']
            if label not in class_mapping:
                class_mapping[label] = class_name
        self.class_mapping = class_mapping

    def get_class_name(self, class_idx):
        return self.class_mapping[class_idx]

    def get_class_idx(self, class_name):
        return self.class_mapping.index(class_name)

