import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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
        
        if self.transform:
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
    
def extract_features(dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features = []
    labels = []
    
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_features = feature_extractor(batch_images).squeeze()
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    return np.concatenate(features), np.concatenate(labels)
    
if __name__ == '__main__':
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    root_dir = 'dataset'
    train_dataset = CUB200Dataset(root_dir, split='Train', transform=train_transform)
    test_dataset = CUB200Dataset(root_dir, split='Test', transform=test_transform)
    
    # Load pre-trained ResNet50 model
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the last fully connected layer
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()  # Set to evaluation mode

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = feature_extractor.to(device)

    # Extract features from train and test datasets
    print("Extracting features from training set...")
    X_train, y_train = extract_features(train_dataset)
    print("Extracting features from test set...")
    X_test, y_test = extract_features(test_dataset)

    # Train an SVM classifier
    print("Training RF classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = rf_classifier.predict(X_test)
    y_train_pred = rf_classifier.predict(X_train)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"RF Accuracy: {accuracy:.4f}")
    print(f"RF Train Accuracy: {train_accuracy:.4f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(classification_report(y_train, y_train_pred))

    # Calculate average accuracy per class
    class_accuracies = classification_report(y_test, y_pred, output_dict=True)
    avg_accuracy_per_class = np.mean([class_accuracies[str(i)]['f1-score'] for i in range(200)])
    print(f"\nAverage Accuracy per Class: {avg_accuracy_per_class:.4f}")




