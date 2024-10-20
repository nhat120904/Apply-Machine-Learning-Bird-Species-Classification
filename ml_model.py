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
from load_data import CUB200Dataset

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
feature_extractor.eval() 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = feature_extractor.to(device)
def extract_features(dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    features = []
    labels = []
    
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_features = feature_extractor(batch_images).squeeze()
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    return np.concatenate(features), np.concatenate(labels)
def compute_average_accuracy_per_class(y_true, y_pred, num_classes=200):
    class_report = classification_report(y_true, y_pred, output_dict=True)
    accuracies_per_class = [class_report[str(i)]['recall'] for i in range(num_classes) if str(i) in class_report]
    return np.mean(accuracies_per_class)

if __name__ == '__main__':
    
    # Extract features from train and test datasets
    X_train, y_train = extract_features(train_dataset)
    X_test, y_test = extract_features(test_dataset)

    # Initialize a dictionary to store results
    model_results = {}
    # Random Forest Classifier
    print("Training RF classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    print("Making predictions for RF...")
    y_pred_rf = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_avg_accuracy_per_class = compute_average_accuracy_per_class(y_test, y_pred_rf)

    # Store results for Random Forest
    model_results['Random Forest'] = {'Top-1 Accuracy': rf_accuracy, 'Average Accuracy per Class': rf_avg_accuracy_per_class}
    
    # SVM Classifier
    print("Training SVM classifier...")
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)

    print("Making predictions for SVM...")
    y_pred_svm = svm_classifier.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_avg_accuracy_per_class = compute_average_accuracy_per_class(y_test, y_pred_svm)

    # Store results for SVM
    model_results['SVM'] = {'Top-1 Accuracy': svm_accuracy, 'Average Accuracy per Class': svm_avg_accuracy_per_class}
    from sklearn.tree import DecisionTreeClassifier

    # Decision Tree Classifier
    print("Training Decision Tree classifier...")
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    print("Making predictions for Decision Tree...")
    y_pred_dt = dt_classifier.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    dt_avg_accuracy_per_class = compute_average_accuracy_per_class(y_test, y_pred_dt)

    # Store results for Decision Tree
    model_results['Decision Tree'] = {'Top-1 Accuracy': dt_accuracy, 'Average Accuracy per Class': dt_avg_accuracy_per_class}
    results_df = pd.DataFrame(model_results).T

    print(results_df)