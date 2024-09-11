import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from load_data import CUB200Dataset
from model import BirdClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFilter
from sklearn.model_selection import train_test_split


class GaussianBlur(object):
    def __init__(self, radius):
        self.radius = radius
    
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

# More aggressive data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    GaussianBlur(radius=2),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test transform remains the same
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root_dir = 'dataset'
train_dataset = CUB200Dataset(root_dir, split='Train', transform=train_transform)
test_dataset = CUB200Dataset(root_dir, split='Test', transform=test_transform)

# Assuming your original train_dataset is the full training set
train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)

train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=16, 
    sampler=train_subsampler,
    num_workers=4
)

val_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=16, 
    sampler=val_subsampler,
    num_workers=4
)
# Create the model
model = BirdClassifier(architecture='efficientnet_b7').to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20):
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    model.to(device)
    
    # Create the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update the learning rate
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")
    
    writer.close()
    print("Training completed")
    
    # # Save the trained model
    # torch.save(model.state_dict(), 'bird_classifier_efficientnet_b0.pth')
    # print("Model saved successfully.")

if __name__ == "__main__":
    # Assuming you have separate train and validation loaders
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=30)