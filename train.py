import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from load_data import CUB200Dataset
from model import BirdClassifier
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFilter
from sklearn.model_selection import train_test_split
from bird_detect import detect_birds_in_image

class BirdCropTransform:
    def __init__(self):
        pass
 
    def __call__(self, image):
        # bird detection and cropping function
        bird_cropped_image = detect_birds_in_image(image, conf_threshold=0.0)
        if bird_cropped_image == []:
            return image
        return bird_cropped_image

# Data augmentation
train_transform = transforms.Compose([
    BirdCropTransform(),
    transforms.Resize((256, 256)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.RandomGrayscale(p=0.1), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
 
test_transform = transforms.Compose([
    BirdCropTransform(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
 
root_dir = 'dataset'
 
# Load datasets with appropriate transforms
train_dataset = CUB200Dataset(root_dir, split='Train', transform=train_transform)
val_dataset = CUB200Dataset(root_dir, split='Test', transform=test_transform)  # Validation uses test transform
 
# Create data loaders for train and validation
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle = True,
    num_workers=4
)
 
# Validation loader uses val_dataset (with test transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=4
)
 
model = BirdClassifier(architecture='efficientnet_b5').to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
 
# # Training function
def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
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
            torch.save(model.state_dict(), './checkpoints/effb5_nhat.pth')
            print("Saved best model")
   
    writer.close()
    print("Training completed")

if __name__ == "__main__":
    # training function
    train_model(model, criterion, optimizer, train_loader, num_epochs=25)