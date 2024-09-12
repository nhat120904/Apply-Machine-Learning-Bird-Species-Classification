import torch
from model import BirdClassifier
from torchvision import transforms
from load_data import CUB200Dataset

def evaluate_model(model, test_loader, num_classes=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Calculate Top-1 accuracy
    top1_accuracy = 100. * correct / total
    print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
    
    # Calculate Average accuracy per class
    class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    avg_accuracy_per_class = sum(class_accuracies) / num_classes
    print(f'Average Accuracy per Class: {avg_accuracy_per_class:.2f}%')
    
    return top1_accuracy, avg_accuracy_per_class, class_accuracies


    


if __name__ == "__main__":
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = BirdClassifier(architecture="efficientnet_b0").to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('bird_classifier_efficientnet_b0.pth', weights_only=True))
    # evaluation
    top1_acc, avg_acc_per_class, class_accuracies = evaluate_model(model, test_loader)

    # print individual class accuracies
    for i, acc in enumerate(class_accuracies):
        print(f"Class {i} accuracy: {acc:.2f}%")