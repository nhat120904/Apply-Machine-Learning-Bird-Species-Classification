import torch
from model import BirdClassifier
from torchvision import transforms
from load_data import CUB200Dataset
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
 
    test_transform = transforms.Compose([
        BirdCropTransform(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    root_dir = 'dataset'
    test_dataset = CUB200Dataset(root_dir, split='Test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
 
    model = BirdClassifier(architecture="efficientnet_b5").to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./checkpoints/effb5_nhat.pth', weights_only=True))
    # evaluate model
    top1_acc, avg_acc_per_class, class_accuracies = evaluate_model(model, test_loader)
 
    # print individual class accuracies
    # for i, acc in enumerate(class_accuracies):
    #     print(f"Class {i} accuracy: {acc:.2f}%")