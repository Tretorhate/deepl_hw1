"""
BONUS: Transfer Learning
ResNet18-di CIFAR-10-ge daldendiru
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


def load_cifar10_for_pretrained():
    """Pretrained modelder ushin CIFAR-10"""
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, num_workers=0)
    
    return train_loader, test_loader


def create_resnet_model():
    """ResNet18 dayyndau"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Qabattardy tondyru
    for param in model.parameters():
        param.requires_grad = False
    
    # Songy qabatty auystyru
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    return model


def train_transfer(model, train_loader, test_loader, epochs=2, lr=0.001):
    """Transfer learning oqytu"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    history = {'train_loss': [], 'test_acc': []}
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}', end='\r')
        
        train_loss /= len(train_loader)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        test_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Test: {test_acc:.2f}%')
    
    elapsed = time.time() - start
    return history, elapsed


def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    print('\n' + '='*50)
    print('BONUS: ResNet18 Transfer Learning')
    print('='*50)
    
    train_loader, test_loader = load_cifar10_for_pretrained()
    model = create_resnet_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params: {total_params:,}')
    print(f'Trainable: {trainable_params:,}')
    print(f'Frozen: {total_params - trainable_params:,}')
    
    print('\nTraining (final layer only)...')
    history, elapsed = train_transfer(model, train_loader, test_loader, epochs=2)
    
    print(f'\nTest Accuracy: {history["test_acc"][-1]:.2f}%')
    print(f'Time: {elapsed:.1f}s')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Transfer Learning - Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['test_acc'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Transfer Learning - Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/bonus_transfer_learning.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print('\n=== Comparison ===')
    print('Transfer Learning: ~80-85% accuracy')
    print('Basic CNN: ~65-70% accuracy')
    print('\nWhy it works:')
    print('- ImageNet features generalize well')
    print('- ResNet knows edges, textures, shapes')
    print('- Only fine-tune the classifier')


if __name__ == '__main__':
    main()
