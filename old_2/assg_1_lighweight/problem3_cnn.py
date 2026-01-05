"""
3-Esep: Konvolyucialyk zhuyke zheleleri
A, B, C bolimderi - CIFAR-10 CNN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

torch.manual_seed(42)
np.random.seed(42)

# GPU tekseru
print("=" * 60)
print("GPU Diagnostics:")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')

CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_cifar10():
    """CIFAR-10 zhukteu"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=100, num_workers=0)
    
    return train_loader, test_loader


class BasicCNN(nn.Module):
    """Negizgi CNN arhitekturasy"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_cnn(model, train_loader, test_loader, epochs=2, lr=0.001):
    """CNN oqytu"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'test_acc': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        val_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:2d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_acc:.2f}%')
    
    elapsed = time.time() - start_time
    return history, elapsed


def get_predictions(model, loader):
    """Boljamdарды alu"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            preds = model(data).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, classes):
    """Shatasu matricasy"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/part3a_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


class CNN_V1(nn.Module):
    """Kishi CNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class CNN_V2(nn.Module):
    """Teren CNN - 3 conv qabat"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class CNN_V3(nn.Module):
    """AvgPool CNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def visualize_filters(model):
    """Birinshi conv filtrleri"""
    weights = model.conv1.weight.data.cpu()
    weights = weights - weights.min()
    weights = weights / weights.max()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            filt = weights[i].permute(1, 2, 0).numpy()
            ax.imshow(filt)
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
    
    plt.suptitle('Conv1 Filters (8/32)')
    plt.tight_layout()
    plt.savefig('results/part3c_filters.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_activations(model, loader):
    """Aktivacia kartalary"""
    model.eval()
    
    data, labels = next(iter(loader))
    data = data[:3].to(device)
    
    activations = []
    def hook(module, input, output):
        activations.append(output.detach().cpu())
    
    handle = model.conv1.register_forward_hook(hook)
    _ = model(data)
    handle.remove()
    
    acts = activations[0]
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    
    for row in range(3):
        img = data[row].cpu()
        img = img * std.view(3, 1, 1) + mean.view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'Original: {CIFAR_CLASSES[labels[row]]}')
        axes[row, 0].axis('off')
        
        for i in range(4):
            act_map = acts[row, i].numpy()
            axes[row, i+1].imshow(act_map, cmap='viridis')
            axes[row, i+1].set_title(f'Filter {i+1}')
            axes[row, i+1].axis('off')
    
    plt.suptitle('Conv1 Activation Maps')
    plt.tight_layout()
    plt.savefig('results/part3c_activations.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    train_loader, test_loader = load_cifar10()
    
    # A bolimi
    print('\n' + '='*50)
    print('PART A: Basic CNN')
    print('='*50)
    
    model = BasicCNN()
    print(f'\nArchitecture:\n{model}')
    print(f'\nParameters: {count_params(model):,}')
    
    history, train_time = train_cnn(model, train_loader, test_loader, epochs=2)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part3a_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'\nTest Accuracy: {history["test_acc"][-1]:.2f}%')
    
    preds, labels = get_predictions(model, test_loader)
    plot_confusion_matrix(preds, labels, CIFAR_CLASSES)
    
    # B bolimi
    print('\n' + '='*50)
    print('PART B: Architecture Experiments')
    print('='*50)
    
    architectures = {
        'V1 (Small)': CNN_V1(),
        'V2 (Deep)': CNN_V2(),
        'V3 (AvgPool)': CNN_V3(),
        'Basic': BasicCNN()
    }
    
    arch_results = {}
    
    for name, arch_model in architectures.items():
        print(f'\n--- {name} ---')
        print(f'Parameters: {count_params(arch_model):,}')
        hist, t = train_cnn(arch_model, train_loader, test_loader, epochs=2)
        arch_results[name] = {
            'params': count_params(arch_model),
            'time': t,
            'acc': hist['test_acc'][-1],
            'history': hist
        }
    
    print('\n' + '='*60)
    print(f'{"Architecture":<15} {"Params":<12} {"Time (s)":<12} {"Accuracy":<10}')
    print('-'*49)
    for name, res in arch_results.items():
        print(f'{name:<15} {res["params"]:<12,} {res["time"]:<12.1f} {res["acc"]:.2f}%')
    print('='*60)
    
    plt.figure(figsize=(10, 5))
    for name, res in arch_results.items():
        plt.plot(res['history']['test_acc'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Architecture Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/part3b_architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # C bolimi
    print('\n' + '='*50)
    print('PART C: Visualization')
    print('='*50)
    
    visualize_filters(model)
    visualize_activations(model, test_loader)
    
    print('\nPart 3 results saved to results/ folder')


if __name__ == '__main__':
    main()
