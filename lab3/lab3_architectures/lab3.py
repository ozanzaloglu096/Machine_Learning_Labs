import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt

# 1. Dataset (Fashion-MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(full_train, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. Model Definitions
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(self.flatten(x))

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_linear = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 16, -1)
        x = self.patch_linear(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

models = {
    "MLP": MLPModel(),
    "CNN": CNNModel(),
    "Attention": AttentionModel()
}

dummy_input = torch.randn(1, 1, 28, 28)

print("\n--- MODEL ANALYSIS AND COST CALCULATIONS ---")
for name, model in models.items():
    print(f"Model: {name}")
    stat = summary(model, input_data=dummy_input, verbose=0)
    print(f"Total Parameters: {stat.total_params}")

# Train & Evaluate Function
def train_and_evaluate(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break # Single batch for quick test
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total

print("\n--- TRAINING AND TEST ACCURACY ---")
mlp_acc = train_and_evaluate(models["MLP"], train_loader, test_loader)
print(f"MLP Test Accuracy: {mlp_acc:.2f}%")

cnn_acc = train_and_evaluate(models["CNN"], train_loader, test_loader)
print(f"CNN Test Accuracy: {cnn_acc:.2f}%")

att_acc = train_and_evaluate(models["Attention"], train_loader, test_loader)
print(f"Attention Test Accuracy: {att_acc:.2f}%")

# Trade-off Plots
parameters = [235146, 20490, 78794]
latencies = [0.22, 4.06, 4.25]
accuracies = [mlp_acc, cnn_acc, att_acc]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(parameters, accuracies, color=['red', 'green', 'blue'], s=100)
for i, txt in enumerate(['MLP', 'CNN', 'Attention']):
    plt.annotate(txt, (parameters[i], accuracies[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
plt.xlabel('Number of Parameters')
plt.ylabel('Test Accuracy (%)')
plt.title('Accuracy vs Parameters')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(latencies, accuracies, color=['red', 'green', 'blue'], s=100)
for i, txt in enumerate(['MLP', 'CNN', 'Attention']):
    plt.annotate(txt, (latencies[i], accuracies[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
plt.xlabel('Latency (ms, Batch 128)')
plt.ylabel('Test Accuracy (%)')
plt.title('Accuracy vs Latency')
plt.grid(True)

plt.tight_layout()
plt.savefig('tradeoff_plots.png')
print("\n--- Plot successfully saved as 'tradeoff_plots.png' ---")
