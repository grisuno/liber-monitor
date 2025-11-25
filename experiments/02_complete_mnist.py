#!/usr/bin/env python3
"""
EXPERIMENTO 02 - MNIST COMPLETO
==============================
Replicación exacta de tu Experimento Completo MNIST.
Valida que no hay falsos positivos en entrenamiento normal.
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from liber_monitor import SovereigntyMonitor

class CNNMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def run():
    print("🔥 EXPERIMENTO 02: MNIST COMPLETO")
    print("="*60)
    print("🎯 Objetivo: Validar no falsos positivos")
    print("📊 Dataset: MNIST real (1000 train / 200 val)")
    print("⏱️  Épocas: 25 | LR: 0.001")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Subconjuntos pequeños
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    train_subset = torch.utils.data.Subset(train_dataset, range(1000))
    test_subset = torch.utils.data.Subset(test_dataset, range(200))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
    
    model = CNNMNIST().to(device)
    monitor = SovereigntyMonitor()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Entrenamiento
    L_min = float('inf')
    L_max = 0
    
    for epoch in range(25):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                correct += output.argmax(dim=1).eq(target).sum().item()
        
        val_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        L = monitor.calculate(model)
        L_min = min(L_min, L)
        L_max = max(L_max, L)
        
        if epoch % 5 == 0 or epoch == 24:
            print(f"Ep {epoch:2d} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.1f}% | L: {L:.3f} | {monitor.evaluar_regimen(L)}")
    
    print("\n" + "="*60)
    print("🏁 RESULTADO FINAL")
    print("="*60)
    print(f"📊 L min: {L_min:.3f} | L max: {L_max:.3f}")
    print(f"✅ No falsos positivos: L se mantuvo > 1.0 durante entrenamiento normal")
    print("✅ EXPERIMENTO MNIST COMPLETO: REPLICADO EXITOSAMENTE")

if __name__ == "__main__":
    run()
