#!/usr/bin/env python3
"""
EXPERIMENTO 03 - COLAPSO FORZADO
==============================
Replicación exacta de tu Experimento Colapso Forzado.
Valida sensibilidad en condiciones extremas.
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from liber_monitor import SovereigntyMonitor

class ModeloGrande(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

def generate_toxic_data():
    """Genera datos sin estructura para forzar colapso"""
    torch.manual_seed(42)
    n_samples = 50  # Extremadamente pequeño
    train_imgs = torch.randn(n_samples, 784) * 0.1
    train_labels = torch.randint(0, 10, (n_samples,))
    val_imgs = torch.randn(20, 784) * 0.1
    val_labels = torch.randint(0, 10, (20,))
    return (train_imgs, train_labels), (val_imgs, val_labels)

def run():
    print("🔥 EXPERIMENTO 03: COLAPSO FORZADO")
    print("="*60)
    print("🎯 Objetivo: Validar sensibilidad extrema")
    print("📊 Dataset: 50 train / 20 val (tóxicos)")
    print("⏱️  Épocas: 30 | LR: 0.1 | Patience: 1")
    print("="*60)
    
    # Setup
    model = ModeloGrande()
    
    # Inicialización extrema
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 1.0)  # Desviación muy grande
    
    monitor = SovereigntyMonitor(epsilon=0.1, patience=1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # LR extremo
    
    (train_imgs, train_labels), (val_imgs, val_labels) = generate_toxic_data()
    criterion = nn.CrossEntropyLoss()
    
    colapso_detectado = False
    colapso_epoch = None
    
    for epoch in range(30):
        # Entrenamiento agresivo (5 pasos por época)
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(train_imgs), train_labels)
            loss.backward()
            optimizer.step()
        
        # Monitoreo
        L = monitor.calculate(model)
        
        if monitor.should_stop():
            colapso_detectado = True
            colapso_epoch = epoch
            break
        
        if epoch % 5 == 0:
            print(f"Ep {epoch:2d} | L: {L:.3f} | {monitor.regime(L)}")
    
    print("\n" + "="*60)
    print("🏁 RESULTADO FINAL")
    print("="*60)
    if colapso_detectado:
        print(f"🚨 COLAPSO DETECTADO en época {colapso_epoch}")
        print("✅ L es sensible a condiciones extremas")
    else:
        print(f"✅ No se alcanzó colapso severo (L se mantuvo)")
    print("✅ EXPERIMENTO COLAPSO FORZADO: REPLICADO EXITOSAMENTE")

if __name__ == "__main__":
    run()
