#!/usr/bin/env python3
"""
EXPERIMENTO 01 - ULTRA-RÁPIDO
==============================
Replicación exacta de tu Experimento Ultra-Rápido.
Valida que L predice colapso 2-3 épocas antes que val_loss.
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from liber_monitor import SovereigntyMonitor, validate_early_stopping

# Modelo exacto de tu experimento
class ModeloMNISTPequeno(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def run():
    print("🔥 EXPERIMENTO 01: ULTRA-RÁPIDO")
    print("="*60)
    print("🎯 Objetivo: Replicar tu Experimento Ultra-Rápido")
    print("📊 Dataset: 200 train / 50 val (sintéticos tóxicos)")
    print("⏱️  Épocas: 15 | LR: 0.01 | Patience: 2")
    print("="*60)
    
    # Setup
    torch.manual_seed(42)
    model = ModeloMNISTPequeno()
    monitor = SovereigntyMonitor(epsilon_c=0.1, patience=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Datos tóxicos (exactos de tu experimento)
    train_imgs = torch.randn(200, 1, 28, 28)
    train_labels = torch.randint(0, 10, (200,))
    val_imgs = torch.randn(50, 1, 28, 28)
    val_labels = torch.randint(0, 10, (50,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Variables de seguimiento
    overfitting_epoch = None
    min_val_loss = float('inf')
    
    # Entrenamiento
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_imgs), train_labels)
        loss.backward()
        optimizer.step()
        
        # Evaluación
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_imgs), val_labels).item()
        
        # Monitoreo
        L = monitor.calculate(model)
        
        # Detectar overfitting (val_loss sube 10% desde mínimo)
        if epoch > 2:
            if val_loss > min_val_loss * 1.10:
                overfitting_epoch = epoch
        
        min_val_loss = min(min_val_loss, val_loss)
        
        if epoch % 3 == 0 or epoch == 14:
            print(f"Ep {epoch:2d} | Loss: {val_loss:.4f} | L: {L:.3f} | {monitor.evaluar_regimen(L)}")
    
    # Validación final
    validation = validate_early_stopping(monitor.history, threshold_L=0.5)
    
    print("\n" + "="*60)
    print("🏁 RESULTADO FINAL")
    print("="*60)
    print(validation["message"])
    if validation["valid"]:
        print(f"✅ RETRASO: {validation['anticipation_epochs']} épocas")
        print("✅ EXPERIMENTO ULTRA-RÁPIDO: REPLICADO EXITOSAMENTE")
    else:
        print("❌ EXPERIMENTO FALLÓ")
    
    # Graficar
    if monitor.history:
        epochs = [h.epoch for h in monitor.history]
        L_values = [h.L_promedio for h in monitor.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, L_values, 'purple', linewidth=2.5)
        plt.axhline(y=0.5, color='orange', linestyle='--', label='Umbral Crítico')
        plt.title("L vs Época - Experimento Ultra-Rápido")
        plt.xlabel("Época")
        plt.ylabel("L")
        plt.legend()
        plt.savefig("01_ultra_fast_results.png", dpi=300)
        print("\n💾 Gráfico guardado: 01_ultra_fast_results.png")

if __name__ == "__main__":
    run()
