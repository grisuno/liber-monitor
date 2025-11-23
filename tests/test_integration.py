"""
tests/test_integration.py v1.0.0
==================================
Tests de integración completa que replican tus experimentos originales.
Verifican que liber-monitor funciona exactamente como RESMA.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tempfile
import json

def test_integration_ultra_fast_experiment():
    """
    REPLICA EXPERIMENTO ULTRA-RÁPIDO COMPLETO
    Objetivo: Validar que L predice colapso 2 épocas antes
    """
    from liber_monitor import SovereigntyMonitor, validate_early_stopping
    
    # Modelo CNN pequeño (exacto de tu experimento)
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
    
    model = ModeloMNISTPequeno()
    monitor = SovereigntyMonitor(epsilon=0.1, patience=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # LR alto para colapso rápido
    
    # Datos sintéticos tóxicos (exactos de tu experimento)
    torch.manual_seed(42)
    train_imgs = torch.randn(200, 1, 28, 28)  # 200 muestras, muy pequeño
    train_labels = torch.randint(0, 10, (200,))
    val_imgs = torch.randn(50, 1, 28, 28)
    val_labels = torch.randint(0, 10, (50,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Entrenamiento ultra-rápido (15 épocas)
    overfitting_detected = False
    overfitting_epoch = None
    
    for epoch in range(15):
        # Entrenamiento agresivo (todo el batch)
        model.train()
        optimizer.zero_grad()
        outputs = model(train_imgs)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        # Evaluación
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_imgs)
            val_loss = criterion(val_outputs, val_labels).item()
        
        # Monitoreo L
        L = monitor.calculate(model)
        
        # Detección de overfitting (val_loss sube 10% desde mínimo)
        if epoch > 2 and not overfitting_detected:
            if val_loss > min([h.get("val_loss", float('inf')) for h in monitor.history[-3:]]) * 1.10:
                overfitting_detected = True
                overfitting_epoch = epoch
        
        # Guardar historial completo para análisis
        if epoch == 0:
            min_val_loss = val_loss
        else:
            min_val_loss = min(min_val_loss, val_loss)
        
        monitor.history[-1]["val_loss"] = val_loss
        monitor.history[-1]["min_val_loss"] = min_val_loss
    
    # Validar poder predictivo
    validation = validate_early_stopping(monitor.history, overfitting_epoch, threshold=0.5)
    
    # ASSERT PRINCIPAL: Debe predecir al menos 1 época antes
    assert validation["valid"], f"Test falló: {validation['message']}"
    assert validation["anticipation_epochs"] >= 1, f"Anticipación insuficiente: {validation['anticipation_epochs']}"
    
    print(f"✅ Experimento Ultra-Rápido: {validation['message']}")

def test_integration_complete_mnist():
    """
    REPLICA EXPERIMENTO COMPLETO MNIST
    Objetivo: Validar que no genera falsos positivos en entrenamiento normal
    """
    from liber_monitor import SovereigntyMonitor
    
    # Modelo CNN completo (exacto de tu experimento)
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
    
    model = CNNMNIST()
    monitor = SovereigntyMonitor()
    
    # Inicializar pesos con distribución normal
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.1)
    
    # Calcular L inicial
    L_initial = monitor.calculate(model)
    
    # ASSERT: L inicial debe ser healthy
    assert L_initial > 1.0, f"L inicial {L_initial:.3f} debe ser > 1.0"
    assert monitor.regime(L_initial) == "healthy"
    
    # Simular 5 épocas de entrenamiento normal
    L_values = []
    for epoch in range(5):
        # Perturbación simulada
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        L = monitor.calculate(model)
        L_values.append(L)
        
        # Nunca debe sugerir early stopping en entrenamiento normal
        assert not monitor.should_stop(), f"Falso positivo en época {epoch}"
    
    # L debe mantenerse > 0.5
    min_L = min(L_values)
    assert min_L > 0.5, f"L cayó a {min_L:.3f} en entrenamiento normal"
    
    print(f"✅ Experimento MNIST Completo: Sin falsos positivos (L min={min_L:.3f})")

def test_integration_forced_collapse():
    """
    REPLICA EXPERIMENTO COLAPSO FORZADO
    Objetivo: Validar sensibilidad en condiciones extremas
    """
    from liber_monitor import SovereigntyMonitor
    
    # Modelo grande (exacto de tu experimento)
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
    
    model = ModeloGrande()
    
    # Inicializar con pesos extremos para forzar inestabilidad
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 1.0)  # Desviación muy grande
    
    monitor = SovereigntyMonitor(epsilon=0.1, patience=1)
    
    # Calcular L
    L = monitor.calculate(model)
    regime = monitor.regime(L)
    
    # Debe detectar inestabilidad (warning o critical)
    assert regime in ["warning", "critical"], f"Régimen {regime} no refleja inestabilidad"
    
    # Simular colapso rápido
    monitor.history = [{"epoch": i, "L": L, "layers": []} for i, L in enumerate([4.0, 3.0, 1.5, 0.8, 0.3])]
    
    # Debe activar early stopping
    assert monitor.should_stop(), "No detectó colapso en condiciones extremas"
    
    print(f"✅ Experimento Colapso Forzado: Detectó inestabilidad (L={L:.3f}, {regime})")

def test_pip_install_format():
    """Valida que el paquete siga formato estándar de pip"""
    import liber_monitor
    
    # Debe tener versión
    assert hasattr(liber_monitor, "__version__"), "No tiene __version__"
    assert hasattr(liber_monitor, "__author__"), "No tiene __author__"
    assert hasattr(liber_monitor, "SovereigntyMonitor"), "No exporta SovereigntyMonitor"
    assert hasattr(liber_monitor, "singular_entropy"), "No exporta singular_entropy"
    
    print(f"✅ Formato pip install: v{liber_monitor.__version__}")

if __name__ == "__main__":
    print("🧪 EJECUTANDO TESTS DE INTEGRACIÓN COMPLETA")
    print("="*60)
    
    test_integration_ultra_fast_experiment()
    test_integration_complete_mnist()
    test_integration_forced_collapse()
    test_pip_install_format()
    
    print("\n" + "="*60)
    print("🎉 TODOS LOS TESTS PASARON")
    print("✅ liber-monitor funciona como RESMA")
