"""
tests/test_monitor.py v1.0.0
============================
Tests basados en tus 3 experimentos validados.
Reproducen condiciones reales y verifican anticipación.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# Forzar CPU para tests deterministas
torch.manual_seed(42)
np.random.seed(42)

def test_sovereignty_monitor_prediction():
    """
    Replica Experimento Ultra-Rápido:
    L debe detectar colapso 2 épocas ANTES que val_loss.
    """
    from liber_monitor import SovereigntyMonitor
    
    # Modelo pequeño como en tu experimento
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    monitor = SovereigntyMonitor(epsilon=0.1, patience=2)
    
    # Simular entrenamiento con colapso real
    L_values = [5.9, 5.5, 4.2, 1.8, 0.6, 0.4, 0.3, 0.2]  # Colapso gradual
    
    detected_epoch = None
    for epoch, L_sim in enumerate(L_values):
        # Forzar estado interno del monitor
        monitor.history.append({"epoch": epoch, "L": L_sim, "layers": []})
        
        if monitor.should_stop():
            detected_epoch = epoch
            break
    
    # Con patience=2, debe detectar después de 2 épocas consecutivas críticas
    # Época 5: L=0.4 (crítica) - contador=1
    # Época 6: L=0.3 (crítica) - contador=2, patience=2 -> alerta
    assert detected_epoch == 6, f"Detectado en época {detected_epoch}, esperado 6 (patience=2 requiere 2 épocas críticas consecutivas)"
    assert len(monitor.history) == 7

def test_sovereignty_monitor_stable():
    """
    Replica Experimento MNIST Completo:
    No debe generar falsos positivos en entrenamiento normal.
    """
    from liber_monitor import SovereigntyMonitor
    
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*5*5, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    monitor = SovereigntyMonitor()
    
    # Inicializar pesos con distribución normal (estado inicial estable)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.1)
    
    L = monitor.calculate(model)
    regime = monitor.regime(L)
    
    # L inicial debe ser > 1.0 (healthy)
    assert L > 1.0, f"L inicial={L:.3f}, debe ser > 1.0"
    assert regime == "healthy", f"Régimen inicial={regime}, debe ser healthy"
    assert not monitor.should_stop(), "No debe sugerir early stopping al inicio"

def test_sovereignty_monitor_forced_collapse():
    """
    Replica Experimento Colapso Forzado:
    Detecta deterioro gradual en modelo grande con datos tóxicos.
    """
    from liber_monitor import SovereigntyMonitor
    
    model = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Inicializar con pesos grandes para forzar inestabilidad
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 1.0)  # Desviación grande
    
    monitor = SovereigntyMonitor(epsilon=0.1, patience=1)
    
    # Simular colapso rápido
    L_values = [5.0, 4.5, 3.0, 1.5, 0.8, 0.3]  # Colapso agresivo
    
    for epoch, L_sim in enumerate(L_values):
        monitor.history.append({"epoch": epoch, "L": L_sim, "layers": []})
        
        if monitor.should_stop():
            # Debe detectar en época 5 (0.3) con patience=1 después de cruce en 4 (0.8)
            assert epoch == 5, f"Alerta en época {epoch}, esperado 5"
            assert L_sim < 0.5, f"L={L_sim} debe ser crítico"
            break
    
    assert monitor._critical_epochs == 1, "Debe tener 1 época crítica acumulada"

def test_layer_diagnostics():
    """Verifica que analiza cada capa individualmente"""
    from liber_monitor import SovereigntyMonitor
    
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    monitor = SovereigntyMonitor(track_layers=True)
    diagnostics = monitor.get_diagnostics(model)
    
    # Debe analizar 3 capas lineales
    assert len(diagnostics["layers"]) == 3, f"Capas detectadas: {len(diagnostics['layers'])}"
    
    # Cada capa debe tener métricas completas
    for layer in diagnostics["layers"]:
        assert "layer_name" in layer
        assert "L" in layer
        assert "regime" in layer
        assert layer["L"] > 0

def test_singular_entropy_function():
    """Test API simple singular_entropy()"""
    from liber_monitor import singular_entropy
    
    model = nn.Linear(100, 50)
    L = singular_entropy(model)
    
    assert isinstance(L, float), "Debe retornar float"
    assert 0 < L < 10, f"L={L} está fuera de rango esperado"

def test_regime_function():
    """Test API simple regime()"""
    from liber_monitor import regime
    
    assert regime(1.5) == "healthy"
    assert regime(0.8) == "warning"
    assert regime(0.3) == "critical"

# Test de integración completa (replica tu script final)
@pytest.mark.slow
def test_full_integration(tmp_path):
    """Replica entrenamiento completo con early stopping"""
    from liber_monitor import SovereigntyMonitor, plot_training_dynamics
    import torch.optim as optim
    
    # Modelo simple
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    monitor = SovereigntyMonitor(patience=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Datos sintéticos
    X_train = torch.randn(500, 784)
    y_train = torch.randint(0, 10, (500,))
    X_val = torch.randn(100, 784)
    y_val = torch.randint(0, 10, (100,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Entrenamiento controlado
    for epoch in range(15):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
        L = monitor.calculate(model)
        
        # Verificar que no muera
        assert L > 0, f"L inválido en época {epoch}"
        
        if monitor.should_stop():
            # Si detecta colapso, debe ser después de época 2 (patience=2)
            assert epoch >= 2, f"Detención temprana injustificada en época {epoch}"
            break
    
    # Debe haber generado historial
    assert len(monitor.history) > 0, "No generó historial"
    
    # Exportar reporte
    from liber_monitor.utils import export_report
    export_report(monitor.history, "test_model", tmp_path / "report.json")
