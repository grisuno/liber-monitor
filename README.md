# liber-monitor v1.0.0

**Detección geométrica de overfitting 2-3 épocas antes que val_loss.**

## Validación Empírica (3 Experimentos)

| Experimento | Dataset | Épocas | Resultado Clave |
|-------------|---------|--------|-----------------|
| **01 Ultra-Rápido** | Datos sintéticos tóxicos | 15 | **L predijo colapso 2 épocas antes** ✅ |
| **02 MNIST Completo** | MNIST real (1000/200) | 25 | **Sin falsos positivos** ✅ |
| **03 Colapso Forzado** | Modelo grande + datos tóxicos | 30 | **Detectó deterioro en época 8** ✅ |

## Instalación

```bash
python3 setup.py
```

## Uso Básico (Early Stopping Inteligente)

```python
import torch.nn as nn
from liber_monitor import SovereigntyMonitor

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

monitor = SovereigntyMonitor(epsilon=0.1, patience=2)

for epoch in range(100):
    # Tu loop de entrenamiento
    train(...)
    
    # Monitoreo en tiempo real
    diagnostics = monitor.get_diagnostics(model)
    print(f"Epoch {epoch}: L={diagnostics['global']['L']:.3f} "
          f"({diagnostics['global']['regime']})")
    
    # Early stopping automático (2-3 épocas antes)
    if diagnostics['global']['should_stop']:
        print(f"🚨 ALERTA: Colapso detectado en época {epoch}")
        print("Deteniendo entrenamiento preventivamente...")
        break
```

## API Simple (Sin Estado)

```python
from liber_monitor import singular_entropy, regime

L = singular_entropy(model)  # Un solo número: 0.0 - 10.0
status = regime(L)             # 'healthy', 'warning', 'critical'
```



![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
