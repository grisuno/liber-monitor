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
# Clonar repo
git clone https://github.com/grisuno/liber-monitor
cd liber-monitor

# Instalar en modo desarrollo
pip install -e .[experiments]

# Ejecutar experimentos
python experiments/01_ultra_fast.py
python experiments/02_complete_mnist.py
python experiments/03_forced_collapse.py

# Ejecutar tests
pytest tests/ -v
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

## Interpretación de Métricas
- Régimen Global (L promedio del modelo)
- L > 1.0: healthy → Generalización óptima, continuar entrenando
- 0.5 < L ≤ 1.0: warning → Transición crítica, monitorear de cerca
- L ≤ 0.5: critical → Colapso inminente, detener entrenamiento

Por Capa Individual

Cada capa lineal/convolucional es analizada independientemente, permitiendo identificar qué capa específica está colapsando primero.
Parámetros Calibrados Empíricamente

```python
monitor = SovereigntyMonitor(
    epsilon=0.1,          # Umbral de estabilidad (validado)
    patience=2,           # Épocas críticas consecutivas (2 = 2-3 épocas de anticipación)
    warning_threshold=0.5, # Punto de no retorno (validado)
    track_layers=True     # Monitoreo granular por capa
)
```

## Validación Retroactiva

```python
from liber_monitor.utils import validate_early_stopping

# Tu historial de entrenamiento
history = monitor.history
overfitting_epoch = 12  # Ground truth

validation = validate_early_stopping(history, overfitting_epoch)
# {'valid': True, 'anticipation_epochs': 2, ...}
```

## Exportar Reporte

```python
from liber_monitor.utils import export_report

export_report(monitor.history, "my_model", "report.json")
# Exporta JSON para pipelines de producción
```
## Resultados Esperados

### Experimento 01 (15 épocas)
- Val_loss colapsa en época 6
- L detecta en época 4 (2 épocas de anticipación) ✅

### Experimento 02 (25 épocas)
- L se mantiene entre 4.0-5.9 (SOBERANO)
- No falsos positivos ✅

### Experimento 03 (30 épocas)
- L colapsa en época 14 (crítico)
- Deterioro gradual detectado en época 8 ✅

## Licencia
GPL v3 - Usa, modifica, comparte libremente.
Si usas esta herramienta en investigación, cita el trabajo original:

```bibtex
@software{resma2025,
  title={RESMA: A Geometric Framework for Neural Network Monitoring},
  author={RESMA Project},
  url={https://github.com/grisuno/resma}
}
```


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
