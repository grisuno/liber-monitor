# root

*Community 0 | 9 files | cohesion 1.00*

## Definition

This community groups 9 file(s) rooted at `experiments` with dominant language py (cohesion 1.00). Central symbols: `CNNMNIST`, `ModeloGrande`, `ModeloMNISTPequeno`, `SimpleNet`, `__init__`, `calculate_trend`, `create_plot`, `detect_collapse_epoch`. Core file: `tests/test_integration.py` (13 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisun0[at]proton[dot]me Fecha de creación: 22/11/2025 Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 6 | yes |
| `examples/quick_demo.py` | py | utility | 0 | yes |
| `experiments/01_ultra_fast.py` | py | utility | 4 | yes |
| `experiments/02_complete_mnist.py` | py | utility | 4 | yes |
| `experiments/03_forced_collapse.py` | py | utility | 7 | yes |
| `install.sh` | sh | utility | 0 | no |
| `setup.py` | py | infrastructure | 0 | yes |
| `tests/test_integration.py` | py | testing | 13 | yes |
| `tests/test_monitor.py` | py | testing | 7 | yes |

## Key Symbols

- `SimpleNet` (class, `app.py:23`) `class SimpleNet(Module)`
- `__init__` (method, `app.py:24`) `def __init__(self)`
- `forward` (method, `app.py:30`) `def forward(self, x)`
- `simulate_training` (method, `app.py:35`) `def simulate_training(epochs, lr, dataset_size)` - Simula entrenamiento con detección de overfitting real
- `create_plot` (method, `app.py:97`) `def create_plot(L_values, regimes)` - Crea gráfico de la evolución de L
- `export_report` (method, `app.py:141`) `def export_report(history, model_name)` - Exporta reporte en formato JSON
- `ModeloMNISTPequeno` (class, `experiments/01_ultra_fast.py:20`) `class ModeloMNISTPequeno(Module)`
- `__init__` (method, `experiments/01_ultra_fast.py:21`) `def __init__(self)`
- `forward` (method, `experiments/01_ultra_fast.py:30`) `def forward(self, x)`
- `run` (method, `experiments/01_ultra_fast.py:38`) `def run()`
- `CNNMNIST` (class, `experiments/02_complete_mnist.py:19`) `class CNNMNIST(Module)`
- `__init__` (method, `experiments/02_complete_mnist.py:20`) `def __init__(self)`
- `forward` (method, `experiments/02_complete_mnist.py:30`) `def forward(self, x)`
- `run` (method, `experiments/02_complete_mnist.py:40`) `def run()`
- `setup_matplotlib` (function, `experiments/03_forced_collapse.py:15`) `def setup_matplotlib()` - Configuración de matplotlib consolidada de los 3 experimentos
- `plot_training_dynamics` (function, `experiments/03_forced_collapse.py:39`) `def plot_training_dynamics(history, loss_train, loss_val, save_path, show_layers` - Gráficos comprehensivos consolidados de los 3 experimentos
- `validate_early_stopping` (function, `experiments/03_forced_collapse.py:196`) `def validate_early_stopping(history, loss_val, threshold_L, threshold_loss_incre` - Valida retroactivamente si L predijo el colapso antes que val_loss
- `export_report` (function, `experiments/03_forced_collapse.py:312`) `def export_report(history, model_name, save_path, include_layers)` - Exporta reporte JSON comprehensivo para integración con pipelines
- `detect_collapse_epoch` (function, `experiments/03_forced_collapse.py:420`) `def detect_collapse_epoch(history, threshold)` - Detecta la primera época donde se observó colapso (L < threshold)
- `calculate_trend` (function, `experiments/03_forced_collapse.py:432`) `def calculate_trend(history, window)` - Calcula la tendencia de L en las últimas `window` épocas
- `summary_table` (function, `experiments/03_forced_collapse.py:457`) `def summary_table(history, loss_train, loss_val, n_epochs)` - Genera tabla resumen en formato texto para consola
- `test_integration_ultra_fast_experiment` (function, `tests/test_integration.py:16`) `def test_integration_ultra_fast_experiment()` - REPLICA EXPERIMENTO ULTRA-RÁPIDO COMPLETO
- `ModeloMNISTPequeno` (class, `tests/test_integration.py:24`) `class ModeloMNISTPequeno(Module)`
- `__init__` (method, `tests/test_integration.py:25`) `def __init__(self)`
- `forward` (method, `tests/test_integration.py:34`) `def forward(self, x)`
- `test_integration_complete_mnist` (function, `tests/test_integration.py:115`) `def test_integration_complete_mnist()` - REPLICA EXPERIMENTO COMPLETO MNIST
- `CNNMNIST` (class, `tests/test_integration.py:123`) `class CNNMNIST(Module)`
- `__init__` (method, `tests/test_integration.py:124`) `def __init__(self)`
- `forward` (method, `tests/test_integration.py:134`) `def forward(self, x)`
- `test_integration_forced_collapse` (function, `tests/test_integration.py:180`) `def test_integration_forced_collapse()` - REPLICA EXPERIMENTO COLAPSO FORZADO

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `examples/quick_demo.py`
- `experiments/01_ultra_fast.py`
- `experiments/02_complete_mnist.py`
- `experiments/03_forced_collapse.py`
- `install.sh`
- `setup.py`
- `tests/test_integration.py`
- `tests/test_monitor.py`
