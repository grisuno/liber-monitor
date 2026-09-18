# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `SimpleNet` | class | `app.py:23` | `class SimpleNet(Module)` |
| `__init__` | method | `app.py:24` | `def __init__(self)` |
| `create_plot` | method | `app.py:97` | `def create_plot(L_values, regimes)` |
| `export_report` | method | `app.py:141` | `def export_report(history, model_name)` |
| `forward` | method | `app.py:30` | `def forward(self, x)` |
| `simulate_training` | method | `app.py:35` | `def simulate_training(epochs, lr, dataset_size)` |
| `ModeloMNISTPequeno` | class | `experiments/01_ultra_fast.py:20` | `class ModeloMNISTPequeno(Module)` |
| `__init__` | method | `experiments/01_ultra_fast.py:21` | `def __init__(self)` |
| `forward` | method | `experiments/01_ultra_fast.py:30` | `def forward(self, x)` |
| `run` | method | `experiments/01_ultra_fast.py:38` | `def run()` |
| `CNNMNIST` | class | `experiments/02_complete_mnist.py:19` | `class CNNMNIST(Module)` |
| `__init__` | method | `experiments/02_complete_mnist.py:20` | `def __init__(self)` |
| `forward` | method | `experiments/02_complete_mnist.py:30` | `def forward(self, x)` |
| `run` | method | `experiments/02_complete_mnist.py:40` | `def run()` |
| `calculate_trend` | function | `experiments/03_forced_collapse.py:432` | `def calculate_trend(history, window)` |
| `detect_collapse_epoch` | function | `experiments/03_forced_collapse.py:420` | `def detect_collapse_epoch(history, threshold)` |
| `export_report` | function | `experiments/03_forced_collapse.py:312` | `def export_report(history, model_name, save_path, include_layers)` |
| `plot_training_dynamics` | function | `experiments/03_forced_collapse.py:39` | `def plot_training_dynamics(history, loss_train, loss_val, save_path, show_layers, dpi)` |
| `setup_matplotlib` | function | `experiments/03_forced_collapse.py:15` | `def setup_matplotlib()` |
| `summary_table` | function | `experiments/03_forced_collapse.py:457` | `def summary_table(history, loss_train, loss_val, n_epochs)` |
| `validate_early_stopping` | function | `experiments/03_forced_collapse.py:196` | `def validate_early_stopping(history, loss_val, threshold_L, threshold_loss_increase)` |
| `CNNMNIST` | class | `tests/test_integration.py:123` | `class CNNMNIST(Module)` |
| `ModeloGrande` | class | `tests/test_integration.py:188` | `class ModeloGrande(Module)` |
| `ModeloMNISTPequeno` | class | `tests/test_integration.py:24` | `class ModeloMNISTPequeno(Module)` |
| `__init__` | method | `tests/test_integration.py:25` | `def __init__(self)` |
| `__init__` | method | `tests/test_integration.py:124` | `def __init__(self)` |
| `__init__` | method | `tests/test_integration.py:189` | `def __init__(self)` |
| `forward` | method | `tests/test_integration.py:34` | `def forward(self, x)` |
| `forward` | method | `tests/test_integration.py:134` | `def forward(self, x)` |
| `forward` | method | `tests/test_integration.py:197` | `def forward(self, x)` |
| `test_integration_complete_mnist` | function | `tests/test_integration.py:115` | `def test_integration_complete_mnist()` |
| `test_integration_forced_collapse` | function | `tests/test_integration.py:180` | `def test_integration_forced_collapse()` |
| `test_integration_ultra_fast_experiment` | function | `tests/test_integration.py:16` | `def test_integration_ultra_fast_experiment()` |
| `test_pip_install_format` | function | `tests/test_integration.py:257` | `def test_pip_install_format()` |
| `test_full_integration` | function | `tests/test_monitor.py:213` | `def test_full_integration(tmp_path)` |
| `test_layer_diagnostics` | function | `tests/test_monitor.py:168` | `def test_layer_diagnostics()` |
| `test_regime_function` | function | `tests/test_monitor.py:203` | `def test_regime_function()` |
| `test_singular_entropy_function` | function | `tests/test_monitor.py:193` | `def test_singular_entropy_function()` |
| `test_sovereignty_monitor_forced_collapse` | function | `tests/test_monitor.py:109` | `def test_sovereignty_monitor_forced_collapse()` |
| `test_sovereignty_monitor_prediction` | function | `tests/test_monitor.py:17` | `def test_sovereignty_monitor_prediction()` |
| `test_sovereignty_monitor_stable` | function | `tests/test_monitor.py:73` | `def test_sovereignty_monitor_stable()` |
