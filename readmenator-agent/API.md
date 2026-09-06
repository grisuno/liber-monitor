# API

## app.py

### simulate_training `def simulate_training(epochs, lr, dataset_size)`
- Defined: `app.py:35`
- Doc: Simula entrenamiento con detección de overfitting real

### create_plot `def create_plot(L_values, regimes)`
- Defined: `app.py:97`
- Doc: Crea gráfico de la evolución de L

### export_report `def export_report(history, model_name)`
- Defined: `app.py:141`
- Doc: Exporta reporte en formato JSON

### __init__ `def __init__(self)`
- Defined: `app.py:24`

### forward `def forward(self, x)`
- Defined: `app.py:30`

## experiments/01_ultra_fast.py

### run `def run()`
- Defined: `experiments/01_ultra_fast.py:38`

### __init__ `def __init__(self)`
- Defined: `experiments/01_ultra_fast.py:21`

### forward `def forward(self, x)`
- Defined: `experiments/01_ultra_fast.py:30`

## experiments/02_complete_mnist.py

### run `def run()`
- Defined: `experiments/02_complete_mnist.py:40`

### __init__ `def __init__(self)`
- Defined: `experiments/02_complete_mnist.py:20`

### forward `def forward(self, x)`
- Defined: `experiments/02_complete_mnist.py:30`

## experiments/03_forced_collapse.py

### setup_matplotlib `def setup_matplotlib()`
- Defined: `experiments/03_forced_collapse.py:15`
- Doc: Configuración de matplotlib consolidada de los 3 experimentos

### plot_training_dynamics `def plot_training_dynamics(history, loss_train, loss_val, save_path, show_layers, dpi)`
- Defined: `experiments/03_forced_collapse.py:39`
- Doc: Gráficos comprehensivos consolidados de los 3 experimentos

### validate_early_stopping `def validate_early_stopping(history, loss_val, threshold_L, threshold_loss_increase)`
- Defined: `experiments/03_forced_collapse.py:196`
- Doc: Valida retroactivamente si L predijo el colapso antes que val_loss

### export_report `def export_report(history, model_name, save_path, include_layers)`
- Defined: `experiments/03_forced_collapse.py:312`
- Doc: Exporta reporte JSON comprehensivo para integración con pipelines

### detect_collapse_epoch `def detect_collapse_epoch(history, threshold)`
- Defined: `experiments/03_forced_collapse.py:420`
- Doc: Detecta la primera época donde se observó colapso (L < threshold)

### calculate_trend `def calculate_trend(history, window)`
- Defined: `experiments/03_forced_collapse.py:432`
- Doc: Calcula la tendencia de L en las últimas `window` épocas

### summary_table `def summary_table(history, loss_train, loss_val, n_epochs)`
- Defined: `experiments/03_forced_collapse.py:457`
- Doc: Genera tabla resumen en formato texto para consola

## tests/test_integration.py

### test_integration_ultra_fast_experiment `def test_integration_ultra_fast_experiment()`
- Defined: `tests/test_integration.py:16`
- Doc: REPLICA EXPERIMENTO ULTRA-RÁPIDO COMPLETO

### test_integration_complete_mnist `def test_integration_complete_mnist()`
- Defined: `tests/test_integration.py:115`
- Doc: REPLICA EXPERIMENTO COMPLETO MNIST

### test_integration_forced_collapse `def test_integration_forced_collapse()`
- Defined: `tests/test_integration.py:180`
- Doc: REPLICA EXPERIMENTO COLAPSO FORZADO

### test_pip_install_format `def test_pip_install_format()`
- Defined: `tests/test_integration.py:257`
- Doc: Valida que el paquete siga formato estándar de pip

### __init__ `def __init__(self)`
- Defined: `tests/test_integration.py:25`

### forward `def forward(self, x)`
- Defined: `tests/test_integration.py:34`

### __init__ `def __init__(self)`
- Defined: `tests/test_integration.py:124`

### forward `def forward(self, x)`
- Defined: `tests/test_integration.py:134`

### __init__ `def __init__(self)`
- Defined: `tests/test_integration.py:189`

### forward `def forward(self, x)`
- Defined: `tests/test_integration.py:197`

## tests/test_monitor.py

### test_sovereignty_monitor_prediction `def test_sovereignty_monitor_prediction()`
- Defined: `tests/test_monitor.py:17`
- Doc: Replica Experimento Ultra-Rápido:

### test_sovereignty_monitor_stable `def test_sovereignty_monitor_stable()`
- Defined: `tests/test_monitor.py:73`
- Doc: Replica Experimento MNIST Completo:

### test_sovereignty_monitor_forced_collapse `def test_sovereignty_monitor_forced_collapse()`
- Defined: `tests/test_monitor.py:109`
- Doc: Replica Experimento Colapso Forzado:

### test_layer_diagnostics `def test_layer_diagnostics()`
- Defined: `tests/test_monitor.py:168`
- Doc: Verifica que analiza cada capa individualmente

### test_singular_entropy_function `def test_singular_entropy_function()`
- Defined: `tests/test_monitor.py:193`
- Doc: Test API simple singular_entropy()

### test_regime_function `def test_regime_function()`
- Defined: `tests/test_monitor.py:203`
- Doc: Test API simple regime()

### test_full_integration `def test_full_integration(tmp_path)`
- Defined: `tests/test_monitor.py:213`
- Doc: Replica entrenamiento completo con early stopping
