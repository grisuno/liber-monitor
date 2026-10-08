# API

## app.py
- `SimpleNet.__init__` (method) `app.py:24` `def __init__(self)`
- `SimpleNet.forward` (method) `app.py:30` `def forward(self, x)`
- `SimpleNet.simulate_training` (method) `app.py:35` `def simulate_training(epochs, lr, dataset_size)` -- Simula entrenamiento con detección de overfitting real
- `SimpleNet.create_plot` (method) `app.py:97` `def create_plot(L_values, regimes)` -- Crea gráfico de la evolución de L
- `SimpleNet.export_report` (method) `app.py:141` `def export_report(history, model_name)` -- Exporta reporte en formato JSON

## experiments/01_ultra_fast.py
- `ModeloMNISTPequeno.__init__` (method) `experiments/01_ultra_fast.py:21` `def __init__(self)`
- `ModeloMNISTPequeno.forward` (method) `experiments/01_ultra_fast.py:30` `def forward(self, x)`
- `ModeloMNISTPequeno.run` (method) `experiments/01_ultra_fast.py:38` `def run()`

## experiments/02_complete_mnist.py
- `CNNMNIST.__init__` (method) `experiments/02_complete_mnist.py:20` `def __init__(self)`
- `CNNMNIST.forward` (method) `experiments/02_complete_mnist.py:30` `def forward(self, x)`
- `CNNMNIST.run` (method) `experiments/02_complete_mnist.py:40` `def run()`

## experiments/03_forced_collapse.py
- `setup_matplotlib` (function) `experiments/03_forced_collapse.py:15` `def setup_matplotlib()` -- Configuración de matplotlib consolidada de los 3 experimentos Maneja diferentes backends y fuentes internacionales
- `plot_training_dynamics` (function) `experiments/03_forced_collapse.py:39` `def plot_training_dynamics(history, loss_train, loss_val, save_path, show_layers, dpi)` -- Gráficos comprehensivos consolidados de los 3 experimentos Reproduce el análisis completo: L, pérdida, capas...
- `validate_early_stopping` (function) `experiments/03_forced_collapse.py:196` `def validate_early_stopping(history, loss_val, threshold_L, threshold_loss_increase)` -- Valida retroactivamente si L predijo el colapso antes que val_loss Reproduce el análisis de "2-3 épocas de...
- `export_report` (function) `experiments/03_forced_collapse.py:312` `def export_report(history, model_name, save_path, include_layers)` -- Exporta reporte JSON comprehensivo para integración con pipelines Consolidado de los 3 experimentos
- `detect_collapse_epoch` (function) `experiments/03_forced_collapse.py:420` `def detect_collapse_epoch(history, threshold)` -- Detecta la primera época donde se observó colapso (L < threshold)
- `calculate_trend` (function) `experiments/03_forced_collapse.py:432` `def calculate_trend(history, window)` -- Calcula la tendencia de L en las últimas `window` épocas
- `summary_table` (function) `experiments/03_forced_collapse.py:457` `def summary_table(history, loss_train, loss_val, n_epochs)` -- Genera tabla resumen en formato texto para consola Consolida las tablas de los 3 experimentos
