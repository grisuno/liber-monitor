# Subsystem: experiments

## experiments/01_ultra_fast.py
- Layer: utility
- Language: py
- Symbols:
  - `ModeloMNISTPequeno` (class, line 20) `class ModeloMNISTPequeno(Module)`
  - `run` (method, line 38) `def run()`
  - `__init__` (method, line 21) `def __init__(self)`
  - `forward` (method, line 30) `def forward(self, x)`

## experiments/02_complete_mnist.py
- Layer: utility
- Language: py
- Symbols:
  - `CNNMNIST` (class, line 19) `class CNNMNIST(Module)`
  - `run` (method, line 40) `def run()`
  - `__init__` (method, line 20) `def __init__(self)`
  - `forward` (method, line 30) `def forward(self, x)`

## experiments/03_forced_collapse.py
- Layer: utility
- Language: py
- Symbols:
  - `setup_matplotlib` (function, line 15) `def setup_matplotlib()`
  - `plot_training_dynamics` (function, line 39) `def plot_training_dynamics(history, loss_train, loss_val, save_path, show_layers, dpi)`
  - `validate_early_stopping` (function, line 196) `def validate_early_stopping(history, loss_val, threshold_L, threshold_loss_increase)`
  - `export_report` (function, line 312) `def export_report(history, model_name, save_path, include_layers)`
  - `detect_collapse_epoch` (function, line 420) `def detect_collapse_epoch(history, threshold)`
  - `calculate_trend` (function, line 432) `def calculate_trend(history, window)`
  - `summary_table` (function, line 457) `def summary_table(history, loss_train, loss_val, n_epochs)`
