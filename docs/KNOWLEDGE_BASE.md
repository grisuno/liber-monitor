# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 12 | **Total Symbols Extracted:** 71 | **Total Imports:** 77

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    tests_test_monitor_py["test_monitor.py (py)"]
    class tests_test_monitor_py mod;
    tests_test_monitor_py_test_sovereignty_monitor_prediction["test_sovereignty_monitor_prediction"]
    class tests_test_monitor_py_test_sovereignty_monitor_prediction fn;
    tests_test_monitor_py --> tests_test_monitor_py_test_sovereignty_monitor_prediction
    tests_test_monitor_py_test_sovereignty_monitor_stable["test_sovereignty_monitor_stable"]
    class tests_test_monitor_py_test_sovereignty_monitor_stable fn;
    tests_test_monitor_py --> tests_test_monitor_py_test_sovereignty_monitor_stable
    tests_test_monitor_py_test_sovereignty_monitor_forced_collapse["test_sovereignty_monitor_forced_collapse"]
    class tests_test_monitor_py_test_sovereignty_monitor_forced_collapse fn;
    tests_test_monitor_py --> tests_test_monitor_py_test_sovereignty_monitor_forced_collapse
    tests_test_monitor_py_test_layer_diagnostics["test_layer_diagnostics"]
    class tests_test_monitor_py_test_layer_diagnostics fn;
    tests_test_monitor_py --> tests_test_monitor_py_test_layer_diagnostics
    tests_test_monitor_py_test_singular_entropy_function["test_singular_entropy_function"]
    class tests_test_monitor_py_test_singular_entropy_function fn;
    tests_test_monitor_py --> tests_test_monitor_py_test_singular_entropy_function
    tests_test_integration_py["test_integration.py (py)"]
    class tests_test_integration_py mod;
    tests_test_integration_py_test_integration_ultra_fast_experiment["test_integration_ultra_fast_experiment"]
    class tests_test_integration_py_test_integration_ultra_fast_experiment fn;
    tests_test_integration_py --> tests_test_integration_py_test_integration_ultra_fast_experiment
    tests_test_integration_py_test_integration_complete_mnist["test_integration_complete_mnist"]
    class tests_test_integration_py_test_integration_complete_mnist fn;
    tests_test_integration_py --> tests_test_integration_py_test_integration_complete_mnist
    tests_test_integration_py_test_integration_forced_collapse["test_integration_forced_collapse"]
    class tests_test_integration_py_test_integration_forced_collapse fn;
    tests_test_integration_py --> tests_test_integration_py_test_integration_forced_collapse
    tests_test_integration_py_test_pip_install_format["test_pip_install_format"]
    class tests_test_integration_py_test_pip_install_format fn;
    tests_test_integration_py --> tests_test_integration_py_test_pip_install_format
    tests_test_integration_py_ModeloMNISTPequeno["ModeloMNISTPequeno"]
    class tests_test_integration_py_ModeloMNISTPequeno cls;
    tests_test_integration_py --> tests_test_integration_py_ModeloMNISTPequeno
    app_py["app.py (py)"]
    class app_py mod;
    app_py_SimpleNet["SimpleNet"]
    class app_py_SimpleNet cls;
    app_py --> app_py_SimpleNet
    app_py_simulate_training["simulate_training"]
    class app_py_simulate_training fn;
    app_py --> app_py_simulate_training
    app_py_create_plot["create_plot"]
    class app_py_create_plot fn;
    app_py --> app_py_create_plot
    app_py_export_report["export_report"]
    class app_py_export_report fn;
    app_py --> app_py_export_report
    app_py___init__["__init__"]
    class app_py___init__ fn;
    app_py --> app_py___init__
    liber_monitor_monitor_py["monitor.py (py)"]
    class liber_monitor_monitor_py mod;
    liber_monitor_monitor_py_Regime["Regime"]
    class liber_monitor_monitor_py_Regime cls;
    liber_monitor_monitor_py --> liber_monitor_monitor_py_Regime
    liber_monitor_monitor_py_LayerDiagnostics["LayerDiagnostics"]
    class liber_monitor_monitor_py_LayerDiagnostics cls;
    liber_monitor_monitor_py --> liber_monitor_monitor_py_LayerDiagnostics
    liber_monitor_monitor_py_EpochSnapshot["EpochSnapshot"]
    class liber_monitor_monitor_py_EpochSnapshot cls;
    liber_monitor_monitor_py --> liber_monitor_monitor_py_EpochSnapshot
    liber_monitor_monitor_py_SovereigntyMonitor["SovereigntyMonitor"]
    class liber_monitor_monitor_py_SovereigntyMonitor cls;
    liber_monitor_monitor_py --> liber_monitor_monitor_py_SovereigntyMonitor
    liber_monitor_monitor_py_singular_entropy["singular_entropy"]
    class liber_monitor_monitor_py_singular_entropy fn;
    liber_monitor_monitor_py --> liber_monitor_monitor_py_singular_entropy
    experiments_01_ultra_fast_py["01_ultra_fast.py (py)"]
    class experiments_01_ultra_fast_py mod;
    experiments_01_ultra_fast_py_ModeloMNISTPequeno["ModeloMNISTPequeno"]
    class experiments_01_ultra_fast_py_ModeloMNISTPequeno cls;
    experiments_01_ultra_fast_py --> experiments_01_ultra_fast_py_ModeloMNISTPequeno
    experiments_01_ultra_fast_py_run["run"]
    class experiments_01_ultra_fast_py_run fn;
    experiments_01_ultra_fast_py --> experiments_01_ultra_fast_py_run
    experiments_01_ultra_fast_py___init__["__init__"]
    class experiments_01_ultra_fast_py___init__ fn;
    experiments_01_ultra_fast_py --> experiments_01_ultra_fast_py___init__
    experiments_01_ultra_fast_py_forward["forward"]
    class experiments_01_ultra_fast_py_forward fn;
    experiments_01_ultra_fast_py --> experiments_01_ultra_fast_py_forward
    experiments_02_complete_mnist_py["02_complete_mnist.py (py)"]
    class experiments_02_complete_mnist_py mod;
    experiments_02_complete_mnist_py_CNNMNIST["CNNMNIST"]
    class experiments_02_complete_mnist_py_CNNMNIST cls;
    experiments_02_complete_mnist_py --> experiments_02_complete_mnist_py_CNNMNIST
    experiments_02_complete_mnist_py_run["run"]
    class experiments_02_complete_mnist_py_run fn;
    experiments_02_complete_mnist_py --> experiments_02_complete_mnist_py_run
    experiments_02_complete_mnist_py___init__["__init__"]
    class experiments_02_complete_mnist_py___init__ fn;
    experiments_02_complete_mnist_py --> experiments_02_complete_mnist_py___init__
    experiments_02_complete_mnist_py_forward["forward"]
    class experiments_02_complete_mnist_py_forward fn;
    experiments_02_complete_mnist_py --> experiments_02_complete_mnist_py_forward
    experiments_03_forced_collapse_py["03_forced_collapse.py (py)"]
    class experiments_03_forced_collapse_py mod;
    experiments_03_forced_collapse_py_setup_matplotlib["setup_matplotlib"]
    class experiments_03_forced_collapse_py_setup_matplotlib fn;
    experiments_03_forced_collapse_py --> experiments_03_forced_collapse_py_setup_matplotlib
    experiments_03_forced_collapse_py_plot_training_dynamics["plot_training_dynamics"]
    class experiments_03_forced_collapse_py_plot_training_dynamics fn;
    experiments_03_forced_collapse_py --> experiments_03_forced_collapse_py_plot_training_dynamics
    experiments_03_forced_collapse_py_validate_early_stopping["validate_early_stopping"]
    class experiments_03_forced_collapse_py_validate_early_stopping fn;
    experiments_03_forced_collapse_py --> experiments_03_forced_collapse_py_validate_early_stopping
    experiments_03_forced_collapse_py_export_report["export_report"]
    class experiments_03_forced_collapse_py_export_report fn;
    experiments_03_forced_collapse_py --> experiments_03_forced_collapse_py_export_report
    experiments_03_forced_collapse_py_detect_collapse_epoch["detect_collapse_epoch"]
    class experiments_03_forced_collapse_py_detect_collapse_epoch fn;
    experiments_03_forced_collapse_py --> experiments_03_forced_collapse_py_detect_collapse_epoch
    liber_monitor_utils_py["utils.py (py)"]
    class liber_monitor_utils_py mod;
    liber_monitor_utils_py_setup_matplotlib["setup_matplotlib"]
    class liber_monitor_utils_py_setup_matplotlib fn;
    liber_monitor_utils_py --> liber_monitor_utils_py_setup_matplotlib
    liber_monitor_utils_py_plot_training_dynamics["plot_training_dynamics"]
    class liber_monitor_utils_py_plot_training_dynamics fn;
    liber_monitor_utils_py --> liber_monitor_utils_py_plot_training_dynamics
    liber_monitor_utils_py_validate_early_stopping["validate_early_stopping"]
    class liber_monitor_utils_py_validate_early_stopping fn;
    liber_monitor_utils_py --> liber_monitor_utils_py_validate_early_stopping
    liber_monitor_utils_py_export_report["export_report"]
    class liber_monitor_utils_py_export_report fn;
    liber_monitor_utils_py --> liber_monitor_utils_py_export_report
    liber_monitor_utils_py_detect_collapse_epoch["detect_collapse_epoch"]
    class liber_monitor_utils_py_detect_collapse_epoch fn;
    liber_monitor_utils_py --> liber_monitor_utils_py_detect_collapse_epoch
    liber_monitor___init___py["__init__.py (py)"]
    class liber_monitor___init___py mod;
    liber_monitor___init___py_validate_early_stopping["validate_early_stopping"]
    class liber_monitor___init___py_validate_early_stopping fn;
    liber_monitor___init___py --> liber_monitor___init___py_validate_early_stopping
    liber_monitor___init___py_export_report["export_report"]
    class liber_monitor___init___py_export_report fn;
    liber_monitor___init___py --> liber_monitor___init___py_export_report
    liber_monitor___init___py_plot_training_dynamics["plot_training_dynamics"]
    class liber_monitor___init___py_plot_training_dynamics fn;
    liber_monitor___init___py --> liber_monitor___init___py_plot_training_dynamics
    examples_quick_demo_py["quick_demo.py (py)"]
    class examples_quick_demo_py mod;
    setup_py["setup.py (py)"]
    class setup_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_gradio["gradio"]
    class ext_gradio ext;
    app_py -.->|imports| ext_gradio
    ext_torch["torch"]
    class ext_torch ext;
    app_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    app_py -.->|imports| ext_torch_nn
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    app_py -.->|imports| ext_torch_optim
    ext_numpy["numpy"]
    class ext_numpy ext;
    app_py -.->|imports| ext_numpy
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    app_py -.->|imports| ext_matplotlib_pyplot
    ext_liber_monitor["liber_monitor"]
    class ext_liber_monitor ext;
    app_py -.->|imports| ext_liber_monitor
    ext_json["json"]
    class ext_json ext;
    app_py -.->|imports| ext_json
    examples_quick_demo_py -.->|imports| ext_torch
    examples_quick_demo_py -.->|imports| ext_liber_monitor
    ext_sys["sys"]
    class ext_sys ext;
    experiments_01_ultra_fast_py -.->|imports| ext_sys
    experiments_01_ultra_fast_py -.->|imports| ext_torch
    experiments_01_ultra_fast_py -.->|imports| ext_torch_nn
    experiments_01_ultra_fast_py -.->|imports| ext_torch_optim
    experiments_01_ultra_fast_py -.->|imports| ext_numpy
    experiments_01_ultra_fast_py -.->|imports| ext_matplotlib_pyplot
    experiments_01_ultra_fast_py -.->|imports| ext_liber_monitor
    experiments_02_complete_mnist_py -.->|imports| ext_sys
    experiments_02_complete_mnist_py -.->|imports| ext_torch
    experiments_02_complete_mnist_py -.->|imports| ext_torch_nn
    experiments_02_complete_mnist_py -.->|imports| ext_torch_optim
    experiments_02_complete_mnist_py -.->|imports| ext_numpy
    ext_torchvision["torchvision"]
    class ext_torchvision ext;
    experiments_02_complete_mnist_py -.->|imports| ext_torchvision
    experiments_02_complete_mnist_py -.->|imports| ext_liber_monitor
    experiments_03_forced_collapse_py -.->|imports| ext_matplotlib_pyplot
    experiments_03_forced_collapse_py -.->|imports| ext_numpy
    ext_typing["typing"]
    class ext_typing ext;
    experiments_03_forced_collapse_py -.->|imports| ext_typing
    ext_warnings["warnings"]
    class ext_warnings ext;
    experiments_03_forced_collapse_py -.->|imports| ext_warnings
    experiments_03_forced_collapse_py -.->|imports| ext_json
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    experiments_03_forced_collapse_py -.->|imports| ext_pathlib
    ext_monitor["monitor"]
    class ext_monitor ext;
    liber_monitor___init___py -.->|imports| ext_monitor
    ext_utils["utils"]
    class ext_utils ext;
    liber_monitor___init___py -.->|imports| ext_utils
    liber_monitor___init___py -.->|imports| ext_json
    liber_monitor_monitor_py -.->|imports| ext_torch
    liber_monitor_monitor_py -.->|imports| ext_numpy
    liber_monitor_monitor_py -.->|imports| ext_typing
    liber_monitor_monitor_py -.->|imports| ext_warnings
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    liber_monitor_monitor_py -.->|imports| ext_dataclasses
    ext_enum["enum"]
    class ext_enum ext;
    liber_monitor_monitor_py -.->|imports| ext_enum
    ext_scipy_sparse_linalg["scipy.sparse.linalg"]
    class ext_scipy_sparse_linalg ext;
    liber_monitor_monitor_py -.->|imports| ext_scipy_sparse_linalg
    liber_monitor_utils_py -.->|imports| ext_matplotlib_pyplot
    liber_monitor_utils_py -.->|imports| ext_numpy
    liber_monitor_utils_py -.->|imports| ext_typing
    liber_monitor_utils_py -.->|imports| ext_warnings
    liber_monitor_utils_py -.->|imports| ext_json
    liber_monitor_utils_py -.->|imports| ext_pathlib
    ext_setuptools["setuptools"]
    class ext_setuptools ext;
    setup_py -.->|imports| ext_setuptools
    ext_pytest["pytest"]
    class ext_pytest ext;
    tests_test_integration_py -.->|imports| ext_pytest
    tests_test_integration_py -.->|imports| ext_torch
    tests_test_integration_py -.->|imports| ext_torch_nn
    tests_test_integration_py -.->|imports| ext_torch_optim
    tests_test_integration_py -.->|imports| ext_numpy
    ext_tempfile["tempfile"]
    class ext_tempfile ext;
    tests_test_integration_py -.->|imports| ext_tempfile
    tests_test_integration_py -.->|imports| ext_json
    tests_test_integration_py -.->|imports| ext_liber_monitor
    tests_test_integration_py -.->|imports| ext_liber_monitor
    tests_test_integration_py -.->|imports| ext_liber_monitor
    tests_test_integration_py -.->|imports| ext_liber_monitor
    tests_test_integration_py -.->|imports| ext_liber_monitor
    ext_liber_monitor_monitor["liber_monitor.monitor"]
    class ext_liber_monitor_monitor ext;
    tests_test_integration_py -.->|imports| ext_liber_monitor_monitor
    tests_test_integration_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_pytest
    tests_test_monitor_py -.->|imports| ext_torch
    tests_test_monitor_py -.->|imports| ext_torch_nn
    tests_test_monitor_py -.->|imports| ext_numpy
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_liber_monitor
    tests_test_monitor_py -.->|imports| ext_torch_optim
    tests_test_monitor_py -.->|imports| ext_liber_monitor_monitor
```

---

## Architecture Reference

### PY (11 files)

#### `app.py`
**Path:** `app.py`

**Classs:**
- `SimpleNet` (line 23)

**Functions:**
- `simulate_training` (line 35) - *Simula entrenamiento con detección de overfitting real*
- `create_plot` (line 97) - *Crea gráfico de la evolución de L*
- `export_report` (line 141) - *Exporta reporte en formato JSON*
- `__init__` (line 24)
- `forward` (line 30)

#### `quick_demo.py`
**Path:** `examples/quick_demo.py`

*No symbols extracted*

#### `01_ultra_fast.py`
**Path:** `experiments/01_ultra_fast.py`

**Classs:**
- `ModeloMNISTPequeno` (line 20)

**Functions:**
- `run` (line 38)
- `__init__` (line 21)
- `forward` (line 30)

#### `02_complete_mnist.py`
**Path:** `experiments/02_complete_mnist.py`

**Classs:**
- `CNNMNIST` (line 19)

**Functions:**
- `run` (line 40)
- `__init__` (line 20)
- `forward` (line 30)

#### `03_forced_collapse.py`
**Path:** `experiments/03_forced_collapse.py`

**Functions:**
- `setup_matplotlib` (line 15) - *Configuración de matplotlib consolidada de los 3 experimentos
Maneja diferentes backends y fuentes internacionales*
- `plot_training_dynamics` (line 39) - *Gráficos comprehensivos consolidados de los 3 experimentos
Reproduce el análisis completo: L, pérdida, capas, correlaciones

Args:
    history: Lista de snapshots de época (monitor.history)
    loss_train: Pérdidas de entrenamiento (opcional)
    loss_val: Pérdidas de validación (opcional)
    save_path: Ruta para guardar el gráfico
    show_layers: True para mostrar L por capa individual
    dpi: Resolución del gráfico*
- `validate_early_stopping` (line 196) - *Valida retroactivamente si L predijo el colapso antes que val_loss
Reproduce el análisis de "2-3 épocas de anticipación" de los experimentos

Args:
    history: Historial de L calculado (monitor.history)
    loss_val: Pérdidas de validación (opcional)
    threshold_L: L < 0.5 = colapso (validado)
    threshold_loss_increase: Aumento % de val_loss para declarar overfitting

Returns:
    Dict con análisis completo de poder predictivo*
- `export_report` (line 312) - *Exporta reporte JSON comprehensivo para integración con pipelines
Consolidado de los 3 experimentos

Args:
    history: Historial completo (monitor.history)
    model_name: Nombre del modelo para identificación
    save_path: Ruta para guardar JSON
    include_layers: True para incluir diagnósticos detallados por capa

Returns:
    Dict con el reporte completo*
- `detect_collapse_epoch` (line 420) - *Detecta la primera época donde se observó colapso (L < threshold)

Returns:
    int: Época del colapso, o None si no hubo colapso*
- `calculate_trend` (line 432) - *Calcula la tendencia de L en las últimas `window` épocas

Returns:
    str: "ascending", "descending", "stable", o "insufficient_data"*
- `summary_table` (line 457) - *Genera tabla resumen en formato texto para consola
Consolida las tablas de los 3 experimentos

Returns:
    str: Tabla formateada para impresión*

#### `__init__.py`
**Path:** `liber_monitor/__init__.py`

**Functions:**
- `validate_early_stopping` (line 23) - *Validación retroactiva de early stopping*
- `export_report` (line 27) - *Exportar historial de monitoreo como JSON*
- `plot_training_dynamics` (line 33) - *Gráficos de dinámica de entrenamiento*

#### `monitor.py`
**Path:** `liber_monitor/monitor.py`

**Classs:**
- `Regime` (line 24) - *Regímenes validados empíricamente*
- `LayerDiagnostics` (line 31) - *Diagnóstico detallado por capa individual*
- `EpochSnapshot` (line 44) - *Snapshot completo de una época*
- `SovereigntyMonitor` (line 63) - *Monitor de Soberanía para Redes Neuronales - Versión Consolidada
Detecta colapso 2-3 épocas ANTES que val_loss (validado empíricamente)

Uso Básico:
    monitor = SovereigntyMonitor()
    for epoch in range(100):
        train_model(...)
        L = monitor.calculate(model)
        
        if monitor.should_stop():
            print(f"⚠️ Colapso detectado en época {epoch}")
            break

Uso Avanzado con Diagnóstico Completo:
    monitor = SovereigntyMonitor(track_layers=True, patience=2)
    diagnostics = monitor.get_diagnostics(model)
    print(diagnostics)*

**Functions:**
- `singular_entropy` (line 427) - *API simple: un solo número L, sin mantener estado

Uso:
    L = singular_entropy(model)
    if L < 0.5:
        print("⚠️ Modelo en riesgo")*
- `regime` (line 439) - *API simple: clasificación de régimen sin estado

Uso:
    L = 0.7
    reg = regime(L)
    print(f"Régimen: {reg}")  # "emergente"*
- `quick_check` (line 457) - *API simple: diagnóstico rápido sin tracking

Uso:
    status = quick_check(model)
    print(status["message"])*
- `to_dict` (line 40)
- `to_dict` (line 53)
- `__init__` (line 84) - *Args:
    epsilon_c: Umbral de estabilidad (0.1 validado en 3 experimentos)
    patience: Épocas consecutivas críticas antes de early stopping (2 validado)
    umbral_soberano: L > 1.0 = régimen soberano (validado)
    umbral_espurio: L < 0.5 = colapso inminente (validado)
    track_layers: True para monitorear cada capa individualmente
    verbose: True para imprimir diagnósticos detallados*
- `_extract_weights` (line 122) - *Extrae pesos de capas lineales y convolucionales
Consolidado de los 3 experimentos*
- `_calculate_svd_metrics` (line 134) - *Calcula S_vN y rango efectivo usando SVD robusto
Consolidado: fallbacks del experimento extremo + reshaping del rápido*
- `calcular_libertad` (line 193) - *Calcula la métrica L (libertad) de una matriz de pesos
Fórmula RESMA validada: L = 1 / (|S_vN - log(rank + 1)| + ε_c)

Returns:
    tuple: (L, S_vn, rank_effective)*
- `evaluar_regimen` (line 217) - *Evalúa el régimen del modelo basado en umbrales validados
Consolidado de los 3 experimentos*
- `calculate_layer_metrics` (line 229) - *Calcula métricas L por cada capa
Consolidado del experimento completo + extremo*
- `calculate` (line 263) - *Calcula L promedio del modelo completo
Este es el valor principal para early stopping

Returns:
    float: L promedio de todas las capas*
- `should_stop` (line 305) - *Early stopping inteligente con lógica de patience
Validado: predice colapso 2-3 épocas antes que val_loss

Args:
    L: Valor L actual (si None, usa el último calculado)

Returns:
    bool: True si se debe detener el entrenamiento*
- `get_diagnostics` (line 350) - *Reporte completo para debugging, logging y análisis
Consolidado de los 3 experimentos*
- `get_layer_trends` (line 402) - *Retorna tendencias históricas por capa
Útil para análisis post-entrenamiento*
- `reset` (line 411) - *Reinicia el monitor (útil para múltiples entrenamientos)*

#### `utils.py`
**Path:** `liber_monitor/utils.py`

**Functions:**
- `setup_matplotlib` (line 15) - *Configuración de matplotlib consolidada de los 3 experimentos
Maneja diferentes backends y fuentes internacionales*
- `plot_training_dynamics` (line 39) - *Gráficos comprehensivos consolidados de los 3 experimentos
Reproduce el análisis completo: L, pérdida, capas, correlaciones

Args:
    history: Lista de snapshots de época (monitor.history)
    loss_train: Pérdidas de entrenamiento (opcional)
    loss_val: Pérdidas de validación (opcional)
    save_path: Ruta para guardar el gráfico
    show_layers: True para mostrar L por capa individual
    dpi: Resolución del gráfico*
- `validate_early_stopping` (line 196) - *Valida retroactivamente si L predijo el colapso antes que val_loss
Reproduce el análisis de "2-3 épocas de anticipación" de los experimentos

Args:
    history: Historial de L calculado (monitor.history)
    loss_val: Pérdidas de validación (opcional)
    threshold_L: L < 0.5 = colapso (validado)
    threshold_loss_increase: Aumento % de val_loss para declarar overfitting

Returns:
    Dict con análisis completo de poder predictivo*
- `export_report` (line 312) - *Exporta reporte JSON comprehensivo para integración con pipelines
Consolidado de los 3 experimentos

Args:
    history: Historial completo (monitor.history)
    model_name: Nombre del modelo para identificación
    save_path: Ruta para guardar JSON
    include_layers: True para incluir diagnósticos detallados por capa

Returns:
    Dict con el reporte completo*
- `detect_collapse_epoch` (line 420) - *Detecta la primera época donde se observó colapso (L < threshold)

Returns:
    int: Época del colapso, o None si no hubo colapso*
- `calculate_trend` (line 432) - *Calcula la tendencia de L en las últimas `window` épocas

Returns:
    str: "ascending", "descending", "stable", o "insufficient_data"*
- `summary_table` (line 457) - *Genera tabla resumen en formato texto para consola
Consolida las tablas de los 3 experimentos

Returns:
    str: Tabla formateada para impresión*

#### `setup.py`
**Path:** `setup.py`

*No symbols extracted*

#### `test_integration.py`
**Path:** `tests/test_integration.py`

**Classs:**
- `ModeloMNISTPequeno` (line 24)
- `CNNMNIST` (line 123)
- `ModeloGrande` (line 188)

**Functions:**
- `test_integration_ultra_fast_experiment` (line 16) - *REPLICA EXPERIMENTO ULTRA-RÁPIDO COMPLETO
Objetivo: Validar que L predice colapso 2 épocas antes*
- `test_integration_complete_mnist` (line 115) - *REPLICA EXPERIMENTO COMPLETO MNIST
Objetivo: Validar que no genera falsos positivos en entrenamiento normal*
- `test_integration_forced_collapse` (line 180) - *REPLICA EXPERIMENTO COLAPSO FORZADO
Objetivo: Validar sensibilidad en condiciones extremas*
- `test_pip_install_format` (line 257) - *Valida que el paquete siga formato estándar de pip*
- `__init__` (line 25)
- `forward` (line 34)
- `__init__` (line 124)
- `forward` (line 134)
- `__init__` (line 189)
- `forward` (line 197)

#### `test_monitor.py`
**Path:** `tests/test_monitor.py`

**Functions:**
- `test_sovereignty_monitor_prediction` (line 17) - *Replica Experimento Ultra-Rápido:
L debe detectar colapso 2 épocas ANTES que val_loss.*
- `test_sovereignty_monitor_stable` (line 73) - *Replica Experimento MNIST Completo:
No debe generar falsos positivos en entrenamiento normal.*
- `test_sovereignty_monitor_forced_collapse` (line 109) - *Replica Experimento Colapso Forzado:
Detecta deterioro gradual en modelo grande con datos tóxicos.*
- `test_layer_diagnostics` (line 168) - *Verifica que analiza cada capa individualmente*
- `test_singular_entropy_function` (line 193) - *Test API simple singular_entropy()*
- `test_regime_function` (line 203) - *Test API simple regime()*
- `test_full_integration` (line 213) - *Replica entrenamiento completo con early stopping*

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
