# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 9 | **Total Symbols Extracted:** 41 | **Total Imports:** 61

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:e63a2e6 | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Suggested Questions](#suggested-questions)
6. [Hotspot Analysis](#hotspot-analysis)
7. [Change Impact Analysis](#change-impact-analysis)
8. [Suggested Linting Rules](#suggested-linting-rules)
9. [Orphans](#orphans)
10. [Query Recipes](#query-recipes)
11. [Structural Knowledge Map](#structural-knowledge-map)
12. [Code Property Graph](#code-property-graph)
13. [Architecture Reference](#architecture-reference)
    - [PY (8 files)](#py-8-files)
    - [SH (1 files)](#sh-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 9 |
| Total Symbols | 41 |
| Total Imports | 61 |
| Call Edges | 628 |
| Inheritance Edges | 6 |
| Languages | 2 |
| Avg Symbols/File | 4.6 |
| Avg Imports/File | 6.8 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `test_monitor.py` | 16 | 7 | py |
| `test_integration.py` | 14 | 13 | py |
| `app.py` | 8 | 6 | py |
| `01_ultra_fast.py` | 7 | 4 | py |
| `02_complete_mnist.py` | 7 | 4 | py |
| `03_forced_collapse.py` | 6 | 7 | py |
| `quick_demo.py` | 2 | 0 | py |
| `setup.py` | 1 | 0 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| utility | 5 |
| testing | 2 |
| presentation | 1 |
| infrastructure | 1 |

### utility

- `app.py` (py, 6 symbols)
- `01_ultra_fast.py` (py, 4 symbols)
- `02_complete_mnist.py` (py, 4 symbols)
- `03_forced_collapse.py` (py, 7 symbols)
- `install.sh` (sh, 0 symbols)

### presentation

- `quick_demo.py` (py, 0 symbols)

### infrastructure

- `setup.py` (py, 0 symbols)

### testing

- `test_integration.py` (py, 13 symbols)
- `test_monitor.py` (py, 7 symbols)

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `quick_demo.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 2 | `03_forced_collapse.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 3 | `test_monitor.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 4 | `app.py` | 0.0667 | 0.0000 | 0.0000 | 0.00 | 0.67 |
| 5 | `test_integration.py` | 0.0308 | 0.0000 | 0.0000 | 0.00 | 0.31 |
| 6 | `01_ultra_fast.py` | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| 7 | `02_complete_mnist.py` | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| 8 | `install.sh` | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| 9 | `setup.py` | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `test_integration.py` | 1.3 | | 0.0000 |
| `03_forced_collapse.py` | 0.7 | | 0.0000 |
| `test_monitor.py` | 0.7 | | 0.0000 |
| `app.py` | 0.6 | | 0.0000 |
| `01_ultra_fast.py` | 0.4 | | 0.0000 |
| `02_complete_mnist.py` | 0.4 | | 0.0000 |
| `quick_demo.py` | 0.0 | | 0.0000 |
| `install.sh` | 0.0 | | 0.0000 |
| `setup.py` | 0.0 | | 0.0000 |

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does test_integration.py depend on, and what depends on it? (0 connections)
- What does 03_forced_collapse.py depend on, and what depends on it? (0 connections)
- What does test_monitor.py depend on, and what depends on it? (0 connections)
- What is SimpleNet in app.py and how is it used?
- What is ModeloMNISTPequeno in 01_ultra_fast.py and how is it used?

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `quick_demo.py` | 0.000 | 0.125 | 0.075 | 0 | 2 |
| `03_forced_collapse.py` | 0.538 | 0.375 | 0.440 | 7 | 6 |
| `test_monitor.py` | 0.538 | 1.000 | 0.815 | 7 | 16 |
| `app.py` | 0.462 | 0.500 | 0.485 | 6 | 8 |
| `test_integration.py` | 1.000 | 0.875 | 0.925 | 13 | 14 |
| `01_ultra_fast.py` | 0.308 | 0.438 | 0.386 | 4 | 7 |
| `02_complete_mnist.py` | 0.308 | 0.438 | 0.386 | 4 | 7 |
| `install.sh` | 0.000 | 0.000 | 0.000 | 0 | 0 |
| `setup.py` | 0.000 | 0.062 | 0.037 | 0 | 1 |

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `app.py` | 0 | 0 | 0 |
| `quick_demo.py` | 0 | 0 | 0 |
| `01_ultra_fast.py` | 0 | 0 | 0 |
| `02_complete_mnist.py` | 0 | 0 | 0 |
| `03_forced_collapse.py` | 0 | 0 | 0 |
| `install.sh` | 0 | 0 | 0 |
| `setup.py` | 0 | 0 | 0 |
| `test_integration.py` | 0 | 0 | 0 |
| `test_monitor.py` | 0 | 0 | 0 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM001` | info | Large number of functions in py: 35 total | py | 35 |
| `RM002` | info | Print statement found (consider logging instead) | python | 43 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `01_ultra_fast.py` (4 symbols, no doc)
- `02_complete_mnist.py` (4 symbols, no doc)
- `install.sh` (0 symbols, no doc)
- `setup.py` (0 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

## Structural Knowledge Map

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
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
    app_py["app.py (py)"]
    class app_py mod;
    experiments_01_ultra_fast_py["01_ultra_fast.py (py)"]
    class experiments_01_ultra_fast_py mod;
    experiments_02_complete_mnist_py["02_complete_mnist.py (py)"]
    class experiments_02_complete_mnist_py mod;
    experiments_03_forced_collapse_py["03_forced_collapse.py (py)"]
    class experiments_03_forced_collapse_py mod;
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

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://readmenator.dev/cpg/v1", "analysis": {"communities": [], "god_nodes": [{"node_id": "tests/test_integration.py", "score": 1.3}, {"node_id": "experiments/03_forced_collapse.py", "score": 0.7}, {"node_id": "tests/test_monitor.py", "score": 0.7}, {"node_id": "app.py", "score": 0.6}, {"node_id": "experiments/01_ultra_fast.py", "score": 0.4}, {"node_id": "experiments/02_complete_mnist.py", "score": 0.4}, {"node_id": "examples/quick_demo.py", "score": 0.0}, {"node_id": "install.sh", "score": 0.0}, {"node_id": "setup.py", "score": 0.0}], "surprising_connections": []}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "gradio"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "examples/quick_demo.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "examples/quick_demo.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/01_ultra_fast.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/01_ultra_fast.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/01_ultra_fast.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/01_ultra_fast.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/01_ultra_fast.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/01_ultra_fast.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/01_ultra_fast.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/02_complete_mnist.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/02_complete_mnist.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/02_complete_mnist.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/02_complete_mnist.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/02_complete_mnist.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/02_complete_mnist.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/02_complete_mnist.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/03_forced_collapse.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/03_forced_collapse.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/03_forced_collapse.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/03_forced_collapse.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/03_forced_collapse.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/03_forced_collapse.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "setup.py", "target": "setuptools"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "pytest"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "tempfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "liber_monitor.monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_integration.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "pytest"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor.monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor.monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "tests/test_monitor.py", "target": "liber_monitor.monitor"}], "generator": "readmenator", "metadata": {"edge_count": 695, "file_count": 9, "language_count": 2, "symbol_count": 41}, "nodes": [{"doc": "_*_ coding: utf8 _*_", "id": "app.py", "kind": "module", "label": "app.py", "language": "py", "sha256": "d7ecbe96e1f31754", "symbol_count": 6, "symbols": [{"kind": "class", "line": 23, "name": "SimpleNet", "signature": "class SimpleNet(Module)"}, {"doc": "Simula entrenamiento con detección de overfitting real", "kind": "method", "line": 35, "name": "simulate_training", "signature": "def simulate_training(epochs, lr, dataset_size)"}, {"doc": "Crea gráfico de la evolución de L", "kind": "method", "line": 97, "name": "create_plot", "signature": "def create_plot(L_values, regimes)"}, {"doc": "Exporta reporte en formato JSON", "kind": "method", "line": 141, "name": "export_report", "signature": "def export_report(history, model_name)"}, {"kind": "method", "line": 24, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 30, "name": "forward", "signature": "def forward(self, x)"}]}, {"doc": "examples/quick_demo.py", "id": "examples/quick_demo.py", "kind": "module", "label": "quick_demo.py", "language": "py", "sha256": "5660731a916bb148", "symbol_count": 0, "symbols": []}, {"id": "experiments/01_ultra_fast.py", "kind": "module", "label": "01_ultra_fast.py", "language": "py", "sha256": "35c8099ba2cbd941", "symbol_count": 4, "symbols": [{"kind": "class", "line": 20, "name": "ModeloMNISTPequeno", "signature": "class ModeloMNISTPequeno(Module)"}, {"kind": "method", "line": 38, "name": "run", "signature": "def run()"}, {"kind": "method", "line": 21, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 30, "name": "forward", "signature": "def forward(self, x)"}]}, {"id": "experiments/02_complete_mnist.py", "kind": "module", "label": "02_complete_mnist.py", "language": "py", "sha256": "df15fe28a4a2b0e6", "symbol_count": 4, "symbols": [{"kind": "class", "line": 19, "name": "CNNMNIST", "signature": "class CNNMNIST(Module)"}, {"kind": "method", "line": 40, "name": "run", "signature": "def run()"}, {"kind": "method", "line": 20, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 30, "name": "forward", "signature": "def forward(self, x)"}]}, {"id": "experiments/03_forced_collapse.py", "kind": "module", "label": "03_forced_collapse.py", "language": "py", "sha256": "d919d5ddc4bcf756", "symbol_count": 7, "symbols": [{"doc": "Configuración de matplotlib consolidada de los 3 experimentos\nManeja diferentes backends y fuentes internacionales", "kind": "function", "line": 15, "name": "setup_matplotlib", "signature": "def setup_matplotlib()"}, {"doc": "Gráficos comprehensivos consolidados de los 3 experimentos\nReproduce el análisis completo: L, pérdida, capas, correlaciones\n\nArgs:\n    history: Lista de snapshots de época (monitor.history)\n    loss_train: Pérdidas de entrenamiento (opcional)\n    loss_val: Pérdidas de validación (opcional)\n    save_path: Ruta para guardar el gráfico\n    show_layers: True para mostrar L por capa individual\n    dpi: Resolución del gráfico", "kind": "function", "line": 39, "name": "plot_training_dynamics", "signature": "def plot_training_dynamics(history, loss_train, loss_val, save_path, show_layers, dpi)"}, {"doc": "Valida retroactivamente si L predijo el colapso antes que val_loss\nReproduce el análisis de \"2-3 épocas de anticipación\" de los experimentos\n\nArgs:\n    history: Historial de L calculado (monitor.history)\n    loss_val: Pérdidas de validación (opcional)\n    threshold_L: L < 0.5 = colapso (validado)\n    threshold_loss_increase: Aumento % de val_loss para declarar overfitting\n\nReturns:\n    Dict con análisis completo de poder predictivo", "kind": "function", "line": 196, "name": "validate_early_stopping", "signature": "def validate_early_stopping(history, loss_val, threshold_L, threshold_loss_increase)"}, {"doc": "Exporta reporte JSON comprehensivo para integración con pipelines\nConsolidado de los 3 experimentos\n\nArgs:\n    history: Historial completo (monitor.history)\n    model_name: Nombre del modelo para identificación\n    save_path: Ruta para guardar JSON\n    include_layers: True para incluir diagnósticos detallados por capa\n\nReturns:\n    Dict con el reporte completo", "kind": "function", "line": 312, "name": "export_report", "signature": "def export_report(history, model_name, save_path, include_layers)"}, {"doc": "Detecta la primera época donde se observó colapso (L < threshold)\n\nReturns:\n    int: Época del colapso, o None si no hubo colapso", "kind": "function", "line": 420, "name": "detect_collapse_epoch", "signature": "def detect_collapse_epoch(history, threshold)"}, {"doc": "Calcula la tendencia de L en las últimas `window` épocas\n\nReturns:\n    str: \"ascending\", \"descending\", \"stable\", o \"insufficient_data\"", "kind": "function", "line": 432, "name": "calculate_trend", "signature": "def calculate_trend(history, window)"}, {"doc": "Genera tabla resumen en formato texto para consola\nConsolida las tablas de los 3 experimentos\n\nReturns:\n    str: Tabla formateada para impresión", "kind": "function", "line": 457, "name": "summary_table", "signature": "def summary_table(history, loss_train, loss_val, n_epochs)"}]}, {"id": "install.sh", "kind": "module", "label": "install.sh", "language": "sh", "sha256": "c907d80fd6734993", "symbol_count": 0, "symbols": []}, {"id": "setup.py", "kind": "module", "label": "setup.py", "language": "py", "sha256": "9b546ab490841195", "symbol_count": 0, "symbols": []}, {"id": "tests/test_integration.py", "kind": "module", "label": "test_integration.py", "language": "py", "sha256": "efae52e7344e7d44", "symbol_count": 13, "symbols": [{"doc": "REPLICA EXPERIMENTO ULTRA-RÁPIDO COMPLETO\nObjetivo: Validar que L predice colapso 2 épocas antes", "kind": "function", "line": 16, "name": "test_integration_ultra_fast_experiment", "signature": "def test_integration_ultra_fast_experiment()"}, {"doc": "REPLICA EXPERIMENTO COMPLETO MNIST\nObjetivo: Validar que no genera falsos positivos en entrenamiento normal", "kind": "function", "line": 115, "name": "test_integration_complete_mnist", "signature": "def test_integration_complete_mnist()"}, {"doc": "REPLICA EXPERIMENTO COLAPSO FORZADO\nObjetivo: Validar sensibilidad en condiciones extremas", "kind": "function", "line": 180, "name": "test_integration_forced_collapse", "signature": "def test_integration_forced_collapse()"}, {"doc": "Valida que el paquete siga formato estándar de pip", "kind": "function", "line": 257, "name": "test_pip_install_format", "signature": "def test_pip_install_format()"}, {"kind": "class", "line": 24, "name": "ModeloMNISTPequeno", "signature": "class ModeloMNISTPequeno(Module)"}, {"kind": "class", "line": 123, "name": "CNNMNIST", "signature": "class CNNMNIST(Module)"}, {"kind": "class", "line": 188, "name": "ModeloGrande", "signature": "class ModeloGrande(Module)"}, {"kind": "method", "line": 25, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 34, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 124, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 134, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 189, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 197, "name": "forward", "signature": "def forward(self, x)"}]}, {"id": "tests/test_monitor.py", "kind": "module", "label": "test_monitor.py", "language": "py", "sha256": "4412d285b2103347", "symbol_count": 7, "symbols": [{"doc": "Replica Experimento Ultra-Rápido:\nL debe detectar colapso 2 épocas ANTES que val_loss.", "kind": "function", "line": 17, "name": "test_sovereignty_monitor_prediction", "signature": "def test_sovereignty_monitor_prediction()"}, {"doc": "Replica Experimento MNIST Completo:\nNo debe generar falsos positivos en entrenamiento normal.", "kind": "function", "line": 73, "name": "test_sovereignty_monitor_stable", "signature": "def test_sovereignty_monitor_stable()"}, {"doc": "Replica Experimento Colapso Forzado:\nDetecta deterioro gradual en modelo grande con datos tóxicos.", "kind": "function", "line": 109, "name": "test_sovereignty_monitor_forced_collapse", "signature": "def test_sovereignty_monitor_forced_collapse()"}, {"doc": "Verifica que analiza cada capa individualmente", "kind": "function", "line": 168, "name": "test_layer_diagnostics", "signature": "def test_layer_diagnostics()"}, {"doc": "Test API simple singular_entropy()", "kind": "function", "line": 193, "name": "test_singular_entropy_function", "signature": "def test_singular_entropy_function()"}, {"doc": "Test API simple regime()", "kind": "function", "line": 203, "name": "test_regime_function", "signature": "def test_regime_function()"}, {"doc": "Replica entrenamiento completo con early stopping", "kind": "function", "line": 213, "name": "test_full_integration", "signature": "def test_full_integration(tmp_path)"}]}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### PY (8 files)

#### `app.py`
**Path:** `app.py`
**File Doc:** *_*_ coding: utf8 _*_*

**Classes:**
- `SimpleNet` (line 23) `class SimpleNet(Module)`

**Methods:**
- `simulate_training` (line 35) `def simulate_training(epochs, lr, dataset_size)` - *Simula entrenamiento con detección de overfitting real*
- `create_plot` (line 97) `def create_plot(L_values, regimes)` - *Crea gráfico de la evolución de L*
- `export_report` (line 141) `def export_report(history, model_name)` - *Exporta reporte en formato JSON*
- `__init__` (line 24) `def __init__(self)`
- `forward` (line 30) `def forward(self, x)`

#### `quick_demo.py`
**Path:** `examples/quick_demo.py`
**File Doc:** *examples/quick_demo.py*

*No symbols extracted*

#### `01_ultra_fast.py`
**Path:** `experiments/01_ultra_fast.py`

**Classes:**
- `ModeloMNISTPequeno` (line 20) `class ModeloMNISTPequeno(Module)`

**Methods:**
- `run` (line 38) `def run()`
- `__init__` (line 21) `def __init__(self)`
- `forward` (line 30) `def forward(self, x)`

#### `02_complete_mnist.py`
**Path:** `experiments/02_complete_mnist.py`

**Classes:**
- `CNNMNIST` (line 19) `class CNNMNIST(Module)`

**Methods:**
- `run` (line 40) `def run()`
- `__init__` (line 20) `def __init__(self)`
- `forward` (line 30) `def forward(self, x)`

#### `03_forced_collapse.py`
**Path:** `experiments/03_forced_collapse.py`

**Functions:**
- `setup_matplotlib` (line 15) `def setup_matplotlib()` - *Configuración de matplotlib consolidada de los 3 experimentos
Maneja diferentes backends y fuentes internacionales*
- `plot_training_dynamics` (line 39) `def plot_training_dynamics(history, loss_train, loss_val, save_path, show_layers, dpi)` - *Gráficos comprehensivos consolidados de los 3 experimentos
Reproduce el análisis completo: L, pérdida, capas, correlaciones

Args:
    history: Lista de snapshots de época (monitor.history)
    loss_train: Pérdidas de entrenamiento (opcional)
    loss_val: Pérdidas de validación (opcional)
    save_path: Ruta para guardar el gráfico
    show_layers: True para mostrar L por capa individual
    dpi: Resolución del gráfico*
- `validate_early_stopping` (line 196) `def validate_early_stopping(history, loss_val, threshold_L, threshold_loss_increase)` - *Valida retroactivamente si L predijo el colapso antes que val_loss
Reproduce el análisis de "2-3 épocas de anticipación" de los experimentos

Args:
    history: Historial de L calculado (monitor.history)
    loss_val: Pérdidas de validación (opcional)
    threshold_L: L < 0.5 = colapso (validado)
    threshold_loss_increase: Aumento % de val_loss para declarar overfitting

Returns:
    Dict con análisis completo de poder predictivo*
- `export_report` (line 312) `def export_report(history, model_name, save_path, include_layers)` - *Exporta reporte JSON comprehensivo para integración con pipelines
Consolidado de los 3 experimentos

Args:
    history: Historial completo (monitor.history)
    model_name: Nombre del modelo para identificación
    save_path: Ruta para guardar JSON
    include_layers: True para incluir diagnósticos detallados por capa

Returns:
    Dict con el reporte completo*
- `detect_collapse_epoch` (line 420) `def detect_collapse_epoch(history, threshold)` - *Detecta la primera época donde se observó colapso (L < threshold)

Returns:
    int: Época del colapso, o None si no hubo colapso*
- `calculate_trend` (line 432) `def calculate_trend(history, window)` - *Calcula la tendencia de L en las últimas `window` épocas

Returns:
    str: "ascending", "descending", "stable", o "insufficient_data"*
- `summary_table` (line 457) `def summary_table(history, loss_train, loss_val, n_epochs)` - *Genera tabla resumen en formato texto para consola
Consolida las tablas de los 3 experimentos

Returns:
    str: Tabla formateada para impresión*

#### `setup.py`
**Path:** `setup.py`

*No symbols extracted*

#### `test_integration.py`
**Path:** `tests/test_integration.py`

**Classes:**
- `ModeloMNISTPequeno` (line 24) `class ModeloMNISTPequeno(Module)`
- `CNNMNIST` (line 123) `class CNNMNIST(Module)`
- `ModeloGrande` (line 188) `class ModeloGrande(Module)`

**Functions:**
- `test_integration_ultra_fast_experiment` (line 16) `def test_integration_ultra_fast_experiment()` - *REPLICA EXPERIMENTO ULTRA-RÁPIDO COMPLETO
Objetivo: Validar que L predice colapso 2 épocas antes*
- `test_integration_complete_mnist` (line 115) `def test_integration_complete_mnist()` - *REPLICA EXPERIMENTO COMPLETO MNIST
Objetivo: Validar que no genera falsos positivos en entrenamiento normal*
- `test_integration_forced_collapse` (line 180) `def test_integration_forced_collapse()` - *REPLICA EXPERIMENTO COLAPSO FORZADO
Objetivo: Validar sensibilidad en condiciones extremas*
- `test_pip_install_format` (line 257) `def test_pip_install_format()` - *Valida que el paquete siga formato estándar de pip*

**Methods:**
- `__init__` (line 25) `def __init__(self)`
- `forward` (line 34) `def forward(self, x)`
- `__init__` (line 124) `def __init__(self)`
- `forward` (line 134) `def forward(self, x)`
- `__init__` (line 189) `def __init__(self)`
- `forward` (line 197) `def forward(self, x)`

#### `test_monitor.py`
**Path:** `tests/test_monitor.py`

**Functions:**
- `test_sovereignty_monitor_prediction` (line 17) `def test_sovereignty_monitor_prediction()` - *Replica Experimento Ultra-Rápido:
L debe detectar colapso 2 épocas ANTES que val_loss.*
- `test_sovereignty_monitor_stable` (line 73) `def test_sovereignty_monitor_stable()` - *Replica Experimento MNIST Completo:
No debe generar falsos positivos en entrenamiento normal.*
- `test_sovereignty_monitor_forced_collapse` (line 109) `def test_sovereignty_monitor_forced_collapse()` - *Replica Experimento Colapso Forzado:
Detecta deterioro gradual en modelo grande con datos tóxicos.*
- `test_layer_diagnostics` (line 168) `def test_layer_diagnostics()` - *Verifica que analiza cada capa individualmente*
- `test_singular_entropy_function` (line 193) `def test_singular_entropy_function()` - *Test API simple singular_entropy()*
- `test_regime_function` (line 203) `def test_regime_function()` - *Test API simple regime()*
- `test_full_integration` (line 213) `def test_full_integration(tmp_path)` - *Replica entrenamiento completo con early stopping*

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
