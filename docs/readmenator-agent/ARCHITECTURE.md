# Architecture

## Internal Dependencies

- (no internal resolved imports)

## External Imports

- `app.py` -> gradio, json, liber_monitor, matplotlib.pyplot, numpy, torch, torch.nn, torch.optim
- `examples/quick_demo.py` -> liber_monitor, torch
- `experiments/01_ultra_fast.py` -> liber_monitor, matplotlib.pyplot, numpy, sys, torch, torch.nn, torch.optim
- `experiments/02_complete_mnist.py` -> liber_monitor, numpy, sys, torch, torch.nn, torch.optim, torchvision
- `experiments/03_forced_collapse.py` -> json, matplotlib.pyplot, numpy, pathlib, typing, warnings
- `setup.py` -> setuptools
- `tests/test_integration.py` -> json, liber_monitor, liber_monitor.monitor, numpy, pytest, tempfile, torch, torch.nn, torch.optim
- `tests/test_monitor.py` -> liber_monitor, liber_monitor.monitor, numpy, pytest, torch, torch.nn, torch.optim
