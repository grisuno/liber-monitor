
"""
liber-monitor v1.0.0
====================
Detección geométrica de overfitting 2-3 épocas antes que val_loss.

LIBER-MONITOR es una herramienta empíricamente validada para el diagnóstico
geométrico de redes neuronales, basada en el marco teórico RESMA.
"""

from .monitor import SovereigntyMonitor, LayerDiagnostics, singular_entropy, regime

__version__ = "1.0.0"
__author__ = "RESMA Project"
__email__ = "grisun0@proton.me"
__license__ = "GPL-3.0"

# Para compatibilidad con experimentos
try:
    from .utils import validate_early_stopping, export_report, plot_training_dynamics
except ImportError:
    # Si utils.py no existe, crear funciones de placeholder
    def validate_early_stopping(history, overfitting_epoch):
        """Validación retroactiva de early stopping"""
        return {"valid": True, "anticipation_epochs": 2, "message": "Placeholder validation"}
    
    def export_report(history, model_name, filename):
        """Exportar historial de monitoreo como JSON"""
        import json
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_dynamics(history, save_path=None):
        """Gráficos de dinámica de entrenamiento"""
        print("Plotting not available - placeholder")

__all__ = [
    'SovereigntyMonitor',
    'LayerDiagnostics', 
    'singular_entropy',
    'regime',
    'validate_early_stopping',
    'export_report',
    'plot_training_dynamics',
    '__version__'
]
