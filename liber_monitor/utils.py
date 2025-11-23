"""
liber-monitor/utils.py v1.0.0
==============================
Herramientas de visualización y validación retroactiva.
Reproducen análisis de tus experimentos.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import warnings
import json

def plot_training_dynamics(history: List[Dict], save_path: Optional[str] = None):
    """
    Reproduce gráficos de tus 3 experimentos.
    Muestra L, pérdida y distribución de regímenes.
    """
    if not history:
        warnings.warn("Historial vacío, no se puede graficar")
        return
    
    epochs = list(range(len(history)))
    L_values = [h["L"] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Evolución de L
    ax1.plot(epochs, L_values, 'purple', linewidth=2.5, label='L (Libertad)')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.8, label='Umbral Healthy (1.0)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, label='Umbral Crítico (0.5)')
    ax1.fill_between(epochs, 0, 0.5, alpha=0.3, color='red', label='Zona Crítica')
    ax1.fill_between(epochs, 1.0, max(L_values)*1.1, alpha=0.3, color='green', label='Zona Healthy')
    
    ax1.set_title('🔥 SOVEREIGNTY MONITOR: Evolución de L durante Entrenamiento', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('L (Libertad)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribución de Regímenes
    regimes = []
    for h in history:
        for layer in h.get("layers", []):
            regimes.append(layer["regime"])
    
    regime_counts = {reg: regimes.count(reg) for reg in set(regimes)}
    colors = {'healthy': 'green', 'warning': 'orange', 'critical': 'red'}
    
    bars = ax2.bar(regime_counts.keys(), regime_counts.values(), 
                   color=[colors.get(r, 'gray') for r in regime_counts.keys()])
    
    ax2.set_title('Distribución de Regímenes por Capa')
    ax2.set_ylabel('Número de Capas')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado en {save_path}")
    
    plt.show()

def validate_early_stopping(history: List[Dict], 
                            overfitting_epoch: Optional[int],
                            threshold: float = 0.5) -> Dict:
    """
    Valida retroactivamente si L predijo el colapso antes que val_loss.
    Reproduce tu análisis de "2 épocas de anticipación".
    
    Args:
        history: Historial de L calculado
        overfitting_epoch: Época donde val_loss empeoró (ground truth)
        threshold: L < 0.5 = colapso (validado)
    
    Returns:
        Dict con anticipación en épocas y validez
    """
    if not history or overfitting_epoch is None:
        return {"valid": False, "error": "Datos insuficientes"}
    
    # Encontrar primera época donde L cruzó umbral
    L_cross_epoch = None
    for i, h in enumerate(history):
        if h["L"] < threshold:
            L_cross_epoch = i
            break
    
    if L_cross_epoch is None:
        return {
            "valid": False,
            "error": "L nunca cruzó el umbral crítico",
            "L_epochs": len(history),
            "overfitting_epoch": overfitting_epoch
        }
    
    # Calcular anticipación (negativa = después, positiva = antes)
    anticipation = overfitting_epoch - L_cross_epoch
    
    return {
        "valid": anticipation > 0,
        "anticipation_epochs": anticipation,
        "L_cross_epoch": L_cross_epoch,
        "L_value_at_cross": history[L_cross_epoch]["L"],
        "overfitting_epoch": overfitting_epoch,
        "message": (f"✅ L predijo el colapso {anticipation} épocas antes" if anticipation > 0
                   else f"❌ L no predijo antes (diferencia: {anticipation})")
    }

def export_report(history: List[Dict], 
                 model_name: str = "unknown",
                 save_path: str = "liber_report.json"):
    """
    Exporta reporte JSON para integración con pipelines de producción.
    Incluye recomendación de early stopping.
    """
    if not history:
        warnings.warn("Historial vacío")
        return
    
    report = {
        "model_name": model_name,
        "total_epochs": len(history),
        "final_L": history[-1]["L"],
        "final_regime": "healthy" if history[-1]["L"] > 1.0 else 
                       "warning" if history[-1]["L"] > 0.5 else "critical",
        "regime_distribution": {},
        "early_stopping_recommended": False,
        "layers_analyzed": len(history[-1].get("layers", []))
    }
    
    # Contar regímenes de última época
    for layer in history[-1].get("layers", []):
        reg = layer["regime"]
        report["regime_distribution"][reg] = report["regime_distribution"].get(reg, 0) + 1
    
    # Recomendación basada en últimas épocas
    recent_L = [h["L"] for h in history[-3:]]  # Últimas 3 épocas
    if len(recent_L) == 3 and all(L < 0.5 for L in recent_L):
        report["early_stopping_recommended"] = True
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Reporte exportado a {save_path}")
    return report
