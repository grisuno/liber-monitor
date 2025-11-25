"""
liber-monitor/utils.py v2.0.0
==============================
Herramientas de visualización, validación y exportación
Consolidado de los 3 experimentos (Completo, Ultra-Rápido, Extremo)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
import json
from pathlib import Path

def setup_matplotlib():
    """
    Configuración de matplotlib consolidada de los 3 experimentos
    Maneja diferentes backends y fuentes internacionales
    """
    warnings.filterwarnings('default')
    
    try:
        plt.switch_backend("Agg")  # Backend sin GUI para servidores
    except:
        pass
    
    try:
        plt.style.use("seaborn-v0_8")
    except:
        plt.style.use("default")
    
    # Fuentes internacionales (consolidado)
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", 
        "Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

def plot_training_dynamics(history: List[Dict],
                          loss_train: Optional[List[float]] = None,
                          loss_val: Optional[List[float]] = None,
                          save_path: str = "sovereignty_analysis.png",
                          show_layers: bool = True,
                          dpi: int = 300) -> None:
    """
    Gráficos comprehensivos consolidados de los 3 experimentos
    Reproduce el análisis completo: L, pérdida, capas, correlaciones
    
    Args:
        history: Lista de snapshots de época (monitor.history)
        loss_train: Pérdidas de entrenamiento (opcional)
        loss_val: Pérdidas de validación (opcional)
        save_path: Ruta para guardar el gráfico
        show_layers: True para mostrar L por capa individual
        dpi: Resolución del gráfico
    """
    if not history:
        warnings.warn("Historial vacío, no se puede graficar")
        return
    
    setup_matplotlib()
    
    epochs = [h.epoch for h in history]
    L_values = [h.L_promedio for h in history]
    regimes = [h.regime_promedio for h in history]
    entropias = [h.entropia_promedio for h in history]
    
    # Configurar subplot grid
    n_rows = 3 if (loss_train and loss_val) else 2
    fig = plt.figure(figsize=(16, 6 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.3, wspace=0.25)
    
    # ========================================================================
    # PLOT 1: EVOLUCIÓN DE L (EL MÁS IMPORTANTE)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, L_values, 'purple', linewidth=3, label='L Promedio', marker='o', markersize=4)
    
    # Umbrales validados empíricamente
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Umbral Soberano (1.0)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Umbral Espurio (0.5)')
    
    # Zonas de régimen
    ax1.fill_between(epochs, 0, 0.5, alpha=0.2, color='red', label='Zona Espurio')
    ax1.fill_between(epochs, 0.5, 1.0, alpha=0.2, color='orange', label='Zona Emergente')
    max_L = max(L_values) * 1.1
    ax1.fill_between(epochs, 1.0, max_L, alpha=0.2, color='green', label='Zona Soberano')
    
    ax1.set_title('🔥 SOVEREIGNTY MONITOR: Evolución de L durante Entrenamiento', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=11)
    ax1.set_ylabel('L (Libertad)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0, top=max_L)
    
    # ========================================================================
    # PLOT 2: DISTRIBUCIÓN DE REGÍMENES
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    regime_counts = {}
    for h in history:
        for layer in h.layers:
            reg = layer.regime
            regime_counts[reg] = regime_counts.get(reg, 0) + 1
    
    if regime_counts:
        colors = {'soberano': 'green', 'emergente': 'orange', 'espurio': 'red'}
        bars = ax2.bar(regime_counts.keys(), regime_counts.values(),
                      color=[colors.get(r, 'gray') for r in regime_counts.keys()],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax2.set_title('Distribución de Regímenes por Capa', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Número de Mediciones', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Anotar valores
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # PLOT 3: L POR CAPA INDIVIDUAL (si está disponible)
    # ========================================================================
    if show_layers and history[0].layers:
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Extraer L por capa a través del tiempo
        layer_names = [layer.layer_name for layer in history[0].layers]
        layer_colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
        
        for i, layer_name in enumerate(layer_names):
            L_layer = []
            for h in history:
                for layer in h.layers:
                    if layer.layer_name == layer_name:
                        L_layer.append(layer.L)
                        break
            
            if L_layer:
                ax3.plot(epochs[:len(L_layer)], L_layer, 
                        linewidth=2, label=layer_name, 
                        color=layer_colors[i], marker='o', markersize=3)
        
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Umbral Crítico')
        ax3.set_title('L por Capa Individual', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Época', fontsize=11)
        ax3.set_ylabel('L', fontsize=11)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # PLOT 4: ENTROPÍA DE VON NEUMANN
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, entropias, 'darkorange', linewidth=2.5, label='Entropía vN', marker='s', markersize=4)
    ax4.set_title('Evolución de Entropía de von Neumann', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Época', fontsize=11)
    ax4.set_ylabel('S_vN', fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # PLOT 5 y 6: PÉRDIDAS (si están disponibles)
    # ========================================================================
    if loss_train and loss_val:
        # Plot 5: Evolución de pérdida
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs[:len(loss_train)], loss_train, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
        ax5.plot(epochs[:len(loss_val)], loss_val, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
        ax5.set_title('Evolución de Pérdida: Detección de Overfitting', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Época', fontsize=11)
        ax5.set_ylabel('Loss', fontsize=11)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Correlación L vs Val Loss
        ax6 = fig.add_subplot(gs[2, 1])
        if len(L_values) == len(loss_val):
            scatter = ax6.scatter(L_values, loss_val, c=epochs, cmap='viridis', s=80, alpha=0.7, edgecolors='black')
            ax6.set_xlabel('L Promedio', fontsize=11)
            ax6.set_ylabel('Val Loss', fontsize=11)
            ax6.set_title('Correlación L vs Val Loss (color = época)', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax6, label='Época')
        else:
            ax6.text(0.5, 0.5, 'Dimensiones incompatibles\npara correlación', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"📊 Gráfico guardado en: {save_path}")
    plt.close()

def validate_early_stopping(history: List[object],
                           loss_val: Optional[List[float]] = None,
                           threshold_L: float = 0.5,
                           threshold_loss_increase: float = 0.10) -> Dict:
    """
    Valida retroactivamente si L predijo el colapso antes que val_loss
    Reproduce el análisis de "2-3 épocas de anticipación" de los experimentos
    
    Args:
        history: Historial de L calculado (monitor.history)
        loss_val: Pérdidas de validación (opcional)
        threshold_L: L < 0.5 = colapso (validado)
        threshold_loss_increase: Aumento % de val_loss para declarar overfitting
    
    Returns:
        Dict con análisis completo de poder predictivo
    """
    if not history:
        return {"valid": False, "error": "Historial vacío"}
    
    # Encontrar primera época donde L cruzó umbral crítico
    L_cross_epoch = None
    L_cross_value = None
    
    for h in history:
        if h.L_promedio < threshold_L:
            L_cross_epoch = h.epoch
            L_cross_value = h.L_promedio
            break
    
    # Detectar overfitting en val_loss (si está disponible)
    overfitting_epoch = None
    overfitting_value = None
    
    if loss_val and len(loss_val) > 2:
        # Buscar mínimo de val_loss
        min_val_loss = float('inf')
        min_epoch = 0
        
        for i, val_loss in enumerate(loss_val):
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_epoch = i
        
        # Buscar cuando val_loss sube threshold_loss_increase% desde el mínimo
        threshold_val = min_val_loss * (1 + threshold_loss_increase)
        for i in range(min_epoch + 1, len(loss_val)):
            if loss_val[i] > threshold_val:
                overfitting_epoch = i
                overfitting_value = loss_val[i]
                break
    
    # Construir reporte
    report = {
        "valid": L_cross_epoch is not None,
        "message": "",
        "anticipation_epochs": 0,
        "L_analysis": {
            "cross_detected": L_cross_epoch is not None,
            "cross_epoch": L_cross_epoch,
            "cross_value": round(L_cross_value, 4) if L_cross_value else None,
            "threshold": threshold_L
        },
        "overfitting_analysis": {
            "detected": overfitting_epoch is not None,
            "epoch": overfitting_epoch,
            "value": round(overfitting_value, 4) if overfitting_value else None,
            "threshold_increase": threshold_loss_increase
        },
        "predictive_power": {}
    }
    
    # Analizar poder predictivo
    if L_cross_epoch is not None and overfitting_epoch is not None:
        anticipation = overfitting_epoch - L_cross_epoch
        report["anticipation_epochs"] = anticipation
        
        report["predictive_power"] = {
            "valid": anticipation > 0,
            "anticipation_epochs": anticipation,
            "percentage_early": round((anticipation / overfitting_epoch) * 100, 1) if overfitting_epoch > 0 else 0
        }
        
        if anticipation > 0:
            report["message"] = (
                f"✅ ÉXITO: L predijo el colapso {anticipation} épocas ANTES del overfitting visible. "
                f"Esto confirma la hipótesis de RESMA sobre detección temprana."
            )
        else:
            report["message"] = (
                f"⚠️ RESULTADO MIXTO: L detectó el colapso pero no antes del overfitting "
                f"(diferencia: {anticipation} épocas)."
            )
    
    elif L_cross_epoch is not None:
        report["message"] = (
            f"🔮 L detectó colapso en época {L_cross_epoch}, "
            f"pero no se observó overfitting claro en val_loss. "
            f"Esto sugiere que L es más sensible a deterioro interno."
        )
    
    elif overfitting_epoch is not None:
        report["message"] = (
            f"⚠️ Se detectó overfitting en época {overfitting_epoch}, "
            f"pero L no cruzó el umbral crítico. "
            f"Considerar ajustar umbrales o aumentar épocas de monitoreo."
        )
    
    else:
        report["message"] = (
            f"✅ No se detectó colapso en L ni overfitting en val_loss. "
            f"El modelo mantuvo capacidad de generalización."
        )
    
    return report

def export_report(history: List[Dict],
                 model_name: str = "unknown",
                 save_path: str = "sovereignty_report.json",
                 include_layers: bool = False) -> Dict:
    """
    Exporta reporte JSON comprehensivo para integración con pipelines
    Consolidado de los 3 experimentos
    
    Args:
        history: Historial completo (monitor.history)
        model_name: Nombre del modelo para identificación
        save_path: Ruta para guardar JSON
        include_layers: True para incluir diagnósticos detallados por capa
    
    Returns:
        Dict con el reporte completo
    """
    if not history:
        warnings.warn("Historial vacío")
        return {}
    
    # Reporte global
    report = {
        "metadata": {
            "model_name": model_name,
            "total_epochs": len(history),
            "timestamp": str(np.datetime64('now'))
        },
        "final_state": {
            "L_promedio": history[-1].L_promedio,
            "regime": history[-1].regime_promedio,
            "entropia_promedio": history[-1].entropia_promedio,
            "rango_promedio": history[-1].rango_promedio
        },
        "statistics": {
            "L_inicial": history[0].L_promedio,
            "L_minimo": min(h.L_promedio for h in history),
            "L_maximo": max(h.L_promedio for h in history),
            "L_cambio_porcentual": None
        },
        "regime_distribution": {},
        "recommendations": {
            "early_stopping": False,
            "warnings": [],
            "actions": []
        }
    }
    
    # Calcular cambio porcentual
    L_inicial = history[0].L_promedio
    L_final = history[-1].L_promedio
    if L_inicial != 0:
        cambio = ((L_final - L_inicial) / L_inicial) * 100
        report["statistics"]["L_cambio_porcentual"] = round(cambio, 2)
    
    # Distribución de regímenes
    for h in history:
        regime = h.regime_promedio
        report["regime_distribution"][regime] = report["regime_distribution"].get(regime, 0) + 1
    
    # Análisis de tendencias y recomendaciones
    recent_L = [h.L_promedio for h in history[-3:]]
    
    if len(recent_L) == 3 and all(L < 0.5 for L in recent_L):
        report["recommendations"]["early_stopping"] = True
        report["recommendations"]["warnings"].append(
            "🚨 CRÍTICO: L < 0.5 por 3 épocas consecutivas - COLAPSO INMINENTE"
        )
        report["recommendations"]["actions"].append(
            "Detener entrenamiento inmediatamente o reducir learning rate drásticamente"
        )
    
    elif L_final < 0.5:
        report["recommendations"]["warnings"].append(
            f"⚠️ L crítico: {L_final:.3f} < 0.5"
        )
        report["recommendations"]["actions"].append(
            "Monitorear estrechamente. Considerar early stopping si persiste."
        )
    
    elif report["statistics"]["L_cambio_porcentual"] and report["statistics"]["L_cambio_porcentual"] < -20:
        report["recommendations"]["warnings"].append(
            f"📉 L bajó {abs(report['statistics']['L_cambio_porcentual']):.1f}% desde el inicio"
        )
        report["recommendations"]["actions"].append(
            "Degradación significativa detectada. Revisar hiperparámetros."
        )
    
    # Incluir capas si se solicita
    if include_layers:
        report["layer_details"] = []
        for h in history[-3:]:  # Últimas 3 épocas
            epoch_layers = {
                "epoch": h.epoch,
                "layers": [layer.to_dict() for layer in h.layers]
            }
            report["layer_details"].append(epoch_layers)
    
    # Guardar a archivo
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Reporte exportado a {save_path}")
    return report

def detect_collapse_epoch(history: List[object], threshold: float = 0.5) -> Optional[int]:
    """
    Detecta la primera época donde se observó colapso (L < threshold)
    
    Returns:
        int: Época del colapso, o None si no hubo colapso
    """
    for h in history:
        if h.L_promedio < threshold:
            return h.epoch
    return None

def calculate_trend(history: List[object], window: int = 3) -> str:
    """
    Calcula la tendencia de L en las últimas `window` épocas
    
    Returns:
        str: "ascending", "descending", "stable", o "insufficient_data"
    """
    if len(history) < window:
        return "insufficient_data"
    
    recent_L = [h.L_promedio for h in history[-window:]]
    
    # Calcular pendiente simple
    epochs = list(range(window))
    slope = np.polyfit(epochs, recent_L, 1)[0]
    
    threshold_slope = 0.01  # Umbral de estabilidad
    
    if slope > threshold_slope:
        return "ascending"
    elif slope < -threshold_slope:
        return "descending"
    else:
        return "stable"

def summary_table(history: List[Dict], 
                 loss_train: Optional[List[float]] = None,
                 loss_val: Optional[List[float]] = None,
                 n_epochs: Optional[int] = None) -> str:
    """
    Genera tabla resumen en formato texto para consola
    Consolida las tablas de los 3 experimentos
    
    Returns:
        str: Tabla formateada para impresión
    """
    if not history:
        return "❌ Historial vacío"
    
    n_epochs = n_epochs or len(history)
    
    # Header
    table = "\n" + "="*80 + "\n"
    table += "📋 SOVEREIGNTY MONITOR - RESUMEN DE ENTRENAMIENTO\n"
    table += "="*80 + "\n"
    
    # Encabezado de columnas
    header = f"{'Ep':<4} {'L':<7} {'Régimen':<12} {'S_vN':<7}"
    if loss_train:
        header += f" {'Train':<8}"
    if loss_val:
        header += f" {'Val':<8}"
    
    table += header + "\n"
    table += "-"*80 + "\n"
    
    # Filas
    for i, h in enumerate(history[:n_epochs]):
        row = f"{h.epoch:<4} {h.L_promedio:<7.3f} {h.regime_promedio:<12} {h.entropia_promedio:<7.3f}"
        
        if loss_train and i < len(loss_train):
            row += f" {loss_train[i]:<8.4f}"
        
        if loss_val and i < len(loss_val):
            row += f" {loss_val[i]:<8.4f}"
        
        table += row + "\n"
    
    table += "-"*80 + "\n"
    
    # Estadísticas finales
    L_inicial = history[0].L_promedio
    L_final = history[-1].L_promedio
    cambio = ((L_final - L_inicial) / L_inicial) * 100 if L_inicial != 0 else 0
    
    table += f"L inicial: {L_inicial:.3f} | L final: {L_final:.3f} | Cambio: {cambio:+.1f}%\n"
    table += "="*80 + "\n"
    
    return table
