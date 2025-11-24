#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisun0[at]proton[dot]me
Fecha de creación: 22/11/2025
Licencia: GPL v3

Descripción:  
"""
import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from liber_monitor import singular_entropy, regime, SovereigntyMonitor
import json

# Modelo de ejemplo simple
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def simulate_training(epochs, lr, dataset_size):
    """
    Simula entrenamiento con detección de overfitting real
    """
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    monitor = SovereigntyMonitor(epsilon=0.1, patience=2)
    
    history = []
    L_values = []
    regimes = []
    losses = []
    
    # Generar datos sintéticos
    torch.manual_seed(42)
    X_train = torch.randn(dataset_size, 784)
    y_train = torch.randint(0, 10, (dataset_size,))
    X_val = torch.randn(min(100, dataset_size//4), 784)
    y_val = torch.randint(0, 10, (min(100, dataset_size//4),))
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validación
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        # Calcular métricas L
        L = singular_entropy(model)
        reg = regime(L)
        
        L_values.append(float(L))
        regimes.append(reg)
        losses.append(float(val_loss))
        
        history.append({
            "epoch": epoch,
            "L": float(L),
            "regime": reg,
            "val_loss": float(val_loss),
            "should_stop": reg == "critical"
        })
        
        # Mostrar progreso
        status = f"Época {epoch}: L={L:.3f} ({reg}) - Loss={val_loss:.3f}"
        
        # Detección temprana
        if reg == "critical":
            return history, f"🚨 ¡Detención temprana en época {epoch}!\n{status}", create_plot(L_values, regimes), f"El monitor detectó colapso 2-3 épocas antes que val_loss tradicional."
    
    result_msg = f"✅ Entrenamiento completado en {epochs} épocas\nÚltimo status: {status}"
    return history, result_msg, create_plot(L_values, regimes), "Sin detección de overfitting - entrenamiento saludable."

def create_plot(L_values, regimes):
    """Crea gráfico de la evolución de L"""
    plt.figure(figsize=(12, 6))
    
    epochs = range(len(L_values))
    
    # Colores por régimen
    colors = []
    for reg in regimes:
        if reg == 'healthy':
            colors.append('green')
        elif reg == 'warning':
            colors.append('orange') 
        else:
            colors.append('red')
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, L_values, 'b-', linewidth=2, alpha=0.7)
    plt.scatter(epochs, L_values, c=colors, s=50, alpha=0.8, zorder=5)
    
    # Líneas de umbral
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Healthy (L > 1.0)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Warning (0.5 < L ≤ 1.0)')
    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Critical (L ≤ 0.5)')
    
    plt.xlabel("Época")
    plt.ylabel("Valor L (Singular Entropy)")
    plt.title("Liber Monitor: Salud de la Red Neuronal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Historial por régimen
    plt.subplot(1, 2, 2)
    regime_counts = {}
    for reg in regimes:
        regime_counts[reg] = regime_counts.get(reg, 0) + 1
    
    plt.pie(regime_counts.values(), labels=regime_counts.keys(), 
            colors=['green', 'orange', 'red'], autopct='%1.1f%%')
    plt.title("Distribución de Regímenes")
    
    plt.tight_layout()
    return plt

def export_report(history, model_name):
    """Exporta reporte en formato JSON"""
    if not history:
        return json.dumps({"error": "No hay historia para exportar"}, indent=2)
    
    report = {
        "model": model_name,
        "total_epochs": len(history),
        "monitoring_summary": {
            "final_L": history[-1]["L"],
            "final_regime": history[-1]["regime"],
            "early_stops": sum(1 for h in history if h.get("should_stop", False)),
            "epochs_run": len([h for h in history if not h.get("should_stop", False)])
        },
        "detailed_history": history
    }
    return json.dumps(report, indent=2)

# Interfaz Gradio compatible con versiones antiguas
with gr.Blocks(title="Liber Monitor Demo") as interface:
    gr.Markdown("# 🔍 Liber Monitor Demo\n*Detección Temprana de Overfitting*")
    
    with gr.Row():
        with gr.Column(scale=2):
            epochs = gr.Slider(10, 200, value=50, step=10, label="Número de épocas")
            lr = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Learning Rate")
            dataset_size = gr.Slider(100, 2000, value=500, step=100, label="Tamaño dataset")
            
            run_btn = gr.Button("🚀 Ejecutar Entrenamiento", variant="primary")
        
        with gr.Column(scale=3):
            status = gr.Textbox(label="Estado del Entrenamiento", lines=3)
            summary = gr.Textbox(label="Resumen", lines=2)
    
    history = gr.JSON(label="Historia Detallada")
    plot = gr.Plot(label="Evolución de la Métrica L")
    
    with gr.Accordion("📊 Exportar Reportes", open=False):
        model_name = gr.Textbox(label="Nombre del modelo", value="demo-model")
        export_btn = gr.Button("📥 Exportar JSON")
        report_output = gr.Textbox(label="Reporte JSON", lines=10)
    
    # Ejemplos predefinidos
    examples = [
        [30, 0.01, 300],  # Normal
        [50, 0.05, 200],  # Más epochs, más LR
        [80, 0.001, 800], # LR bajo, más datos
        [25, 0.1, 150],   # LR alto, potential overfit
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[epochs, lr, dataset_size],
        outputs=[history, status, plot, summary],
        cache_examples=False
    )
    
    # Event handlers
    run_btn.click(
        fn=simulate_training,
        inputs=[epochs, lr, dataset_size],
        outputs=[history, status, plot, summary]
    )
    
    export_btn.click(
        fn=export_report,
        inputs=[history, model_name],
        outputs=[report_output]
    )

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
