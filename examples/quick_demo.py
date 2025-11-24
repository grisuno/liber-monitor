# examples/quick_demo.py
import torch
from liber_monitor import singular_entropy, regime

model = torch.nn.Linear(100, 50)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    # Simula training
    loss = torch.randn(1).item()
    optimizer.step()
    
    L = singular_entropy(model)
    print(f"Epoch {epoch}: L={L:.2f} ({regime(L)})")
    
