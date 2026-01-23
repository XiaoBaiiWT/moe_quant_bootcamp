import torch
import torch.nn.functional as F
from moe_core import MoeLayer
import pickle
import os 

def train_moe(anneal_aux=True, epochs=401):
    """
    Trains the MoE model using standard PyTorch Autograd.
    Simple, Robust, and Industry Standard.
    """
    # 1. Initialize Model
    model = MoeLayer(input_dim=1, hidden_dim=32, num_experts=4)
    
    # 2. Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print('--- 🎓 Training Started (Standard Autograd) ---')
    
    # 3. Training Loop
    for epoch in range(epochs):    
        # A. Data Generation
        x = (torch.rand(100, 1) * 2) - 1
        y_target = torch.where(x < 0, torch.sin(3.14 * x), x)
        
        # B. Forward Pass
        out, weights, indices = model.forward(x)

        # C. Loss Calculation
        task_loss = F.mse_loss(out, y_target)
        
        # Dynamic Aux Weight
        if anneal_aux:
            alpha = 10.0 * (0.95**(epoch // 20))
        else:
            alpha = 0.1
        
        aux_loss = model.load_balancing_loss(weights, indices)
        
        # Combine them into one scalar
        total_loss = task_loss + (alpha * aux_loss)
        
        # D. Backward Pass (The "One-Line" Solution)
        optimizer.zero_grad()
        total_loss.backward() # PyTorch handles Task + Aux derivatives automatically
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step() 

        # E. Logging
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Task: {task_loss.item():.4f} | Aux: {aux_loss.item():.4f} | Alpha: {alpha:.3f}")
            
    return model



def save_trained_moe(model, filename="moe_brain.pkl"):
    """
    Extracts and serializes the raw weights/biases.
    Updated to use .weight and .bias naming convention.
    """   
    model_state = {
        'router': { 
            'w': model.router.gate.weight.detach().clone(),
            'b': model.router.gate.bias.detach().clone()
        },
        'experts': []
    }
    
    for e in model.experts:
        model_state['experts'].append({
            'w1': e.layer1.weight.detach().clone(),
            'b1': e.layer1.bias.detach().clone(),
            'w2': e.layer2.weight.detach().clone(),
            'b2': e.layer2.bias.detach().clone()
        })
        
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model_state, f)
        
        abs_path = os.path.abspath(filename)
        print("-" * 30)
        print(f"✅ SAVE SUCCESSFUL")
        print(f"📍 Path: {abs_path}")
        print(f"📦 Size: {os.path.getsize(abs_path) / 1024:.2f} KB")
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ SAVE FAILED: {e}")

if __name__ == "__main__":
    trained_model = train_moe(anneal_aux=True)
    save_trained_moe(trained_model)