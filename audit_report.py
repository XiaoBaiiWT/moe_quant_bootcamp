import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
from moe_core import MoeLayer
from quantization_lab import MoeQuantizer

BRAIN_PATH = "moe_brain.pkl"
RESULTS_IMG = "moe_audit_results.png"

def load_model():
    """loads a fresh, uncorrupted model structure."""
    return MoeLayer(input_dim=1, hidden_dim=32, num_experts=4)

def calculate_entropy(indices, num_experts=4):
    """
    Measures Routing Health.
    1.0 = Healthy (All experts used equally)
    0.0 = Dead/Collapse (Only 1 expert used)
    """
    counts = torch.bincount(indices.view(-1), minlength=num_experts).float()
    probs = counts / counts.sum()
    # Shannon Entropy with safety buffer
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8)).item()
    # Normalize (0 to 1)
    max_entropy = np.log2(num_experts)
    return entropy / max_entropy

def analyze_noise_ratio(state_path):
    """
    Scans the saved weights to find the 'Loudness' difference between experts.
    This proves WHY symmetric quantization fails.
    """
    print("\n   🔍 Expert Noise Scan (Raw Weight Analysis):")
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
    
    max_vals = []
    for i, expert in enumerate(state['experts']):
        # Check Layer 1 (W1)
        w_tensor = expert['w1'] # Raw tensor from pickle
        max_w = w_tensor.abs().max().item()
        max_vals.append(max_w)
        print(f"Expert {i} Max Signal: {max_w:.4f}")
    
    # Calculate Ratio
    loudest = max(max_vals)
    quietest = min(max_vals) + 1e-6
    ratio = loudest / quietest
    print(f" Noise Asymmetry Ratio: {ratio:.2f}x")
    print(f"(Expert {max_vals.index(max(max_vals))} is {ratio:.1f} times louder than Expert {max_vals.index(min(max_vals))})")
    return ratio

def run_scenario(scenario_name, model, x_test, y_test):
    """
    Runs a clinical trial on the model.
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        y_pred, weights, indices = model.forward(x_test)
        
        # Metrics
        mse = F.mse_loss(y_pred, y_test).item()
        entropy = calculate_entropy(indices, model.num_experts)
        
        # Get Top-1 Expert Choice for visualization
        # indices is [Batch, TopK]. We just want the primary expert [Batch, 0]
        primary_choices = indices[:, 0].numpy()
        
    print(f" {scenario_name} Results:")
    print(f"      MSE Loss:       {mse:.5f}")
    print(f"      Routing Entropy: {entropy:.4f}")
    return y_pred.numpy(), primary_choices, entropy, mse

# =========================================================================
# THE AUDIT RUNNER
# =========================================================================

if __name__ == "__main__":
    print(f"---  STARTING AUDIT ---")
    
    # 1. Generate Standardized Test Data (The "Exam")
    # 500 points from -1 to 1
    x_test = torch.linspace(-1, 1, 500).view(-1, 1)
    # Ground Truth: Sine wave for x < 0, Linear for x > 0
    y_test = torch.where(x_test < 0, torch.sin(3.14 * x_test), x_test)
    
    # 2. Analyze the Cause (The Noise Ratio)
    analyze_noise_ratio(BRAIN_PATH)

    # --- SCENARIO A: The Baseline (FP32) ---
    print("\n[A] ESTABLISHING BASELINE (FP32)...")
    model_A = load_model()
    # Inject using high precision to act as a "loader"
    MoeQuantizer.inject_quantisation(model_A, BRAIN_PATH, expert_bits=32, router_bits=32)
    y_A, choices_A, ent_A, mse_A = run_scenario("Baseline", model_A, x_test, y_test)

    # --- SCENARIO B: The Collapse (Symmetric INT4) ---
    print("\n[B] INJECTING FAILURE MODE (Symmetric INT4)...")
    model_B = load_model()
    MoeQuantizer.inject_quantisation(
        model_B, BRAIN_PATH, 
        expert_bits=4, 
        router_bits=32, # Keep router healthy to prove experts are the issue
        method='symmetric', 
        granularity='per_tensor' # The naive approach (High risk)
    )
    y_B, choices_B, ent_B, mse_B = run_scenario("Collapse", model_B, x_test, y_test)

    # --- SCENARIO C: The Fix (Asymmetric Group-Wise INT4) ---
    print("\n[C] APPLYING SURGICAL FIX (Asymmetric Group-Wise)...")
    model_C = load_model()
    MoeQuantizer.inject_quantisation(
        model_C, BRAIN_PATH, 
        expert_bits=4, 
        router_bits=32,
        method='asymmetric', 
        granularity='group_wise', # The fix
        group_size=32 # Small group size for high precision
    )
    y_C, choices_C, ent_C, mse_C = run_scenario("Recovery", model_C, x_test, y_test)

    # =========================================================================
    # VISUALIZATION (The "One-Page Report")
    # =========================================================================
    print(f"\n---  GENERATING graph ({RESULTS_IMG}) ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    x_np = x_test.numpy().flatten()
    
    scenarios = [
        ("Baseline (FP32)", y_A, choices_A, ent_A, mse_A),
        ("Collapse (Sym INT4)", y_B, choices_B, ent_B, mse_B),
        ("Fix (Asym Group INT4)", y_C, choices_C, ent_C, mse_C)
    ]

    for i, (title, y_pred, choices, ent, mse) in enumerate(scenarios):
        # Row 1: Function Prediction
        ax_func = axes[0, i]
        ax_func.plot(x_np, y_test.numpy(), 'k--', alpha=0.3, label="Ground Truth")
        ax_func.plot(x_np, y_pred, 'b-', linewidth=2, label="Prediction")
        ax_func.set_title(f"{title}\nMSE: {mse:.5f}")
        ax_func.grid(True, alpha=0.3)
        if i == 0: ax_func.legend()

        # Row 2: Expert Specialization Map
        ax_spec = axes[1, i]
        # Color map for 4 experts
        scatter = ax_spec.scatter(x_np, choices, c=choices, cmap='viridis', vmin=0, vmax=3, s=10, alpha=0.8)
        ax_spec.set_title(f"Expert Routing Map\nEntropy: {ent:.2f}")
        ax_spec.set_ylim(-0.5, 3.5)
        ax_spec.set_yticks(range(4))
        ax_spec.set_xlabel("Input Space (-1 to 1)")
        ax_spec.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_IMG)
    plt.show()
    print("---  AUDIT COMPLETE ---")