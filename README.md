# moe_quant_bootcamp
MOE Benchmarking &amp; Optimization Toolkit
# MoE-Quant-Audit: Stabilizing 4-Bit Mixture-of-Experts 

A research-oriented implementation of a **Mixture-of-Experts (MoE) Hybrid Engine** designed to audit and resolve **Routing Instability** and **Numerical Fidelity Loss** under 4-bit quantization.

## 🔬 Research Context
In MoE architectures, experts naturally develop asymmetric weight manifolds (Dynamic Range variance). Naive 4-bit quantization (Symmetric) creates a "Global Scaling Error" that mutes quieter experts, leading to silent functional collapse even when routing entropy remains high.

This project reproduces the **IBM Research** findings on MoE quantization instability and provides a surgical fix using **Asymmetric Group-Wise Quantization**.

## 🛠️ Key Features
- **Hybrid Gradient Engine:** Manual backpropagation for task-specific weights coupled with PyTorch Autograd for load-balancing auxiliary loss.
- **Dynamic Aux-Loss Annealing:** A decay schedule for $\alpha$ (10.0 → 0.1) to stabilize expert specialization before quantization stress.
- **Surgical Quantization Suite:** Supports Per-Tensor, Per-Expert, and Group-Wise granularities with Dynamic Zero-Point Shifting.
- **Stability Auditor:** A 3-tier clinical audit (Oracle FP32, Symmetric INT4, Asymmetric INT4) with Shannon Entropy tracking.

## 📊 Experimental Results (Sample Audit)
| Metric | Baseline (FP32) | Symmetric (INT4) | Asymmetric (Fixed) |
| :--- | :--- | :--- | :--- |
| **MSE Loss** | 0.00002 | 0.00801 | 0.00028 |
| **Entropy** | 1.00 | 1.00 (Silent Collapse) | 1.00 |
| **Fidelity** | 100% | < 1% | **~96% Recovery** |

**Insight:** My audit proves that **Routing Entropy is a deceptive metric**. High entropy does not guarantee model health if the underlying specialist weights have suffered from static zero-point mapping.

## 🚀 How to Run
1. **Train the "Healthy Brain":** `python main_experiment.py`
2. **Run the Surgical Audit:** `python final_audit.py`
