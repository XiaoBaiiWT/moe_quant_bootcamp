# moe_quant_bootcamp
MoE Benchmarking & Optimization Toolkit

# MoE-From-Scratch: Learning Routing, Experts, and Quantization by Hand

A from-scratch **Mixture-of-Experts (MoE)** implementation, built to understand how architectures like DeepSeek's and Kimi K2's MoE routing actually work — by implementing the router, experts, and backward passes by hand rather than relying on autograd.

## 🔬 What's here
- **Router:** top-k gating over experts, softmax over the selected logits, plus a load-balancing auxiliary loss to discourage the router from collapsing onto a few favorite experts.
- **Expert:** a small 2-layer MLP per expert (`Linear → ReLU → Linear`).
- **MoeLayer:** dispatches each token to its top-k experts via boolean masking, runs only the selected tokens through each expert, and scatters the weighted outputs back.

## 🛠️ Manual Backward Pass
Gradients for the router (including the top-k softmax gradient) and each expert are derived and implemented by hand, then handed back into PyTorch's `.grad` slots via `sync_gradients` so a standard optimizer can still be used, with manual gradient clipping applied at the handoff.

Some early exploration into quantization stability (per-tensor / per-expert / group-wise, with a stability-audit script) is also included, but not yet verified end to end — see below.

## 🚀 How to Run
1. **Train:** `python main_experiment.py`
2. **Audit:** `python final_audit.py`
