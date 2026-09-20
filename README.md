moe_quant_bootcamp

A from-scratch Mixture-of-Experts implementation, built to understand how architectures like DeepSeek's and Kimi K2's MoE routing actually work — by implementing the router, experts, and backward passes by hand rather than relying on autograd.

What's here
Router — top-k gating over experts, softmax over the selected logits, plus a load-balancing auxiliary loss to discourage the router from collapsing onto a few favorite experts.
Expert — a small 2-layer MLP per expert (Linear → ReLU → Linear).
MoeLayer — dispatches each token to its top-k experts via boolean masking, runs only the selected tokens through each expert, and scatters the weighted outputs back.
Manual backward pass — gradients for the router (including the top-k softmax gradient) and each expert are derived and implemented by hand, then handed back into PyTorch's .grad slots via sync_gradients so a standard optimizer can still be used, with manual gradient clipping applied at the handoff.
Some early exploration into quantization stability (per-tensor / per-expert / group-wise, with a stability-audit script) — not yet verified end to end.
Known issues

Re-reading the code surfaced two real bugs, not yet fixed:

MoeLayer.backward: grad_router[mask][pos_mask] = ... silently fails to write back to grad_router — chained boolean indexing in PyTorch returns a copy on the first index, so the assignment never touches the original tensor. Confirmed directly: grad_router stays all-zero after this line runs, no error raised. This means the router currently receives no gradient through the manual backward path.
LinearLayer.update(): self.weight -= lr * self.weight_grad is an in-place op on a leaf nn.Parameter with requires_grad=True, which PyTorch disallows and raises on immediately. Needs torch.no_grad() or to operate on .data.

Because of (1), the quantization audit results aren't yet trustworthy — the router isn't actually learning to route via this path, so any downstream fidelity numbers need to be re-run once the gradient flow is fixed.

How to run
bash
python main_experiment.py   # train
python final_audit.py       # audit
