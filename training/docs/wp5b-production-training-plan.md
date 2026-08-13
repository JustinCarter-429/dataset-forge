# WP5B production training plan

Use validation only during training; do not inspect public or native test until WP6 certification. Reproduce the pinned model revision and isolated dependency profile on a CUDA GPU with sufficient VRAM for 4-bit QLoRA. Begin with QLoRA rank 16, alpha 16, dropout 0.05, NF4 double quantization, 4096-token candidate sequence length, batch size 1 with accumulation 8, and bounded checkpointing. Actual token counts, LoRA target-module matches, GPU sizing, and mixture ratios remain blocked on processor/model access and must be resolved before launch.
