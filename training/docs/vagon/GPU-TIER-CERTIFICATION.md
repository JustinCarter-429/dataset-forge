# GPU-tier certification

After switching tiers, rerun CUDA preflight. The actual GPU configuration determines VRAM and precision. A 24 GB tier should begin with 4-bit NF4, batch size one, 2048-token smoke context, gradient checkpointing, and accumulation. A 10-step smoke is certification only, never production training.
