# Troubleshooting

- If PowerShell blocks scripts, use a user-scoped execution policy or invoke with `-ExecutionPolicy Bypass` for the current session only.
- If a Hugging Face download is interrupted, rerun the download script; the persistent `hf-cache` resumes it.
- If CUDA is unavailable after a tier switch, stop with `GPU_CERTIFICATION_BLOCKED`; do not train on CPU.
- For dependency, LoRA, completion-mask, OOM, or checkpoint-reload failures, stop certification and preserve the reported logs for review.
