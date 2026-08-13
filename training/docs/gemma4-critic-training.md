# Gemma 4 critic training foundation

The required text-only model is `google/gemma-4-E4B-it` at revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. Its Hub metadata reports a Gemma 4 conditional-generation architecture and Apache-2.0 tag. The critic renderer uses a fixed system instruction, deterministic JSON payloads, an explicit requested-target mask, and deterministic null-filled response envelope. No special tokens are added, thinking is disabled, and completion-only masking is required before training.

The pinned tokenizer was downloaded and verified as `GemmaTokenizer` with a 262,144-token vocabulary and the official chat template. The workstation is CPU-only, so no smoke fine-tune is allowed. Full model load, LoRA-module verification, and all-corpus real token statistics remain blocked rather than estimated.
