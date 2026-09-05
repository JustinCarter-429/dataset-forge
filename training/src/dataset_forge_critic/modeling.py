"""Pinned Gemma 4 QLoRA loading with explicit no-fallback assertions."""
from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any

from .training_config import TrainingConfiguration

MODALITY_MARKERS = ("vision", "audio", "image", "multimodal_projector")


def model_source(config: TrainingConfiguration) -> tuple[str, dict[str, str]]:
    """Use the synchronized bucket copy when configured, otherwise the pinned Hub revision."""
    local = os.environ.get("DATASET_FORGE_MODEL_PATH")
    if local:
        path = Path(local).resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"LOCAL_MODEL_CONFIG_MISSING:{path}")
        return str(path), {"local_files_only": True}
    return config.model.model_id, {"revision": config.model.revision}


def discover_lora_modules(model: Any, text_regex: str) -> list[str]:
    """Return exact full names for linear modules in the configured text backbone."""
    try:
        import torch
    except ImportError as exc:
        raise ValueError("PYTORCH_REQUIRED_FOR_MODULE_DISCOVERY") from exc
    pattern = re.compile(text_regex)
    names = []
    for name, module in model.named_modules():
        lowered = name.casefold()
        if isinstance(module, torch.nn.Linear) and pattern.search(name) and not any(marker in lowered for marker in MODALITY_MARKERS):
            names.append(name)
    if not names:
        raise ValueError("LORA_TARGET_DISCOVERY_EMPTY")
    if len(names) != len(set(names)):
        raise ValueError("LORA_TARGET_DISCOVERY_DUPLICATE")
    return sorted(names)


def _target_pattern(names: list[str]) -> str:
    return "^(?:" + "|".join(re.escape(name) for name in names) + ")$"


def trainable_parameter_report(model: Any) -> dict[str, Any]:
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("NO_TRAINABLE_PARAMETERS")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise ValueError(f"UNEXPECTED_TRAINABLE_PARAMETERS:{unexpected[:5]}")
    return {"trainable_parameters": sum(count for _, count in trainable), "trainable_tensors": len(trainable), "names": [name for name, _ in trainable]}


def load_processor(config: TrainingConfiguration):
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ValueError("TRANSFORMERS_REQUIRED") from exc
    source, kwargs = model_source(config)
    return AutoProcessor.from_pretrained(source, **kwargs)


def load_qlora_model(config: TrainingConfiguration, *, resume_adapter=None):
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise ValueError(f"GPU_DEPENDENCY_MISSING:{exc.name}") from exc
    if not torch.cuda.is_available():
        raise ValueError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    if config.model.require_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16_REQUIRED_BUT_UNSUPPORTED")
    compute_dtype = torch.bfloat16 if config.model.require_bf16 else torch.float32
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    source, source_kwargs = model_source(config)
    model = AutoModelForMultimodalLM.from_pretrained(
        source, **source_kwargs, quantization_config=quantization,
        torch_dtype=compute_dtype, device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True,
    )
    architecture = model.config.architectures[0] if getattr(model.config, "architectures", None) else type(model).__name__
    if architecture != config.model.architecture and type(model).__name__ != config.model.architecture:
        raise ValueError(f"MODEL_ARCHITECTURE_MISMATCH:{architecture}")
    placement = getattr(model, "hf_device_map", {})
    if any(str(device).casefold() in {"cpu", "disk"} for device in placement.values()):
        raise ValueError(f"CPU_OR_DISK_OFFLOAD_FORBIDDEN:{placement}")
    for name, parameter in model.named_parameters():
        if any(marker in name.casefold() for marker in MODALITY_MARKERS): parameter.requires_grad = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.model.gradient_checkpointing)
    targets = discover_lora_modules(model, config.model.lora_target_regex)
    if resume_adapter:
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        lora = LoraConfig(r=config.model.lora_rank, lora_alpha=config.model.lora_alpha,
                          lora_dropout=config.model.lora_dropout, bias="none", task_type="CAUSAL_LM",
                          target_modules=_target_pattern(targets))
        model = get_peft_model(model, lora)
    report = trainable_parameter_report(model); report.update({"architecture": architecture, "matched_linear_modules": targets})
    return model, report
