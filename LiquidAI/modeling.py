"""Base-model loading, LoRA adapters and shared answer-logit operations."""

from __future__ import annotations

import json
from pathlib import Path

from .config import LABELS, LOGGER
from .encoding import DecisionCodec
from .prompts import STANCE_QUESTION, EVIDENCE_INSTRUCTIONS


def pool_options(vocab_logits, token_groups):
    """Differentiable equivalent of d1 prompt.readout's max over token aliases."""
    import torch

    return torch.stack([vocab_logits[..., ids].amax(dim=-1) for ids in token_groups], dim=-1)


def forward_answer_logits(model, input_ids, attention_mask, answer_positions):
    import torch

    # Only project the requested answer positions, never [B, all_tokens, vocab].
    # The per-row diagonal selects each question's own final non-padding token.
    output = model(
        input_ids=input_ids, attention_mask=attention_mask,
        logits_to_keep=answer_positions, use_cache=False, return_dict=True,
    )
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    return output.logits[rows, rows].float()


def load_train_model(args):
    """Load a native d1 base model and tokenizer before attaching LoRA."""
    import torch
    from transformers import AutoModel, AutoTokenizer, set_seed

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if dtype_name == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use --dtype float32")
    dtype = getattr(torch, dtype_name)
    set_seed(args.seed)
    LOGGER.info("Loading %s on %s (%s)", args.model, device, dtype_name)
    load_kwargs = {"trust_remote_code": True, "local_files_only": args.local_files_only}
    if args.revision:
        load_kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    base_model = AutoModel.from_pretrained(args.model, dtype=dtype, **load_kwargs)
    if not callable(getattr(base_model, "system_one", None)):
        raise ValueError("Expected a d1 checkpoint loaded with its native custom code")
    context_limit = base_model.config.text_config.max_position_embeddings
    if args.max_length > context_limit:
        raise ValueError(f"--max-length {args.max_length} exceeds model context limit {context_limit}")
    base_model.config.use_cache = False
    base_model.config.text_config.use_cache = False
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = base_model.config.text_config.pad_token_id
    if pad_token_id is None:
        raise ValueError("Checkpoint has no pad token")

    return base_model, tokenizer, pad_token_id, device, dtype_name


def attach_lora(base_model, args):
    import torch
    from peft import LoraConfig, get_peft_model

    targets = [
        name for name, module in base_model.named_modules()
        if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise ValueError("No d1 language-model linear layers found; verify the checkpoint architecture")
    # Exact paths exclude vision/projector layers and the frozen LM readout.
    config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, bias="none",
    )
    model = get_peft_model(base_model, config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return model


def save_adapter(model, tokenizer, destination: Path, metadata: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    (destination / "training_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def load_test_model(args):
    adapter = args.adapter_dir
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No trained LoRA adapter at {adapter}. Finish training first, or pass "
            "--adapter-dir pointing to a final/checkpoint-step-* directory containing adapter_config.json."
        )
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    metadata_path = adapter / "training_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    for key, expected in (
        ("labels", list(LABELS)), ("stance_question", STANCE_QUESTION),
        ("evidence_instructions", EVIDENCE_INSTRUCTIONS),
    ):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"Adapter training {key} differs from this script; use the matching training code")
    base_id = args.model or metadata.get("base_model") or adapter_config.get("base_model_name_or_path")
    if not base_id:
        raise ValueError("Cannot determine adapter's base model; specify --model")
    revision = args.revision
    if revision is None and not args.model:
        revision = metadata.get("resolved_revision") or metadata.get("arguments", {}).get("revision") or adapter_config.get("revision")

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype_name = args.dtype
    if dtype_name == "auto":
        saved_dtype = metadata.get("dtype") or metadata.get("arguments", {}).get("dtype")
        if device.type == "cpu":
            dtype_name = "float32"
        elif saved_dtype in ("float32", "bfloat16"):
            dtype_name = saved_dtype
        else:
            dtype_name = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if dtype_name == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use --dtype float32")
    load_kwargs = {"trust_remote_code": True, "local_files_only": args.local_files_only}
    if revision:
        load_kwargs["revision"] = revision
    LOGGER.info("Loading base %s and adapter %s on %s (%s)", base_id, adapter, device, dtype_name)
    tokenizer_source = str(adapter) if (adapter / "tokenizer_config.json").is_file() else base_id
    tokenizer_kwargs = dict(load_kwargs)
    if tokenizer_source == str(adapter):
        tokenizer_kwargs.pop("revision", None)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    base_model = AutoModel.from_pretrained(base_id, dtype=getattr(torch, dtype_name), **load_kwargs)
    if not callable(getattr(base_model, "system_one", None)):
        raise ValueError("Expected a native d1 checkpoint")
    context_limit = base_model.config.text_config.max_position_embeddings
    if args.max_length > context_limit:
        raise ValueError(f"--max-length {args.max_length} exceeds model context limit {context_limit}")
    codec = DecisionCodec(base_model, tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = base_model.config.text_config.pad_token_id
    if pad_token_id is None:
        raise ValueError("Checkpoint has no pad token")
    base_model.config.use_cache = False
    base_model.config.text_config.use_cache = False
    model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False, local_files_only=args.local_files_only)
    model.to(device).eval()
    model.requires_grad_(False)
    info = {
        "base_model": base_id, "adapter_dir": str(adapter.resolve()),
        "resolved_revision": getattr(base_model.config, "_commit_hash", None),
        "dtype": dtype_name, "device": str(device),
    }
    return model, codec, pad_token_id, device, info
