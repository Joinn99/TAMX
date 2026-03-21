"""Helpers for loading and adapting Qwen multimodal models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


SUPPORTED_QWEN_MODEL_TYPES = {
    "qwen2_5_vl": "qwen2_5_vl",
    "qwen3_vl": "qwen3_vl",
    "qwen3_5": "qwen3_5",
}


@dataclass(frozen=True)
class QwenSpecialTokenIds:
    """Tokenizer-dependent special token ids used by TAM parsing."""

    im_start_id: int
    im_end_id: int
    vision_start_id: int
    vision_end_id: int
    image_pad_id: int
    video_pad_id: int
    assistant_id: int
    user_id: int
    system_id: int
    newline_id: int = 198
    double_newline_id: Optional[int] = None
    think_start_id: Optional[int] = None
    think_end_id: Optional[int] = None

    @property
    def special_ids_set(self) -> set[int]:
        special_ids = {
            self.im_start_id,
            self.im_end_id,
            self.vision_start_id,
            self.vision_end_id,
            self.image_pad_id,
            self.video_pad_id,
            self.assistant_id,
            self.user_id,
            self.system_id,
        }
        if self.think_start_id is not None:
            special_ids.add(self.think_start_id)
        if self.think_end_id is not None:
            special_ids.add(self.think_end_id)
        return special_ids

    def to_dict(self) -> Dict[str, Optional[int]]:
        return asdict(self)


@dataclass(frozen=True)
class QwenModelInfo:
    """Resolved model metadata needed by TAM."""

    model_family: str
    model_type: str
    image_patch_size: int
    special_token_ids: QwenSpecialTokenIds


def resolve_torch_dtype(dtype_name: str):
    """Translate a user-facing dtype string into a torch dtype."""
    name = (dtype_name or "auto").strip().lower()
    if name in {"auto", ""}:
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[name]


def _encode_single_token_id(tokenizer: Any, token_text: str) -> Optional[int]:
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _normalize_special_token_ids(
    special_token_ids: Optional[Mapping[str, Optional[int]] | QwenSpecialTokenIds],
) -> Optional[QwenSpecialTokenIds]:
    if special_token_ids is None:
        return None
    if isinstance(special_token_ids, QwenSpecialTokenIds):
        return special_token_ids
    return QwenSpecialTokenIds(**dict(special_token_ids))


def get_qwen_model_info(
    *,
    model_path: Optional[str] = None,
    config: Optional[Any] = None,
    processor: Optional[Any] = None,
) -> QwenModelInfo:
    """Resolve model-family-specific settings from config and tokenizer."""
    if config is None:
        if model_path is None:
            raise ValueError("Either model_path or config must be provided.")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if processor is None:
        if model_path is None:
            raise ValueError("Either model_path or processor must be provided.")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type not in SUPPORTED_QWEN_MODEL_TYPES:
        raise ValueError(f"Unsupported Qwen model_type for TAMX: {model_type}")

    tokenizer = getattr(processor, "tokenizer", processor)
    special_token_ids = QwenSpecialTokenIds(
        im_start_id=_encode_single_token_id(tokenizer, "<|im_start|>"),
        im_end_id=_encode_single_token_id(tokenizer, "<|im_end|>"),
        vision_start_id=int(getattr(config, "vision_start_token_id")),
        vision_end_id=int(getattr(config, "vision_end_token_id")),
        image_pad_id=int(getattr(config, "image_token_id")),
        video_pad_id=int(getattr(config, "video_token_id")),
        assistant_id=_encode_single_token_id(tokenizer, "assistant"),
        user_id=_encode_single_token_id(tokenizer, "user"),
        system_id=_encode_single_token_id(tokenizer, "system"),
        newline_id=_encode_single_token_id(tokenizer, "\n") or 198,
        double_newline_id=_encode_single_token_id(tokenizer, "\n\n"),
        think_start_id=_encode_single_token_id(tokenizer, "<think>"),
        think_end_id=_encode_single_token_id(tokenizer, "</think>"),
    )

    missing = [
        name
        for name, value in special_token_ids.to_dict().items()
        if name not in {"think_start_id", "think_end_id"} and value is None
    ]
    if missing:
        raise ValueError(
            f"Failed to resolve tokenizer special token ids for {model_type}: {', '.join(missing)}"
        )

    image_patch_size = int(getattr(getattr(config, "vision_config", None), "patch_size", 14))
    return QwenModelInfo(
        model_family=SUPPORTED_QWEN_MODEL_TYPES[model_type],
        model_type=model_type,
        image_patch_size=image_patch_size,
        special_token_ids=special_token_ids,
    )


def load_qwen_model_bundle(
    model_path: str,
    *,
    torch_dtype: Any = "auto",
    device_map: Any = "auto",
    local_files_only: bool = False,
) -> Tuple[Any, Any, QwenModelInfo]:
    """Load model, processor, and resolved model metadata."""
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model_info = get_qwen_model_info(config=model.config, processor=processor)
    return model, processor, model_info


def apply_chat_template(
    processor: Any,
    messages: Any,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    """Build a chat prompt while keeping Qwen3.5 in non-thinking mode by default."""
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def process_qwen_vision_info(
    messages: Any,
    model_info: QwenModelInfo,
) -> Tuple[Optional[list[Any]], Optional[list[Any]], Optional[list[Any]], Dict[str, Any]]:
    """Run qwen_vl_utils with the correct image patch size and unpack video metadata."""
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=model_info.image_patch_size,
        return_video_metadata=True,
        return_video_kwargs=True,
    )

    if video_inputs is None:
        return image_inputs, None, None, video_kwargs or {}

    unpacked_videos = []
    video_metadatas = []
    for item in video_inputs:
        if isinstance(item, tuple) and len(item) == 2:
            video_value, video_metadata = item
        else:
            video_value, video_metadata = item, None
        unpacked_videos.append(video_value)
        video_metadatas.append(video_metadata)

    if not any(metadata is not None for metadata in video_metadatas):
        video_metadatas = None

    return image_inputs, unpacked_videos, video_metadatas, video_kwargs or {}


def build_qwen_inputs(
    processor: Any,
    model_info: QwenModelInfo,
    messages: Any,
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
    return_tensors: str = "pt",
    padding: bool = True,
    do_resize: bool = False,
    device: Optional[torch.device | str] = None,
) -> Tuple[str, Any, Optional[list[Any]], Optional[list[Any]], Optional[list[Any]], Dict[str, Any]]:
    """Construct tokenizer + vision inputs for Qwen multimodal generation."""
    text = apply_chat_template(
        processor,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    image_inputs, video_inputs, video_metadatas, video_kwargs = process_qwen_vision_info(
        messages,
        model_info,
    )
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        padding=padding,
        do_resize=do_resize,
        return_tensors=return_tensors,
        **video_kwargs,
    )
    if device is not None:
        inputs = inputs.to(device)
    return text, inputs, image_inputs, video_inputs, video_metadatas, video_kwargs


def get_lm_head_weight(model: Any) -> torch.Tensor:
    """Return the projection weight used for token logits."""
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    return model.get_input_embeddings().weight


def find_final_norm(model: Any) -> Any:
    """Locate the final language-model normalization layer across Qwen variants."""
    candidate_paths = (
        "model.language_model.norm",
        "model.norm",
        "language_model.norm",
        "norm",
    )
    for path in candidate_paths:
        current = model
        valid = True
        for attr in path.split("."):
            if not hasattr(current, attr):
                valid = False
                break
            current = getattr(current, attr)
        if valid:
            return current
    raise AttributeError(f"Cannot locate final norm for model type {type(model).__name__}")


def normalize_special_token_ids(
    special_token_ids: Optional[Mapping[str, Optional[int]] | QwenSpecialTokenIds],
) -> Optional[QwenSpecialTokenIds]:
    """Public wrapper used by parsing functions."""
    return _normalize_special_token_ids(special_token_ids)
