"""FastAPI server for TAM visualization.

This module provides a web interface to interactively generate and view
Token Activation Maps using a live MLLM.
"""

import base64
import io
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from tamx.core import compute_tam
from tamx.vis.html_gen import (
    build_tam_html,
    collect_media_order,
    build_image_data_urls,
    explode_video_blocks,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMP_MEDIA_DIR = os.path.join(REPO_ROOT, "temp_media")
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")


class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1, le=50)


class VisualizationConfig(BaseModel):
    """Configuration for TAM visualization."""
    apply_filter: bool = True
    apply_eci: bool = True
    kernel_size: int = 3
    use_local_media: bool = False


class VisualizationRequest(BaseModel):
    messages: List[Dict[str, Any]]
    generation: Optional[GenerationConfig] = None
    visualization: Optional[VisualizationConfig] = None


class ModelManager:
    """Manages the lifecycle and loading of the MLLM and its processor."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.lm_head_weight = None

    def load(self) -> Tuple[Any, Any, torch.Tensor]:
        """Loads the model, processor, and LM head weights if not already loaded.

        Returns:
            Tuple of (model, processor, lm_head_weight).
        """
        if self.model is None:
            model_class = Qwen3VLForConditionalGeneration if "Qwen3-VL" in self.model_path else Qwen2_5_VLForConditionalGeneration
            self.model = model_class.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.lm_head_weight = self.model.get_input_embeddings().weight
        return self.model, self.processor, self.lm_head_weight


model_manager = ModelManager(DEFAULT_MODEL_PATH)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _resolve_local_path(raw_path: Optional[str]) -> Optional[str]:
    if not raw_path or not isinstance(raw_path, str):
        return None
    if raw_path.startswith("http://") or raw_path.startswith("https://"):
        return raw_path
    if raw_path.startswith("data:"):
        return raw_path
    path = raw_path if os.path.isabs(raw_path) else os.path.join(REPO_ROOT, raw_path)
    path = os.path.abspath(path)
    if not path.startswith(REPO_ROOT):
        return None
    return path


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for msg in messages:
        role = msg.get("role")
        if not role:
            raise ValueError("Each message must include a role.")
        content = msg.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError("Message content must be a list or string.")
        cleaned_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            p_type = part.get("type")
            if p_type == "text":
                cleaned_parts.append({"type": "text", "text": part.get("text", "")})
            elif p_type in ("image", "image_url"):
                img_key = "image" if p_type == "image" else "image_url"
                raw_source = part.get(img_key)
                source = _resolve_local_path(raw_source)
                if not source:
                    if raw_source and (raw_source.startswith("http://") or raw_source.startswith("https://")):
                         source = raw_source
                    else:
                        raise ValueError(f"{img_key} path must be a local file path or URL.")
                cleaned_parts.append({"type": "image", "image": source})
            elif p_type == "video":
                raw_source = part.get("video")
                source = _resolve_local_path(raw_source)
                if not source:
                    if raw_source and (raw_source.startswith("http://") or raw_source.startswith("https://")):
                        source = raw_source
                    else:
                        raise ValueError("Video path must be a local file path or URL.")
                cleaned_parts.append({"type": "video", "video": source})
        normalized.append({"role": role, "content": cleaned_parts})
    return normalized


def _pil_to_local_url(img: Image.Image) -> str:
    os.makedirs(TEMP_MEDIA_DIR, exist_ok=True)
    filename = f"frame_{uuid.uuid4()}.jpg"
    path = os.path.join(TEMP_MEDIA_DIR, filename)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=75)
    return f"/api/media?path={path}"


def _file_to_local_url(path: str) -> str:
    if not path:
        return ""
    if path.startswith(("http://", "https://", "data:")):
        return path
    if not os.path.exists(path):
        return ""
    return f"/api/media?path={os.path.abspath(path)}"


def _align_hidden_states(hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    target_len = int(input_ids.shape[0]) - 1
    if hidden_states.shape[1] > target_len:
        hidden_states = hidden_states[:, :target_len, :]
    if hidden_states.shape[1] < target_len:
        raise ValueError("Hidden states length is shorter than expected.")
    return hidden_states


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/media")
def media(path: str):
    resolved = _resolve_local_path(path)
    if not resolved or not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail="Media not found.")
    return FileResponse(resolved)


@app.post("/api/visualize")
def visualize(req: VisualizationRequest):
    #try:
    messages = req.messages
    #except ValueError as exc:
    #    raise HTTPException(status_code=400, detail=str(exc))

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    last_role = messages[-1]["role"]
    generate_answer = last_role == "user"

    model, processor, lm_head_weight = model_manager.load()

    add_generation_prompt = generate_answer
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16 if "Qwen3-VL" in model_manager.model_path else 14,
        return_video_metadata=True,
        return_video_kwargs=True,
    )
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        do_resize=False,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    generation_cfg = req.generation or GenerationConfig()
    vis_cfg = req.visualization or VisualizationConfig()

    if generate_answer:
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_cfg.max_new_tokens,
            temperature=generation_cfg.temperature,
            top_p=generation_cfg.top_p,
            top_k=generation_cfg.top_k,
            use_cache=True,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        generated_ids = outputs.sequences
        hidden_states = torch.cat([feats[-1] for feats in outputs.hidden_states], dim=1)
        hidden_states = _align_hidden_states(hidden_states, generated_ids[0])
        candidate_ids_per_step = torch.cat(
            [step_scores[0].topk(generation_cfg.top_k).indices.unsqueeze(0) for step_scores in outputs.scores],
            dim=0,
        )
        input_ids = generated_ids[0]
        generated_text = processor.decode(
            generated_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=1.0,
            top_p=1.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        hidden_states = torch.cat([feats[-1] for feats in outputs.hidden_states], dim=1)
        hidden_states = _align_hidden_states(hidden_states, inputs["input_ids"][0])
        candidate_ids_per_step = None
        input_ids = inputs["input_ids"][0]
        generated_text = None

    tam_results = compute_tam(
        hidden_states=hidden_states[0],
        input_ids=input_ids,
        lm_head_weight=lm_head_weight.to(hidden_states.dtype),
        candidate_token_ids=candidate_ids_per_step,
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        apply_filter=vis_cfg.apply_filter,
        apply_eci=vis_cfg.apply_eci,
        kernel_size=vis_cfg.kernel_size,
    )

    media_order = collect_media_order(messages)
    image_data_urls = build_image_data_urls(
        media_order,
        image_inputs,
        video_inputs,
        use_local_media=vis_cfg.use_local_media,
        save_pil_fn=_pil_to_local_url,
        save_file_fn=_file_to_local_url,
    )

    tam_results = explode_video_blocks(tam_results)

    html = build_tam_html(
        tam_results=tam_results,
        processor=processor,
        image_paths=[],
        input_ids=input_ids.tolist(),
        image_data_urls=image_data_urls,
    )

    return JSONResponse(
        {
            "mode": "generate" if generate_answer else "encode",
            "generated_text": generated_text,
            "html": html,
        }
    )
