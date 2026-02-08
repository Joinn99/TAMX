"""Sample script for generating TAM visualizations from pre-saved model data.

This script demonstrates how to load model hidden states and token IDs from
safetensors files, compute Token Activation Maps, and export an interactive
HTML visualization.
"""

import os
import torch
import json
import uuid
import shutil
from PIL import Image
from safetensors.torch import load_file


from tamx.core import compute_tam
from tamx.vis.html_gen import (
    build_tam_html,
    collect_media_order,
    build_image_data_urls,
    explode_video_blocks,
)

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def save_pil_to_local(img: Image.Image, output_dir: str) -> str:
    """Saves a PIL image to a local media directory and returns its relative path."""
    media_dir = os.path.join(output_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    filename = f"frame_{uuid.uuid4()}.jpg"
    path = os.path.join(media_dir, filename)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=75)
    return os.path.join("media", filename)


def save_file_to_local(source_path: str, output_dir: str) -> str:
    """Copies a media file to a local media directory and returns its relative path."""
    if not source_path or not os.path.exists(source_path):
        return ""
    media_dir = os.path.join(output_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    filename = os.path.basename(source_path)
    dest_path = os.path.join(media_dir, filename)
    if not os.path.exists(dest_path):
        shutil.copy2(source_path, dest_path)
    return os.path.join("media", filename)


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
    assert "Qwen3-VL" in model_path

    # Test cases with their data and messages
    test_cases = [
        {
            "name": "encode",
            "message": "asset/output/encode.json",
            "data": "asset/output/encode.safetensors",
            "output": "asset/output/encode_vis.html"
        },
        {
            "name": "generate",
            "message": "asset/output/generate.json",
            "data": "asset/output/generate.safetensors",
            "output": "asset/output/generate_vis.html"
        }
    ]

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # We still need the lm_head_weight for compute_tam
    print("Loading head weight...")
    model_class = Qwen3VLForConditionalGeneration if "Qwen3-VL" in model_path else Qwen2_5_VLForConditionalGeneration
    model = model_class.from_pretrained(model_path)
    head_weight = model.lm_head.weight.detach().to(torch.bfloat16)
    del model # Free memory

    import time
    start_time = time.time()
    for case in test_cases:
        if not os.path.exists(case["data"]):
            print(f"Skipping {case['name']}, data not found.")
            continue
        if not os.path.exists(case["message"]):
            print(f"Skipping {case['name']}, message file not found.")
            continue

        print(f"\n--- Processing {case['name']} ---")
        
        # 1. Load messages and process vision info
        with open(case["message"], "r") as f:
            messages = json.load(f)
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16 if "Qwen3-VL" in model_path else 14,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        # 2. Load TAM data
        data = load_file(case["data"])
        hidden_states = data["hidden_state"].to(torch.bfloat16)[0]
        input_ids = data["generated_ids"][0]
        candidate_token_ids = data.get("candidate_ids_per_step")
        image_grid_thw = data.get("image_grid_thw")
        video_grid_thw = data.get("video_grid_thw")

        print("Computing TAM...")
        results = compute_tam(
            hidden_states=hidden_states,
            input_ids=input_ids,
            lm_head_weight=head_weight,
            candidate_token_ids=candidate_token_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            apply_filter=True,
            apply_eci=True,
            kernel_size=5,
        )

        # 3. Generate data URLs or local paths for images/videos
        media_order = collect_media_order(messages)
        use_local_media = True # Set to True to avoid large HTML files
        output_dir = os.path.dirname(case["output"])
        os.makedirs(output_dir, exist_ok=True)
        
        image_data_urls = build_image_data_urls(
            media_order, image_inputs, video_inputs, 
            use_local_media=use_local_media,
            save_pil_fn=lambda img: save_pil_to_local(img, output_dir),
            save_file_fn=lambda path: save_file_to_local(path, output_dir)
        )

        # 4. Explode video blocks
        results = explode_video_blocks(results)

        print("Generating HTML...")
        html = build_tam_html(
            tam_results=results,
            processor=processor,
            image_paths=[],
            input_ids=input_ids.tolist(),
            image_data_urls=image_data_urls,
            use_local_media=use_local_media,
        )
        
        with open(case["output"], "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved to {case['output']}")

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")


