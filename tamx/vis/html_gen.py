"""HTML visualization generator for TAM results.

This module provides functions to create interactive HTML reports for visualizing
Token Activation Maps and text relevance scores.
"""

import json
import base64
import os
import io
import uuid
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image


def to_pil_single(img: Any) -> Optional[Image.Image]:
    """Converts a single image tensor or array to a PIL Image.

    Args:
        img: Input image as tensor, array, or PIL Image.

    Returns:
        Converted PIL Image or None if conversion fails.
    """
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if torch.is_tensor(img):
        arr = img.detach().cpu()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):  # [C, H, W]
                arr = arr.permute(1, 2, 0)
        elif arr.ndim != 2:
            return None
        arr = arr.numpy()
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        return None

    if arr.dtype != np.uint8:
        max_val = float(arr.max()) if arr.size else 1.0
        if max_val <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
    return None


def to_pil_list(img: Any) -> List[Image.Image]:
    """Converts a sequence of images (like video frames) to a list of PIL Images.

    Args:
        img: Input images as list, tensor, or array.

    Returns:
        List of converted PIL Images.
    """
    if img is None:
        return []

    # Handle list of items (could be frames or nested tensors)
    if isinstance(img, (list, tuple)):
        all_pils = []
        for item in img:
            all_pils.extend(to_pil_list(item))
        return all_pils

    # Handle single image
    if isinstance(img, Image.Image):
        return [img]

    # Handle tensor
    if torch.is_tensor(img):
        arr = img.detach().cpu()
        if arr.ndim == 4:  # [T, C, H, W] - This is a video tensor
            pils = []
            for i in range(arr.shape[0]):
                # Process each frame as a 3D tensor
                pils.append(to_pil_single(arr[i]))
            return [p for p in pils if p is not None]

        # 2D or 3D tensor - treat as single image
        single = to_pil_single(img)
        return [single] if single is not None else []

    return []


def file_to_data_url(path: str) -> Optional[str]:
    """Converts an image file to a base64 data URL.

    Args:
        path: Path to the image file.

    Returns:
        Base64 data URL or None if file doesn't exist.
    """
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1][1:].lower()
    if ext == "jpg":
        ext = "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def pil_to_data_url(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 data URL.

    Args:
        img: The PIL Image.

    Returns:
        Base64 data URL.
    """
    buffer = io.BytesIO()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def pack_scores(arr: Optional[Union[np.ndarray, List[float]]]) -> Optional[str]:
    """Packs an array of scores into a compact transferable string.

    Encoding strategy:
      - Convert flattened values to little-endian float16.
      - Base64-encode the raw bytes.
      - Prefix with ``f16b64:`` for browser-side decoding.

    This is lossy (float32 -> float16), but substantially smaller than
    fixed-width scientific-notation strings.

    Args:
        arr: Array or list of scores.

    Returns:
        Encoded string, or ``None`` if input is ``None``.
    """
    if arr is None:
        return None
    if isinstance(arr, list):
        if not arr:
            return "f16b64:"
        arr = np.array(arr)
    if not hasattr(arr, "ravel"):
        return arr

    flat = np.asarray(arr).ravel().astype("<f2", copy=False)
    b64 = base64.b64encode(flat.tobytes()).decode("ascii")
    return f"f16b64:{b64}"

def collect_media_order(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Identifies the order and types of media mentioned in the conversation.

    Args:
        messages: List of conversation messages.

    Returns:
        List of dictionaries describing each media item found.
    """
    media = []
    for msg in messages:
        for part in msg.get("content", []):
            if part.get("type") == "image":
                media.append({"type": "image", "source": part.get("image")})
            elif part.get("type") == "video":
                media.append({"type": "video", "source": part.get("video")})
    return media


def build_image_data_urls(
    media_order: List[Dict[str, str]],
    image_inputs: Optional[List[Any]],
    video_inputs: Optional[List[Any]],
    use_local_media: bool = False,
    save_pil_fn: Optional[Any] = None,
    save_file_fn: Optional[Any] = None,
) -> List[str]:
    """Generates data URLs or local paths for media items.

    Args:
        media_order: List of media items from collect_media_order.
        image_inputs: Image inputs from processor.
        video_inputs: Video inputs from processor.
        use_local_media: If True, uses save_pil_fn and save_file_fn for local paths.
        save_pil_fn: Function to save a PIL Image and return its URL/path.
        save_file_fn: Function to handle an existing file and return its URL/path.

    Returns:
        List of data URLs or local paths.
    """
    image_iter = iter(image_inputs or [])
    video_iter = iter(video_inputs or [])
    data_urls = []

    def is_remote_or_data_url(s: Any) -> bool:
        if not isinstance(s, str):
            return False
        return s.startswith(("http://", "https://", "data:"))

    for item in media_order:
        source = item.get("source")
        if item["type"] == "image":
            # Always consume one from image_iter to keep in sync
            img = next(image_iter, None)
            
            # Prefer original source if it's a URL or exists locally
            if source and (is_remote_or_data_url(source) or os.path.exists(source)):
                if is_remote_or_data_url(source):
                    data_urls.append(source)
                elif use_local_media and save_file_fn:
                    data_urls.append(save_file_fn(source))
                else:
                    data_urls.append(file_to_data_url(source) or "")
            else:
                # Fallback to processed PIL image
                pil = to_pil_single(img)
                if pil is not None:
                    # For images without a source, we use data URLs to avoid saving to media folder
                    data_urls.append(pil_to_data_url(pil))
                else:
                    data_urls.append("")
        else:
            # For videos, we usually want the exploded frames (PILs)
            video_item = next(video_iter, None)
            pils = to_pil_list(video_item)
            if pils:
                for p in pils:
                    # Video frames are always saved as PILs
                    if use_local_media and save_pil_fn:
                        data_urls.append(save_pil_fn(p))
                    else:
                        data_urls.append(pil_to_data_url(p))
            else:
                # Fallback to original video source if no frames
                if source and (is_remote_or_data_url(source) or os.path.exists(source)):
                    if is_remote_or_data_url(source):
                        data_urls.append(source)
                    elif use_local_media and save_file_fn:
                        data_urls.append(save_file_fn(source))
                    else:
                        data_urls.append(file_to_data_url(source) or "")
                else:
                    data_urls.append("")
    return data_urls


def explode_video_blocks(tam_results: Dict[str, Any]) -> Dict[str, Any]:
    """Explands video blocks into individual frame blocks for easier visualization.

    This ensures that each frame of a video is treated as a separate visual entity
    in the interactive HTML report.

    Args:
        tam_results: The results dictionary from compute_tam.

    Returns:
        The updated results dictionary with exploded video blocks.
    """
    new_blocks = []
    block_map_indices = []  # mapping from original block index to a list of new block indices

    for i, block in enumerate(tam_results["vision_blocks"]):
        if block.get("type") == "video" and "grid" in block and len(block["grid"]) == 3:
            t, h, w = block["grid"]
            v_pos = block["v_pos"]
            n_tokens_per_frame = h * w
            indices = []
            for frame_idx in range(t):
                start_tok = frame_idx * n_tokens_per_frame
                end_tok = (frame_idx + 1) * n_tokens_per_frame
                new_block = block.copy()
                new_block["type"] = "image"
                new_block["grid"] = [h, w]
                new_block["v_pos"] = v_pos[start_tok:end_tok]
                indices.append(len(new_blocks))
                new_blocks.append(new_block)
            block_map_indices.append(indices)
        else:
            block_map_indices.append([len(new_blocks)])
            new_blocks.append(block)

    # Update turns to point to new block indices
    if "turns" in tam_results:
        for turn in tam_results["turns"]:
            if "vision_blocks" in turn:
                new_turn_vbs = []
                for vb_idx in turn["vision_blocks"]:
                    new_turn_vbs.extend(block_map_indices[vb_idx])
                turn["vision_blocks"] = new_turn_vbs

    # Now explode vision_maps and candidate_vision_maps
    def explode_maps(maps_list):
        if maps_list is None:
            return None
        new_maps_list = []
        for ans_maps in maps_list:
            new_ans_maps = []
            for i, block_map in enumerate(ans_maps):
                indices = block_map_indices[i]
                if len(indices) == 1:
                    new_ans_maps.append(block_map)
                else:
                    if block_map is None:
                        new_ans_maps.extend([None] * len(indices))
                    else:
                        # Split the flat map into frames
                        t = len(indices)
                        frames = np.array_split(block_map, t)
                        new_ans_maps.extend(frames)
            new_maps_list.append(new_ans_maps)
        return new_maps_list

    tam_results["vision_blocks"] = new_blocks
    tam_results["vision_maps"] = explode_maps(tam_results["vision_maps"])
    if "candidate_vision_maps" in tam_results:
        # candidate_vision_maps is List[List[List[np.ndarray]]] (ans_pos x cand x blocks)
        new_cand_maps = []
        for ans_cand_maps in tam_results["candidate_vision_maps"]:
            new_ans_cand_maps = []
            for cand_map_list in ans_cand_maps:
                # explode one candidate's map list across blocks
                exploded_list = []
                for i, block_map in enumerate(cand_map_list):
                    indices = block_map_indices[i]
                    if len(indices) == 1:
                        exploded_list.append(block_map)
                    else:
                        if block_map is None:
                            exploded_list.extend([None] * len(indices))
                        else:
                            t = len(indices)
                            frames = np.array_split(block_map, t)
                            exploded_list.extend(frames)
                new_ans_cand_maps.append(exploded_list)
            new_cand_maps.append(new_ans_cand_maps)
        tam_results["candidate_vision_maps"] = new_cand_maps

    return tam_results


def build_tam_html(
    tam_results: Dict[str, Any],
    processor: Any,
    image_paths: List[str],
    input_ids: List[int],
    image_data_urls: Optional[List[str]] = None,
    use_local_media: bool = False,
) -> str:
    """Builds an interactive HTML visualization for TAM results.

    Args:
        tam_results: Results from compute_tam.
        processor: The model processor/tokenizer.
        image_paths: Paths to original images/videos.
        input_ids: Token IDs in the sequence.
        image_data_urls: Optional pre-computed data URLs for media.
        use_local_media: If True, uses paths instead of embedding media in HTML.

    Returns:
        The generated HTML content as a string.
    """

    # 1. Prepare chat history from turns
    chat_turns = []
    ans_pos_list = tam_results["ans_pos"]
    ans_pos_set = set(ans_pos_list)
    ctx_text_pos_set = set(tam_results["ctx_text_pos"])

    orig_to_new_idx = {}
    display_tokens_count = 0

    for t_idx, turn in enumerate(tam_results["turns"]):
        turn_tokens = []

        # Combine text positions and vision block markers for this turn
        # We want to insert the vision block indicator at the correct relative position
        elements = []
        for p in turn["pos"]:
            elements.append(("text", p))
        for vb_idx in turn.get("vision_blocks", []):
            vb = tam_results["vision_blocks"][vb_idx]
            elements.append(("vision", vb["start"], vb_idx))

        # Sort elements by original position
        elements.sort(key=lambda x: x[1])

        for elem in elements:
            if elem[0] == "text":
                p = elem[1]
                tid = input_ids[p]
                text = processor.tokenizer.decode([tid])

                token_type = "other"
                ans_idx = -1
                if p in ans_pos_set:
                    token_type = "answer"
                    ans_idx = ans_pos_list.index(p)
                elif p in ctx_text_pos_set:
                    token_type = "prompt"

                new_idx = display_tokens_count
                orig_to_new_idx[p] = new_idx
                display_tokens_count += 1

                turn_tokens.append({
                    "text": text,
                    "type": token_type,
                    "ans_idx": ans_idx,
                    "orig_idx": p
                })
            else:
                # Vision block indicator
                vb_idx = elem[2]
                turn_tokens.append({
                    "text": f" [Image {vb_idx}] ",
                    "type": "vision-link",
                    "vision_idx": vb_idx
                })

        chat_turns.append({
            "role": turn["role"],
            "tokens": turn_tokens,
            "turn_idx": t_idx
        })

    # Update ans_pos based on new filtered indices
    new_ans_pos = [orig_to_new_idx[p] for p in ans_pos_list if p in orig_to_new_idx]

    # 2. Process Images to Data URLs
    images_data = []
    if image_data_urls is not None:
        for data_url in image_data_urls:
            if data_url:
                images_data.append({"data_url": data_url})
    else:
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Warning: Image path {path} not found.")
                continue
            if use_local_media:
                images_data.append({"data_url": path})
            else:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(path)[1][1:].lower()
                    if ext == "jpg":
                        ext = "jpeg"
                    images_data.append({
                        "data_url": f"data:image/{ext};base64,{b64}"
                    })

    # 3. Vision Blocks metadata
    vision_blocks_info = []
    for block in tam_results["vision_blocks"]:
        # Use the pre-computed grid if available
        grid = block.get("grid")
        if grid is not None:
            if len(grid) == 3:  # [T, H, W]
                grid_h, grid_w = grid[1], grid[2]
            else:  # [H, W]
                grid_h, grid_w = grid[0], grid[1]
        else:
            # Fallback to square-ish grid if grid info is missing
            n_tokens = len(block["v_pos"])
            side = int(np.sqrt(n_tokens))
            grid_h = side
            grid_w = n_tokens // side

        vision_blocks_info.append({
            "grid_h": grid_h,
            "grid_w": grid_w,
            "turn_idx": block.get("turn_idx", -1)
        })

    # 4. Prepare JSON Data
    serializable_vision_maps = []
    for ans_maps in tam_results["vision_maps"]:
        # ans_maps is List[np.ndarray] (one per vision block)
        serializable_vision_maps.append([pack_scores(v_map) for v_map in ans_maps])

    serializable_candidate_maps = []
    for ans_cand_maps in tam_results.get("candidate_vision_maps", []):
        # ans_cand_maps is List[List[np.ndarray]] (one per candidate, each has blocks)
        serializable_ans_cand_maps = []
        for cand_block_maps in ans_cand_maps:
            serializable_ans_cand_maps.append([pack_scores(v_map) for v_map in cand_block_maps])
        serializable_candidate_maps.append(serializable_ans_cand_maps)

    serializable_text_rel = [pack_scores(rel) for rel in tam_results["text_rel"]]

    serializable_candidate_text_rel = []
    for cand_list in tam_results.get("candidate_text_rel", []):
        # cand_list is List[np.ndarray] (one per candidate)
        serializable_candidate_text_rel.append([pack_scores(rel) for rel in cand_list])

    candidate_logits = tam_results.get("candidate_logits") or []

    max_text_score = 0
    for rel in tam_results["text_rel"]:
        if rel is not None and len(rel) > 0:
            max_text_score = max(max_text_score, float(np.max(rel)))
    for cand_step_rel in tam_results.get("candidate_text_rel", []):
        for rel in cand_step_rel:
            if rel is not None and len(rel) > 0:
                max_text_score = max(max_text_score, float(np.max(rel)))

    candidate_tokens = []
    candidate_token_ids = tam_results.get("candidate_token_ids")
    for i, ans_id in enumerate(tam_results["ans_tokens"]):
        tokens_for_ans = [processor.tokenizer.decode([ans_id])]
        if candidate_token_ids is not None and i < len(candidate_token_ids):
            for cand_id in candidate_token_ids[i]:
                tokens_for_ans.append(processor.tokenizer.decode([cand_id]))
        candidate_tokens.append(tokens_for_ans)

    data_json = {
        "chat_turns": chat_turns,
        "ans_pos": new_ans_pos,
        "images": images_data,
        "vision_blocks": vision_blocks_info,
        "vision_maps": serializable_vision_maps,
        "candidate_vision_maps": serializable_candidate_maps,
        "candidate_tokens": candidate_tokens,
        "candidate_text_rel": serializable_candidate_text_rel,
        "candidate_logits": candidate_logits,
        "text_rel": serializable_text_rel,
        "max_text_score": float(max_text_score)
    }

    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        return obj

    data_json = to_jsonable(data_json)

    # 5. Read CSS and JS
    vis_dir = os.path.dirname(__file__)
    with open(os.path.join(vis_dir, "styles.css"), "r") as f:
        css_content = f.read()
    with open(os.path.join(vis_dir, "renderer.js"), "r") as f:
        js_content = f.read()

    # 6. Build the final HTML
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TAM Interactive Visualization</title>
    <style>
{css_content}
    </style>
</head>
<body>
    <div id="tam-container"></div>
    <script>
        const tamData = {json.dumps(data_json)};
{js_content}
        document.addEventListener('DOMContentLoaded', () => {{
            new TAMVisualizer('tam-container', tamData);
        }});
    </script>
</body>
</html>
"""
    return html_template


def generate_tam_html(
    tam_results: Dict[str, Any],
    processor: Any,
    image_paths: List[str],
    input_ids: List[int],
    output_path: str = "tam_visualization.html",
    image_data_urls: Optional[List[str]] = None,
    use_local_media: bool = False,
):
    """
    Generates an interactive HTML visualization for TAM results.
    """
    html_template = build_tam_html(
        tam_results=tam_results,
        processor=processor,
        image_paths=image_paths,
        input_ids=input_ids,
        image_data_urls=image_data_urls,
        use_local_media=use_local_media,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"Visualization saved to {output_path}")
    return output_path
