"""Low-level computation functions for Token Activation Maps.

This module contains functions for grid assignment, weight computation,
activation extraction, and relevance calculation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .filters import apply_filter_to_map


def least_squares(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Find the scalar that minimizes the squared difference between map1 and scalar * map2.
    """
    def objective(x: float, map1: np.ndarray, map2: np.ndarray) -> float:
        return np.sum((map1 - map2 * x) ** 2)

    if np.sum(map2 ** 2) < 1e-12:
        return 0.0

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, args=(map1, map2))
    return float(result.x)


def normalize_scores(
    img_scores: np.ndarray,
    txt_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize image and text scores together for comparability.

    Args:
        img_scores: Flattened array of vision activation scores.
        txt_scores: Flattened array of text relevance scores.

    Returns:
        Tuple containing (normalized_img_scores, normalized_txt_scores).
    """
    all_scores = np.concatenate([img_scores, txt_scores], 0)
    score_min = all_scores.min()
    score_max = all_scores.max()

    if score_max > score_min:
        all_scores = (all_scores - score_min) / (score_max - score_min)
    else:
        all_scores = np.zeros_like(all_scores)

    img_scores_norm = all_scores[:len(img_scores)]
    txt_scores_norm = all_scores[len(img_scores):]

    return img_scores_norm, txt_scores_norm


def assign_grids(
    vision_blocks: List[Dict],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: Optional[int],
    temporal_merge_size: Optional[int]
) -> None:
    """Assign grid dimensions to each vision block.

    Args:
        vision_blocks: List of vision block metadata.
        image_grid_thw: Raw grid dimensions for images.
        video_grid_thw: Raw grid dimensions for videos.
        spatial_merge_size: Known spatial pooling factor.
        temporal_merge_size: Known temporal pooling factor.
    """
    def infer_merge_factors(n_tokens: int, grid_thw: List[int]) -> Tuple[int, int]:
        t_raw, h_raw, w_raw = grid_thw
        if n_tokens <= 0 or h_raw <= 0 or w_raw <= 0 or t_raw <= 0:
            return 1, 1

        candidates = []
        for s in range(1, max(h_raw, w_raw) + 1):
            if h_raw % s != 0 or w_raw % s != 0:
                continue
            t_eff = n_tokens * (s * s) / (h_raw * w_raw)
            if abs(t_eff - round(t_eff)) > 1e-6:
                continue
            t_eff = int(round(t_eff))
            if t_eff <= 0:
                continue
            if t_raw % t_eff != 0:
                continue
            temporal = max(1, t_raw // t_eff)
            candidates.append((s, temporal))

        if not candidates:
            return 1, 1
        # Prefer the smallest merge that satisfies the grid
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0]

    img_count = 0
    vid_count = 0
    for block in vision_blocks:
        if block["type"] == "image":
            if image_grid_thw is not None and img_count < image_grid_thw.shape[0]:
                grid = image_grid_thw[img_count].tolist()
                if spatial_merge_size is None or spatial_merge_size <= 0:
                    spatial, temporal = infer_merge_factors(len(block["v_pos"]), grid)
                else:
                    spatial = spatial_merge_size
                    temporal = temporal_merge_size if temporal_merge_size is not None else 1
                block["grid"] = [
                    max(1, grid[0] // max(1, temporal)),
                    grid[1] // max(1, spatial),
                    grid[2] // max(1, spatial)
                ]
            img_count += 1
        else:
            if video_grid_thw is not None and vid_count < video_grid_thw.shape[0]:
                grid = video_grid_thw[vid_count].tolist()
                if spatial_merge_size is None or spatial_merge_size <= 0:
                    spatial, temporal = infer_merge_factors(len(block["v_pos"]), grid)
                else:
                    spatial = spatial_merge_size
                    temporal = temporal_merge_size if temporal_merge_size is not None else 1
                block["grid"] = [
                    max(1, grid[0] // max(1, temporal)),
                    grid[1] // max(1, spatial),
                    grid[2] // max(1, spatial)
                ]
            vid_count += 1


def compute_unique_weights(
    ids: List[int],
    candidate_token_ids: Optional[torch.Tensor],
    lm_head_weight: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Extract and move unique token weights from the LM head to device.

    Args:
        ids: Token IDs in the sequence.
        candidate_token_ids: Optional candidate token IDs.
        lm_head_weight: Full weights of the LM head.
        device: Target device.
        dtype: Target data type.

    Returns:
        Tuple containing:
        - w_unique: Tensor of weights for unique tokens.
        - id_to_idx: Mapping from token ID to its index in w_unique.
    """
    unique_ids = sorted(list(set(ids)))
    if candidate_token_ids is not None:
        unique_ids = sorted(list(set(unique_ids) | set(candidate_token_ids.flatten().tolist())))

    w_unique = lm_head_weight[unique_ids].to(device=device, dtype=dtype)
    id_to_idx = {tid: i for i, tid in enumerate(unique_ids)}
    return w_unique, id_to_idx


def compute_vision_activations(
    hidden_states: torch.Tensor,
    vision_blocks: List[Dict],
    w_unique: torch.Tensor
) -> List[np.ndarray]:
    """Compute raw vision activations for all tokens.

    Args:
        hidden_states: Model hidden states.
        vision_blocks: List of vision blocks.
        w_unique: Weights for unique tokens.

    Returns:
        List of activation arrays, one per vision block.
    """
    vision_activations = []
    for block in vision_blocks:
        v_m = hidden_states[block["v_pos"]]
        a_m_all = torch.matmul(v_m, w_unique.t())
        vision_activations.append(F.relu(a_m_all).detach().cpu().float().numpy())
    return vision_activations


def compute_text_relevance(
    hidden_states: torch.Tensor,
    ctx_text_pos: List[int],
    ans_pos: List[int],
    ids: List[int],
    w_unique: torch.Tensor,
    id_to_idx: Dict[int, int]
) -> Tuple[np.ndarray, List[np.ndarray], Dict[int, int]]:
    """Compute text relevance scores for context tokens.

    Args:
        hidden_states: Model hidden states.
        ctx_text_pos: Positions of context text tokens.
        ans_pos: Positions of answer tokens.
        ids: Token IDs.
        w_unique: Weights for unique tokens.
        id_to_idx: Token ID to weight index map.

    Returns:
        Tuple containing:
        - r_full: Full relevance matrix.
        - text_rel_list: List of relevance scores per answer token.
        - pos_to_ctx_idx: Position to context index map.
    """
    ctx_all_pos = [p for p in (ctx_text_pos + ans_pos) if p < len(hidden_states)]
    h_ctx_all = hidden_states[ctx_all_pos]

    ans_token_weights = w_unique[[id_to_idx[ids[p]] for p in ans_pos]]
    r_full = torch.matmul(h_ctx_all, ans_token_weights.t())
    r_full = F.relu(r_full).detach().cpu().float().numpy()

    pos_to_ctx_idx = {p: i for i, p in enumerate(ctx_all_pos)}
    text_rel_list = []
    for i, ans_p in enumerate(ans_pos):
        current_ctx_indices = [pos_to_ctx_idx[p] for p in ctx_all_pos if p < ans_p]
        text_rel_list.append(r_full[current_ctx_indices, i])

    return r_full, text_rel_list, pos_to_ctx_idx


def compute_candidate_text_relevance(
    hidden_states: torch.Tensor,
    ctx_text_pos: List[int],
    ans_pos: List[int],
    ids: List[int],
    w_unique: torch.Tensor,
    id_to_idx: Dict[int, int],
    candidate_token_ids: Optional[torch.Tensor]
) -> List[List[np.ndarray]]:
    if candidate_token_ids is None:
        return []

    ctx_all_pos = [p for p in (ctx_text_pos + ans_pos) if p < len(hidden_states)]
    h_ctx_all = hidden_states[ctx_all_pos]
    pos_to_ctx_idx = {p: i for i, p in enumerate(ctx_all_pos)}

    candidate_text_rel = []
    for i, ans_p in enumerate(ans_pos):
        if i >= candidate_token_ids.shape[0]:
            candidate_text_rel.append([])
            continue
        cand_ids = candidate_token_ids[i].tolist()
        if not cand_ids:
            candidate_text_rel.append([])
            continue

        cand_weights = w_unique[[id_to_idx[cid] for cid in cand_ids]]
        r_full = torch.matmul(h_ctx_all, cand_weights.t())
        r_full = F.relu(r_full).detach().cpu().float().numpy()

        current_ctx_indices = [pos_to_ctx_idx[p] for p in ctx_all_pos if p < ans_p]
        cand_rel = [r_full[current_ctx_indices, j] for j in range(len(cand_ids))]
        candidate_text_rel.append(cand_rel)

    return candidate_text_rel


def compute_candidate_logits(
    hidden_states: torch.Tensor,
    ans_pos: List[int],
    ids: List[int],
    lm_head_weight: torch.Tensor,
    candidate_token_ids: Optional[torch.Tensor]
) -> List[List[Optional[float]]]:
    logits_per_answer = []
    for i, ans_p in enumerate(ans_pos):
        step_idx = ans_p - 1
        tokens = [ids[ans_p]]
        if candidate_token_ids is not None and i < candidate_token_ids.shape[0]:
            tokens.extend(candidate_token_ids[i].tolist())

        if step_idx < 0 or step_idx >= hidden_states.shape[0]:
            logits_per_answer.append([None for _ in tokens])
            continue

        h = hidden_states[step_idx]  # [d]
        token_weights = lm_head_weight[tokens].to(device=h.device, dtype=h.dtype)  # [K, d]
        logits = torch.matmul(token_weights, h).detach().cpu().float().tolist()
        logits_per_answer.append(logits)

    return logits_per_answer


def compute_vision_maps(
    vision_blocks: List[Dict],
    ans_pos: List[int],
    ids: List[int],
    candidate_token_ids: Optional[torch.Tensor],
    id_to_idx: Dict[int, int],
    vision_activations: List[np.ndarray],
    ctx_text_pos: List[int],
    r_full: np.ndarray,
    pos_to_ctx_idx: Dict[int, int],
    apply_eci: bool,
    apply_filter: bool,
    kernel_size: int
) -> Tuple[List[List[Optional[np.ndarray]]], List[List[np.ndarray]]]:
    all_vision_maps = []
    all_candidate_vision_maps = []

    for i, ans_p in enumerate(ans_pos):
        tokens_to_explain = [ids[ans_p]]
        if candidate_token_ids is not None and i < candidate_token_ids.shape[0]:
            tokens_to_explain.extend(candidate_token_ids[i].tolist())

        token_maps_at_pos = []
        for tok_id in tokens_to_explain:
            tok_idx = id_to_idx[tok_id]
            ans_token_map_per_block = []

            for m, block in enumerate(vision_blocks):
                if block["end"] >= ans_p:
                    ans_token_map_per_block.append(None)
                    continue

                a_m_i = vision_activations[m][:, tok_idx]

                if not apply_eci:
                    tilde_a = a_m_i
                else:
                    ctx_m_i_pos = [p for p in ctx_text_pos if block["end"] < p < ans_p]
                    ctx_m_i_pos += [p for p in ans_pos[:i] if block["end"] < p < ans_p]
                    ctx_m_i_pos = [p for p in ctx_m_i_pos if ids[p] != tok_id]

                    if not ctx_m_i_pos:
                        tilde_a = a_m_i
                    else:
                        r_weights = np.array([r_full[pos_to_ctx_idx[p], i] for p in ctx_m_i_pos])
                        r_sum = r_weights.sum()
                        if r_sum > 1e-8:
                            r_weights = r_weights / r_sum
                        else:
                            r_weights = np.ones_like(r_weights) / len(r_weights)

                        ctx_indices = [id_to_idx[ids[p]] for p in ctx_m_i_pos]
                        a_m_ctx = vision_activations[m][:, ctx_indices]
                        e_m_i = np.dot(a_m_ctx, r_weights)

                        scale = least_squares(a_m_i, e_m_i)
                        tilde_a = np.maximum(0, a_m_i - scale * e_m_i)

                if apply_filter:
                    tilde_a = apply_filter_to_map(tilde_a, block.get("grid"), kernel_size)

                ans_token_map_per_block.append(tilde_a)

            token_maps_at_pos.append(ans_token_map_per_block)

        all_vision_maps.append(token_maps_at_pos[0])
        if len(token_maps_at_pos) > 1:
            all_candidate_vision_maps.append(token_maps_at_pos[1:])
        else:
            all_candidate_vision_maps.append([])

    return all_vision_maps, all_candidate_vision_maps


def normalize_answer_scores(
    text_rel_list: List[np.ndarray],
    vision_maps: List[List[Optional[np.ndarray]]],
    candidate_text_rel: Optional[List[List[np.ndarray]]] = None,
    candidate_vision_maps: Optional[List[List[List[Optional[np.ndarray]]]]] = None
) -> Tuple[List[np.ndarray], List[List[Optional[np.ndarray]]], List[List[np.ndarray]], List[List[List[Optional[np.ndarray]]]]]:
    normalized_text = []
    normalized_vision = []
    normalized_cand_text = []
    normalized_cand_vision = []

    for i in range(len(text_rel_list)):
        text_scores = text_rel_list[i]
        block_maps = vision_maps[i] if i < len(vision_maps) else []
        
        # 收集主回答的所有分数
        all_img_parts = []
        img_shapes = []
        for v_map in block_maps:
            if v_map is not None:
                all_img_parts.append(v_map.flatten())
                img_shapes.append(v_map.shape)
            else:
                img_shapes.append(None)
        
        # 收集候选者的分数
        cand_text_list = candidate_text_rel[i] if (candidate_text_rel and i < len(candidate_text_rel)) else []
        cand_vision_list = candidate_vision_maps[i] if (candidate_vision_maps and i < len(candidate_vision_maps)) else []
        
        for c_text in cand_text_list:
            all_img_parts.append(c_text.flatten())
        
        cand_vision_shapes = []
        for c_vis_blocks in cand_vision_list:
            this_cand_shapes = []
            for v_map in c_vis_blocks:
                if v_map is not None:
                    all_img_parts.append(v_map.flatten())
                    this_cand_shapes.append(v_map.shape)
                else:
                    this_cand_shapes.append(None)
            cand_vision_shapes.append(this_cand_shapes)

        # 统一归一化
        img_scores_concat = np.concatenate(all_img_parts) if all_img_parts else np.array([], dtype=np.float32)
        txt_scores_concat = text_scores if text_scores is not None else np.array([], dtype=np.float32)
        
        if img_scores_concat.size == 0 and txt_scores_concat.size == 0:
            normalized_text.append(text_scores)
            normalized_vision.append(block_maps)
            normalized_cand_text.append(cand_text_list)
            normalized_cand_vision.append(cand_vision_list)
            continue

        img_norm, txt_norm = normalize_scores(img_scores_concat, txt_scores_concat)
        
        # 写回主回答
        normalized_text.append(txt_norm)
        rebuilt_blocks = []
        cursor = 0
        for shape in img_shapes:
            if shape is None:
                rebuilt_blocks.append(None)
                continue
            size = np.prod(shape)
            rebuilt_blocks.append(img_norm[cursor:cursor+size].reshape(shape))
            cursor += size
        normalized_vision.append(rebuilt_blocks)
        
        # 写回候选者文本分数
        this_step_cand_text = []
        for c_text in cand_text_list:
            size = len(c_text)
            this_step_cand_text.append(img_norm[cursor:cursor+size])
            cursor += size
        normalized_cand_text.append(this_step_cand_text)
        
        # 写回候选者视觉分数
        this_step_cand_vision = []
        for c_idx, c_vis_shapes in enumerate(cand_vision_shapes):
            this_cand_rebuilt = []
            for shape in c_vis_shapes:
                if shape is None:
                    this_cand_rebuilt.append(None)
                    continue
                size = np.prod(shape)
                this_cand_rebuilt.append(img_norm[cursor:cursor+size].reshape(shape))
                cursor += size
            this_step_cand_vision.append(this_cand_rebuilt)
        normalized_cand_vision.append(this_step_cand_vision)

    return normalized_text, normalized_vision, normalized_cand_text, normalized_cand_vision
