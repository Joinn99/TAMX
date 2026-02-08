"""Core logic for Token Activation Maps (TAM).

This module provides the main function `compute_tam` which computes the relevance
of vision and text tokens for the assistant's response in a multimodal model.
"""

import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from .utils import parse_seq_qwenvl
from .functions import (
    assign_grids,
    compute_unique_weights,
    compute_vision_activations,
    compute_text_relevance,
    compute_candidate_text_relevance,
    compute_candidate_logits,
    compute_vision_maps,
    normalize_answer_scores
)

@torch.no_grad()
def compute_tam(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    candidate_token_ids: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    apply_eci: bool = True,
    apply_filter: bool = True,
    kernel_size: int = 3,
    spatial_merge_size: Optional[int] = None,
    temporal_merge_size: Optional[int] = None
) -> Dict[str, Any]:
    """Compute Token Activation Maps for the last assistant response.

    This function analyzes the relationship between vision tokens and text tokens
    to generate relevance maps (TAMs) explaining why specific tokens were generated.

    Args:
        hidden_states: Hidden states from the model [SequenceLength-1, HiddenDim].
        input_ids: Input and generated token IDs [SequenceLength].
        lm_head_weight: Weights from the language model head [VocabSize, HiddenDim].
        candidate_token_ids: Optional candidate token IDs for each step.
        image_grid_thw: Grid dimensions (T, H, W) for each image in the sequence.
        video_grid_thw: Grid dimensions (T, H, W) for each video in the sequence.
        apply_eci: Whether to apply Error-Corrected Interpretability. Defaults to True.
        apply_filter: Whether to apply smoothing filters to vision maps. Defaults to True.
        kernel_size: Kernel size for the filtering. Defaults to 3.
        spatial_merge_size: Factor for spatial pooling of vision tokens.
        temporal_merge_size: Factor for temporal pooling of vision tokens.

    Returns:
        Dictionary containing TAM results:
        - ans_pos: Positions of response tokens.
        - ans_tokens: IDs of response tokens.
        - vision_blocks: Metadata about identified vision blocks.
        - vision_maps: Activation maps for each vision block per answer token.
        - candidate_vision_maps: Activation maps for candidate tokens.
        - text_rel: Relevance scores for context text tokens.
        - turns: Parsed conversation structure.
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    ids = input_ids.tolist()
    
    # Step 0: Parse sequence structure
    num_gen = candidate_token_ids.shape[0] if candidate_token_ids is not None else None
    struct = parse_seq_qwenvl(ids, num_generated_tokens=num_gen)
    vision_blocks = struct['vision_blocks']
    ans_pos = struct['ans_pos']
    ctx_text_pos = struct['ctx_text_pos']
    
    if not ans_pos:
        return {"error": "No assistant answer found in sequence"}

    assign_grids(
        vision_blocks=vision_blocks,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        temporal_merge_size=temporal_merge_size
    )

    w_unique, id_to_idx = compute_unique_weights(
        ids=ids,
        candidate_token_ids=candidate_token_ids,
        lm_head_weight=lm_head_weight,
        device=device,
        dtype=dtype
    )

    vision_activations = compute_vision_activations(
        hidden_states=hidden_states,
        vision_blocks=vision_blocks,
        w_unique=w_unique
    )

    r_full, text_rel_list, pos_to_ctx_idx = compute_text_relevance(
        hidden_states=hidden_states,
        ctx_text_pos=ctx_text_pos,
        ans_pos=ans_pos,
        ids=ids,
        w_unique=w_unique,
        id_to_idx=id_to_idx
    )

    candidate_text_rel = compute_candidate_text_relevance(
        hidden_states=hidden_states,
        ctx_text_pos=ctx_text_pos,
        ans_pos=ans_pos,
        ids=ids,
        w_unique=w_unique,
        id_to_idx=id_to_idx,
        candidate_token_ids=candidate_token_ids
    )

    candidate_logits = compute_candidate_logits(
        hidden_states=hidden_states,
        ans_pos=ans_pos,
        ids=ids,
        lm_head_weight=lm_head_weight,
        candidate_token_ids=candidate_token_ids
    )

    all_vision_maps, all_candidate_vision_maps = compute_vision_maps(
        vision_blocks=vision_blocks,
        ans_pos=ans_pos,
        ids=ids,
        candidate_token_ids=candidate_token_ids,
        id_to_idx=id_to_idx,
        vision_activations=vision_activations,
        ctx_text_pos=ctx_text_pos,
        r_full=r_full,
        pos_to_ctx_idx=pos_to_ctx_idx,
        apply_eci=apply_eci,
        apply_filter=apply_filter,
        kernel_size=kernel_size
    )

    text_rel_list, all_vision_maps, candidate_text_rel, all_candidate_vision_maps = normalize_answer_scores(
        text_rel_list=text_rel_list,
        vision_maps=all_vision_maps,
        candidate_text_rel=candidate_text_rel,
        candidate_vision_maps=all_candidate_vision_maps
    )

    return {
        "ans_pos": ans_pos,
        "ans_tokens": [ids[p] for p in ans_pos],
        "vision_blocks": vision_blocks,
        "vision_maps": all_vision_maps,
        "candidate_vision_maps": all_candidate_vision_maps,
        "candidate_token_ids": candidate_token_ids.tolist() if candidate_token_ids is not None else None,
        "candidate_text_rel": candidate_text_rel,
        "candidate_logits": candidate_logits,
        "text_rel": text_rel_list,
        "ctx_text_pos": ctx_text_pos,
        "turns": struct['turns']
    }
