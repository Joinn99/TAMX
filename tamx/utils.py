"""Utility functions for sequence parsing and structure identification.

This module helps identify vision blocks, conversation turns, and token positions
within the multimodal model's input/output sequences.
"""

from typing import List, Dict, Set, Any, Optional
from qwen_vl_utils import process_vision_info


def parse_seq_qwenvl(ids: List[int], num_generated_tokens: Optional[int] = None) -> Dict[str, Any]:
    """
    Step 0: Parse sequence structure
    
    Identifies vision blocks (distinguishing Image vs Video), turns (user/assistant), and visibility.
    
    Returns:
        A dictionary containing:
        - 'vision_blocks': List of dicts with 'start', 'end', 'v_pos', 'turn_idx', and 'type' ('image'|'video').
        - 'ans_pos': List of positions for the target response.
        - 'ctx_text_pos': List of positions for text tokens before the answer.
        - 'turns': List of dicts representing each message turn.
    """
    # Define special token IDs for Qwen-VL
    VISION_START_ID = 151652
    VISION_END_ID = 151653
    IMAGE_PAD_ID = 151655
    VIDEO_PAD_ID = 151656
    IM_START_ID = 151644
    IM_END_ID = 151645
    ASSISTANT_ID = 77091 # "assistant"
    SYSTEM_ID = 8948 # "system"
    USER_ID = 872 # "user"
    NEWLINE_ID = 198 # "\n"

    SPECIAL_IDS_SET = {
        IM_START_ID, IM_END_ID,
        ASSISTANT_ID, USER_ID, SYSTEM_ID,
        VISION_START_ID, VISION_END_ID, IMAGE_PAD_ID, VIDEO_PAD_ID
    }

    # 1. Identify all vision blocks and their types
    vision_blocks = []
    i = 0
    while i < len(ids):
        if ids[i] == VISION_START_ID:
            start = i
            v_type = "image" # default
            while i < len(ids) and ids[i] != VISION_END_ID:
                if ids[i] == VIDEO_PAD_ID:
                    v_type = "video"
                i += 1
            if i < len(ids):
                vision_blocks.append({
                    'start': start,
                    'end': i,
                    'v_pos': list(range(start + 1, i)),
                    'turn_idx': -1,
                    'type': v_type
                })
        i += 1

    # 2. Identify turns and associate vision blocks
    turns = []
    i = 0
    while i < len(ids):
        if ids[i] == IM_START_ID:
            role_p = i + 1
            if role_p < len(ids):
                if ids[role_p] == SYSTEM_ID:
                    role = "system"
                else:
                    role = "user" if ids[role_p] == USER_ID else "assistant"
                # Find end of turn
                end_p = len(ids)
                for j in range(role_p + 1, len(ids)):
                    if ids[j] == IM_END_ID:
                        end_p = j
                        break
                
                # Content start (skip role and optional newline)
                content_start = role_p + 1
                if content_start < len(ids) and ids[content_start] == NEWLINE_ID:
                    content_start += 1
                
                turn_idx = len(turns)
                # Check for vision blocks within this turn
                turn_vision_blocks = []
                for idx, vb in enumerate(vision_blocks):
                    if i <= vb['start'] and vb['end'] <= end_p:
                        vb['turn_idx'] = turn_idx
                        turn_vision_blocks.append(idx)
                
                # Collect text positions (excluding vision blocks and special tokens)
                turn_pos = []
                curr = content_start
                while curr < end_p:
                    is_vision = False
                    for vb_idx in turn_vision_blocks:
                        vb = vision_blocks[vb_idx]
                        if vb['start'] <= curr <= vb['end']:
                            is_vision = True
                            curr = vb['end'] + 1
                            break
                    if is_vision: continue
                    if ids[curr] not in SPECIAL_IDS_SET:
                        turn_pos.append(curr)
                    curr += 1
                
                if turn_pos or turn_vision_blocks:
                    turns.append({
                        "role": role,
                        "pos": turn_pos,
                        "vision_blocks": turn_vision_blocks
                    })
                i = end_p
        i += 1

    # 3. Identify target response (ans_pos)
    if num_generated_tokens is not None:
        # If explicitly provided, ans_pos is the last N tokens
        ans_pos = list(range(len(ids) - num_generated_tokens, len(ids)))
    else:
        # Otherwise it's the last turn
        ans_pos = turns[-1]["pos"] if turns else []

    # 4. Context tokens (all text tokens before ans_pos[0])
    first_ans_p = ans_pos[0] if ans_pos else len(ids)
    ctx_text_pos = []
    for turn in turns:
        for p in turn["pos"]:
            if p < first_ans_p:
                ctx_text_pos.append(p)

    return {
        'vision_blocks': vision_blocks,
        'ans_pos': ans_pos,
        'ctx_text_pos': ctx_text_pos,
        'turns': turns,
        'special_ids': SPECIAL_IDS_SET
    }
