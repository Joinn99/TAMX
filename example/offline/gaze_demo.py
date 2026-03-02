"""gaze_demo.py — Offline demo of GazeInjector logit bias computation.

Loads the pre-existing encode.safetensors produced by example/offline/encode.py,
creates a gaze map, then for each answer-token position prints:

  - Original Top-5 tokens (before gaze injection) with logit values
  - Injected Top-5 tokens (after gaze injection) with logit values
  - ECI gate γ_i and the bias applied to the actual chosen token

Usage (from repo root):
    MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct python example/offline/gaze_demo.py
"""

import os
import sys
import torch
import safetensors.torch as st
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Make sure the repo root is on the path so `tamx` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tamx import GazeInjector
from tamx.utils import parse_seq_qwenvl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENCODE_FILE = "asset/output/encode.safetensors"
ALPHA = 1.0
SEED = 42

# ---------------------------------------------------------------------------
# Load model (needed for lm_head weights)
# ---------------------------------------------------------------------------
model_name = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
print(f"[gaze_demo] Loading model: {model_name}")

if "Qwen3-VL" in model_name:
    model_class = Qwen3VLForConditionalGeneration
else:
    model_class = Qwen2_5_VLForConditionalGeneration

model = model_class.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.eval()
processor = AutoProcessor.from_pretrained(model_name)

# ---------------------------------------------------------------------------
# Load encode output
# ---------------------------------------------------------------------------
print(f"[gaze_demo] Loading encoded data from: {ENCODE_FILE}")
data = st.load_file(ENCODE_FILE)

hidden_states: torch.Tensor = data["hidden_state"]          # [1, seq-1, d] or [seq-1, d]
input_ids: torch.Tensor     = data["generated_ids"]          # [1, seq] or [seq]
image_grid_thw              = data.get("image_grid_thw")
video_grid_thw              = data.get("video_grid_thw")

# Normalise shapes (remove batch dim if present)
if hidden_states.dim() == 3:
    hidden_states = hidden_states[0]   # [seq-1, d]
if input_ids.dim() == 2:
    input_ids = input_ids[0]           # [seq]

if image_grid_thw is not None and image_grid_thw.numel() == 0:
    image_grid_thw = None
if video_grid_thw is not None and video_grid_thw.numel() == 0:
    video_grid_thw = None

# Sanitize: bfloat16 hidden states can have values at the dtype max (~3.38e38),
# which overflow float32 dot products.  Clip to the same range as inject.py.
SAFE_CLIP = 300.0
hidden_states = torch.nan_to_num(hidden_states.float()).clamp(-SAFE_CLIP, SAFE_CLIP)

ids = input_ids.tolist()
print(f"[gaze_demo] Sequence length: {len(ids)},  hidden dim: {hidden_states.shape[-1]}")

# ---------------------------------------------------------------------------
# Parse sequence to find vision-token positions and answer positions
# ---------------------------------------------------------------------------
struct = parse_seq_qwenvl(ids)
vision_blocks = struct["vision_blocks"]
ans_pos = struct["ans_pos"]
ctx_text_pos = struct["ctx_text_pos"]

assert vision_blocks, "No vision blocks found – is the encode.safetensors from a multimodal prompt?"
n_v = len(vision_blocks[0]["v_pos"])

print(f"[gaze_demo] #visual tokens: {n_v},  #answer tokens: {len(ans_pos)}")

# ---------------------------------------------------------------------------
# Random gaze map (placeholder – replace with real Generalized-ECI output)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

import pandas as pd
df = pd.read_csv('asset/sample.csv', header=None, delimiter='\t')
gaze_map_np = df.values.astype(np.float32).flatten()


# gaze_map_np = rng.random(n_v).astype(np.float32)




gaze_map = torch.from_numpy(gaze_map_np)
print(f"[gaze_demo] Using random gaze map (seed={SEED})")

# ---------------------------------------------------------------------------
# Build GazeInjector (offline precomputation)
# ---------------------------------------------------------------------------


print("[gaze_demo] Building GazeInjector …")
injector = GazeInjector.from_encode_output(
    gaze_map=gaze_map,
    hidden_states=hidden_states,
    input_ids=input_ids,
    lm_head_weight=model.lm_head.weight,
    image_grid_thw=image_grid_thw,
    video_grid_thw=video_grid_thw,
    alpha=ALPHA,
    tau=None,   # auto-calibrate each step
)
print(f"[gaze_demo] gaze_bias shape: {injector.gaze_bias.shape}")

# ---------------------------------------------------------------------------
# Pre-cache lm_head weight on CPU in float32 for fast logit computation
# ---------------------------------------------------------------------------
lm_head_w = model.lm_head.weight.detach().to(device="cpu", dtype=torch.float32)  # [V, d]

# ---------------------------------------------------------------------------
# Offline walkthrough: compare top-5 before/after gaze injection
# ---------------------------------------------------------------------------
TOP_K = 5
print("\n" + "=" * 72)
print("Per-token Top-5 comparison: original vs gaze-injected logits")
print("=" * 72)

device = hidden_states.device

for step_i, ans_p in enumerate(ans_pos):
    tok_id = ids[ans_p]
    tok_str = processor.tokenizer.decode([tok_id])

    h_idx = ans_p - 1
    if h_idx < 0 or h_idx >= hidden_states.shape[0]:
        print(f"  Step {step_i:3d} | tok='{tok_str}' | h_idx out of range, skipped")
        continue

    h_last = hidden_states[h_idx].clone()   # float32 + clamped

    # Build ctx
    current_ctx_pos = [p for p in ctx_text_pos if p < ans_p]
    current_ctx_pos += [p for p in ans_pos[:step_i] if p < ans_p]
    if current_ctx_pos:
        ctx_hs = hidden_states[current_ctx_pos]
        ctx_ids_list = [ids[p] for p in current_ctx_pos]
    else:
        ctx_hs = h_last.unsqueeze(0)
        ctx_ids_list = [tok_id]

    with torch.no_grad():
        # Original logits: h_last @ lm_head^T
        orig_logits = lm_head_w @ h_last.cpu()           # [V]

        # Gaze bias
        effective_bias, gamma = injector.compute_logit_bias(h_last, ctx_hs, ctx_ids_list)
        new_logits = orig_logits + effective_bias.cpu()  # [V]

    # Top-K before and after
    orig_vals, orig_idxs = torch.topk(orig_logits, TOP_K)
    new_vals,  new_idxs  = torch.topk(new_logits,  TOP_K)

    def fmt(idxs, vals):
        return [(processor.tokenizer.decode([i.item()]), f'{v.item():+.3f}')
                for i, v in zip(idxs, vals)]

    orig_top = fmt(orig_idxs, orig_vals)
    new_top  = fmt(new_idxs,  new_vals)

    bias_for_tok = effective_bias[tok_id].item()

    print(f"\nStep {step_i:3d} | actual token = '{tok_str}' (id={tok_id})  γ={gamma:.4f}  bias_for_tok={bias_for_tok:+.3f}")
    print(f"  {'Original Top-' + str(TOP_K):<30}  {'Injected Top-' + str(TOP_K):<30}")
    print(f"  {'─'*30}  {'─'*30}")
    for (ot, ov), (nt, nv) in zip(orig_top, new_top):
        changed = "*" if ot != nt else " "
        print(f"  {ot!r:>15} {ov:>8}    {changed} {nt!r:>15} {nv:>8}")

print("\n" + "=" * 72)
print("[gaze_demo] Done. No decoding was modified.")

