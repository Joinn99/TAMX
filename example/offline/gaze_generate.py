"""gaze_generate.py — End-to-end gaze logit-bias injection during model.generate().

Accepts a user message (with image link) and a gaze map.
No pre-saved encode.safetensors required.

Pipeline:
  1. Prefill pass  — run the prompt through the model with output_hidden_states=True
                    to obtain visual-token hidden states and calibrate tau_ref.
                    (Mirrors encode.py, but inline and without saving to disk.)
  2. Build injector — construct GazeInjector from the prefill hidden states.
  3. Injected generate — run model.generate() with a GazeLogitsProcessor that:
       a. captures h_last at every step via a forward hook on model.model.norm, and
       b. calls injector.inject(logits, h_last, ctx_hs, ctx_ids) before sampling.

Usage (from repo root):
    MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct python example/offline/gaze_generate.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from transformers import (
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    Qwen3VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tamx import GazeInjector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGE_FILE     = "asset/road_sign_asphalt_road.jpg"
GAZE_CSV       = "asset/sample.csv"
MAX_NEW_TOKENS = 128
SAFE_CLIP      = 300.0   # same as encode.py / inject.py
SHOW_TOP_K     = True    # print top-3 candidates + logits per decoding step
TAU_COVERAGE   = 3.0     # coverage decay speed: lower = faster decay

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model_name = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
print(f"[gaze_generate] Loading model: {model_name}")

model_class = (
    Qwen3VLForConditionalGeneration
    if "Qwen3-VL" in model_name
    else Qwen2_5_VLForConditionalGeneration
)
model = model_class.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.eval()
processor = AutoProcessor.from_pretrained(model_name)

# ---------------------------------------------------------------------------
# Gaze map
# ---------------------------------------------------------------------------
df = pd.read_csv(GAZE_CSV, header=None, delimiter="\t")
gaze_map_np = df.values.astype(np.float32).flatten()
gaze_map = torch.from_numpy(gaze_map_np)
print(f"[gaze_generate] Gaze map loaded, shape={gaze_map.shape}")

# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------
# Provide your own messages here; just make sure at least one vision token
# (image or video) is present so the injector can extract F^v.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_FILE},
            {"type": "text",  "text": "Briefly describe this image."},
        ],
    }
]

# ---------------------------------------------------------------------------
# Helper: build processor inputs from messages
# ---------------------------------------------------------------------------
def build_inputs(msgs, add_generation_prompt: bool):
    text = processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_patch_size = 16 if "Qwen3-VL" in model_name else 14
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        msgs,
        image_patch_size=image_patch_size,
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
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs.to(model.device)

# ---------------------------------------------------------------------------
# Step 1 — Prefill pass (mirrors encode.py)
#
# We run the prompt (with generation prompt, no answer yet) through the model
# with output_hidden_states=True and max_new_tokens=1 so we get the hidden
# states for every input position before the first generated token.
#
# hidden_state shape: [1, seq_len, d]  (last transformer layer)
# We drop the very last position (the one generated token) and keep [seq_len-1, d],
# matching the layout expected by GazeInjector.from_encode_output.
# ---------------------------------------------------------------------------
print("[gaze_generate] Running prefill pass to extract hidden states …")

prefill_inputs = build_inputs(messages, add_generation_prompt=True)

with torch.no_grad():
    prefill_out = model.generate(
        **prefill_inputs,
        max_new_tokens=1,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

# Concatenate last-layer hidden states across all positions, drop the generated token
# (same logic as encode.py line 63):
#   outputs.hidden_states is a tuple of length (seq_len + 1), where entry 0
#   is the full prefill and entries 1..(seq_len) have shape [1, 1, d].
hidden_state_raw = torch.cat(
    [feats[-1] for feats in prefill_out.hidden_states], dim=1
)[:, :-1, :]                             # [1, seq_len, d]  (drop the 1 generated token)

hidden_states = hidden_state_raw[0]     # [seq_len, d]
hidden_states = torch.nan_to_num(hidden_states.float()).clamp(-SAFE_CLIP, SAFE_CLIP)

input_ids = prefill_inputs["input_ids"]  # [1, seq_len]
image_grid_thw = prefill_inputs.get("image_grid_thw")   # may be None or tensor
video_grid_thw = prefill_inputs.get("video_grid_thw")

if image_grid_thw is not None and image_grid_thw.numel() == 0:
    image_grid_thw = None
if video_grid_thw is not None and video_grid_thw.numel() == 0:
    video_grid_thw = None

print(f"[gaze_generate] Prefill done — seq_len={hidden_states.shape[0]}, "
      f"hidden_dim={hidden_states.shape[1]}")

# ---------------------------------------------------------------------------
# Step 2 — Build GazeInjector from prefill hidden states
# ---------------------------------------------------------------------------
print("[gaze_generate] Building GazeInjector …")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.3)
args = parser.parse_args()

injector = GazeInjector.from_encode_output(
    gaze_map=gaze_map,
    hidden_states=hidden_states,
    input_ids=input_ids,
    lm_head_weight=model.lm_head.weight,
    image_grid_thw=image_grid_thw,
    video_grid_thw=video_grid_thw,
    alpha=args.alpha,
    tau=None,          # auto-calibrate from prefill hidden states
    tau_coverage=TAU_COVERAGE,
)
print(f"[gaze_generate] gaze_bias shape: {injector.gaze_bias.shape},  "
      f"tau_ref={injector.tau_ref:.4g}" if injector.tau_ref else
      f"[gaze_generate] gaze_bias shape: {injector.gaze_bias.shape},  tau_ref=None")

# ---------------------------------------------------------------------------
# Step 3a — Forward hook to capture h_last at every decoding step
# ---------------------------------------------------------------------------
class _HiddenStateBuffer:
    """Stores the last-token hidden state written by the norm hook."""
    def __init__(self):
        self.h_last: Optional[torch.Tensor] = None   # [d] float32, CPU

    def hook_fn(self, module, input, output):
        # output: [batch, seq, d]; during cached generation seq == 1
        h = output[0, -1].detach().float()
        h = torch.nan_to_num(h).clamp(-SAFE_CLIP, SAFE_CLIP)
        self.h_last = h.cpu()


hs_buffer = _HiddenStateBuffer()

# Locate the final layer-norm robustly across Qwen model variants:
#   Qwen2.5-VL → model.model.norm
#   Qwen3-VL   → model.model.language_model.norm
def _find_final_norm(model):
    lm = model.model
    if hasattr(lm, 'norm'):
        return lm.norm
    elif hasattr(lm, 'language_model') and hasattr(lm.language_model, 'norm'):
        return lm.language_model.norm
    raise AttributeError(
        f"Cannot find final norm in {type(lm).__name__}. "
        "Inspect model.model with named_children() to find the correct path."
    )

final_norm = _find_final_norm(model)
print(f"[gaze_generate] Hooking final norm: {type(final_norm).__name__}")
_hook_handle = final_norm.register_forward_hook(hs_buffer.hook_fn)

# ---------------------------------------------------------------------------
# Step 3b — LogitsProcessor that reads h_last and injects gaze bias
# ---------------------------------------------------------------------------
class GazeLogitsProcessor(LogitsProcessor):
    """Per-step gaze bias injection via GazeInjector.inject().

    Context tracking:
      - ctx_ids / ctx_hs grow as tokens are generated.
      - h_last comes from the shared _HiddenStateBuffer written by the hook.
    """

    def __init__(
        self,
        injector: GazeInjector,
        hs_buffer: _HiddenStateBuffer,
        show_top_k: bool = False,
        top_k: int = 3,
    ):
        self.injector    = injector
        self.hs_buffer   = hs_buffer
        self.step        = 0
        self.ctx_ids: List[int] = []
        self.ctx_hs_list: List[torch.Tensor] = []   # accumulated [d] tensors, on F_v device
        self._F_v        = injector.visual_hidden_states   # [n_v, d]
        self._fv_device  = injector.visual_hidden_states.device  # target device for h tensors
        self.show_top_k  = show_top_k
        self.top_k       = top_k
        # list of (step, [(token_str, logit), ...]) recorded when show_top_k=True
        self.top_k_log: List[tuple] = []
        # Coverage decay: track last generated token id
        self.last_token_id: Optional[int] = None
        self.injector.reset_coverage()          # reset coverage state for this run

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        if self.hs_buffer.h_last is None:
            return scores          # hook hasn't fired; skip safely

        # h_last comes from the hook as CPU float32; move it to F_v's device
        # so compute_eci_gate's matmul (F_v @ h_last) stays on one device.
        h_last = self.hs_buffer.h_last.to(device=self._fv_device)  # [d]

        # Build context: accumulate h_last from previous steps
        if self.ctx_hs_list:
            ctx_hs = torch.stack(self.ctx_hs_list, dim=0)   # [n_ctx, d] on _fv_device
            ctx_ids = self.ctx_ids
        else:
            # No generated tokens yet — use one row of F_v as proxy
            ctx_hs  = self._F_v[:1].clone().float()          # on _fv_device
            ctx_ids = [input_ids[0, -1].item()]

        out = scores.clone()
        for b in range(scores.shape[0]):
            logits_b = scores[b].float().cpu()
            injected, gamma_i, delta_i = self.injector.inject(
                logits_b, h_last, ctx_hs, ctx_ids,
                last_generated_token_id=self.last_token_id,
            )
            out[b] = injected.to(device=scores.device, dtype=scores.dtype)

        # Record top-k candidates from the (post-injection) logits of batch[0]
        if self.show_top_k:
            logits0 = out[0].float().cpu()
            topk_vals, topk_ids = torch.topk(logits0, self.top_k)
            candidates = [
                (processor.tokenizer.decode([tid.item()]), val.item())
                for tid, val in zip(topk_ids, topk_vals)
            ]
            self.top_k_log.append((self.step, candidates))

        # Save h_last (on _fv_device) and predicted token for next step's context
        self.ctx_hs_list.append(h_last.clone())
        next_tok = out[0].argmax().item()
        self.ctx_ids.append(next_tok)
        # Update last_token_id for coverage decay computation in the next step
        self.last_token_id = next_tok
        self.step += 1

        return out


gaze_processor = GazeLogitsProcessor(injector, hs_buffer, show_top_k=SHOW_TOP_K)

# ---------------------------------------------------------------------------
# Step 3c — Run model.generate() with injection
# ---------------------------------------------------------------------------
print("[gaze_generate] Running model.generate() with gaze injection …")

gen_inputs = build_inputs(messages, add_generation_prompt=True)

with torch.no_grad():
    outputs = model.generate(
        **gen_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,   # greedy — swap in temperature/top_p if desired
        logits_processor=LogitsProcessorList([gaze_processor]),
        use_cache=True,
    )

_hook_handle.remove()   # clean up

# ---------------------------------------------------------------------------
# Decode & report
# ---------------------------------------------------------------------------
n_input      = gen_inputs["input_ids"].shape[1]
generated_ids = outputs[0, n_input:]
generated_text = processor.decode(generated_ids, skip_special_tokens=True)

print("\n" + "=" * 72)
print("[gaze_generate] Generated output (gaze-injected):")
print("=" * 72)
print(generated_text)
print("=" * 72)
print(f"[gaze_generate] Injection applied at {gaze_processor.step} decoding steps.")

if SHOW_TOP_K and gaze_processor.top_k_log:
    print("\n" + "=" * 72)
    print(f"[gaze_generate] Per-step top-{gaze_processor.top_k} candidates (post-injection logits):")
    print("=" * 72)
    header = f"{'Step':>5}  " + "  ".join(f"{'Rank '+str(r+1):<22}" for r in range(gaze_processor.top_k))
    print(header)
    print("-" * len(header))
    for step_idx, candidates in gaze_processor.top_k_log:
        cols = []
        for tok_str, logit in candidates:
            tok_repr = repr(tok_str) if (not tok_str or tok_str.isspace()) else tok_str
            cols.append(f"{tok_repr[:12]:<12} {logit:>8.3f}")
        print(f"{step_idx:>5}  " + "  ".join(cols))
    print("=" * 72)

print("[gaze_generate] Done.")
