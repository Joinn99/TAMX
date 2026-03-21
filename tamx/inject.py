"""Gaze-guided logit injection for personalized VLM decoding.

Implements TAM-gated gaze logit bias injection with coverage decay::

    logits += alpha * gamma_i * delta_i * gaze_bias

where:
  - ``gaze_bias``  = lm_head_weight @ v_G   (offline, gaze-weighted visual feature)
  - ``gamma_i``    = ECI-style gate          (online, per-decoding-step)
  - ``delta_i``    = exp(-coverage / tau_cov)  (coverage decay, reduces injection
                     automatically once gaze-related content has been generated)

Usage::

    # 1. Build injector from encode.py output
    injector = GazeInjector.from_encode_output(
        gaze_map=G_u,               # [n_v] – normalized attention weights over visual tokens
        hidden_states=hidden_state, # [seq-1, d] from encode.py
        input_ids=generated_ids,    # [seq]
        lm_head_weight=model.lm_head.weight,
        image_grid_thw=...,
        alpha=1.0,
        tau_coverage=3.0,           # coverage decay speed (default 3.0)
    )

    # 2. Inside a custom decoding loop, each step
    logits, gamma, delta = injector.inject(
        logits, h_last, ctx_hidden_states, ctx_ids,
        last_generated_token_id=prev_token_id,
    )
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Mapping

from .functions import (
    assign_grids,
    compute_unique_weights,
    compute_vision_activations,
    compute_eci_gate,
)
from .utils import parse_seq_qwenvl


class GazeInjector:
    """Offline precomputation of gaze logit bias + online gated injection.

    Args:
        gaze_bias: Precomputed gaze logit bias vector ``[V]``.
        visual_hidden_states: Visual token hidden states ``[n_v, d]`` (F^v).
        vision_activations: Activation matrix ``[n_v, n_unique]`` reused from TAM.
        id_to_idx: Mapping from token ID to column index in ``vision_activations``.
        alpha: Injection strength (default 1.0).
        tau: Gate sensitivity (``None`` = auto-calibrate per step).
    """

    def __init__(
        self,
        gaze_bias: torch.Tensor,             # [V]
        visual_hidden_states: torch.Tensor,  # [n_v, d], already clamped float32
        vision_activations: np.ndarray,      # [n_v, n_unique]
        id_to_idx: Dict[int, int],
        alpha: float = 1.0,
        tau: Optional[float] = None,
        tau_ref: Optional[float] = None,     # global baseline for auto-tau
        # --- Coverage Decay ---
        v_G: Optional[torch.Tensor] = None,             # [d] calibrated gaze feature
        lm_head_weight: Optional[torch.Tensor] = None,  # [V, d]
        tau_coverage: float = 3.0,
        gaze_bias_normalized: Optional[torch.Tensor] = None,
    ):
        self.gaze_bias = gaze_bias
        self.visual_hidden_states = visual_hidden_states
        self.vision_activations = vision_activations
        self.id_to_idx = id_to_idx
        self.alpha = alpha
        self.tau = tau
        # tau_ref: mean ||relu(F^v @ h_ans)||_1 across answer positions.
        # Used as the auto-tau baseline so gamma varies meaningfully per step.
        self.tau_ref = tau_ref
        # Coverage decay attributes
        self.v_G = v_G                       # calibrated gaze feature vector [d]
        self.lm_head_weight = lm_head_weight  # [V, d]
        self.tau_coverage = tau_coverage
        self.coverage = 0.0
        self.step_count = 0
        self.delta_history: List[dict] = []
        self.gaze_bias_normalized = gaze_bias_normalized

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_encode_output(
        cls,
        gaze_map: torch.Tensor,              # [n_v] – gaze weights over visual tokens
        hidden_states: torch.Tensor,         # [seq-1, d] from encode.py
        input_ids: torch.Tensor,             # [seq]
        lm_head_weight: torch.Tensor,        # [V, d]
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        special_token_ids: Optional[Mapping[str, Optional[int]]] = None,
        alpha: float = 1.0,
        tau: Optional[float] = None,
        spatial_merge_size: Optional[int] = None,
        temporal_merge_size: Optional[int] = None,
        tau_coverage: float = 3.0,           # coverage decay speed parameter
    ) -> "GazeInjector":
        """Construct a :class:`GazeInjector` directly from ``encode.py`` outputs.

        All heavy work reuses functions already present in ``tamx.functions``
        so this stays consistent with ``compute_tam``.
        """
        # --- squeeze batch dim if present ---
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[0]          # [seq-1, d]
        if input_ids.dim() == 2:
            input_ids = input_ids[0]                  # [seq]

        # Sanitize hidden states: bfloat16 values at the dtype max (~3.38e38)
        # cause float32 dot products to overflow.  We clip to the same threshold
        # used in compute_eci_gate so that vision_activations and a_raw are in
        # the same scale (both will be relu(F^v_clipped @ query_clipped)).
        SAFE_CLIP = 300.0
        hidden_states = torch.nan_to_num(hidden_states.float()).clamp(-SAFE_CLIP, SAFE_CLIP)

        device = hidden_states.device
        dtype = hidden_states.dtype    # float32 after clamp
        ids = input_ids.tolist()

        # Step 0: parse sequence structure (identical to compute_tam)
        struct = parse_seq_qwenvl(ids, special_token_ids=special_token_ids)
        vision_blocks = struct["vision_blocks"]

        assign_grids(
            vision_blocks=vision_blocks,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            spatial_merge_size=spatial_merge_size,
            temporal_merge_size=temporal_merge_size,
        )

        # Step 1: unique token weights
        w_unique, id_to_idx = compute_unique_weights(
            ids=ids,
            candidate_token_ids=None,
            lm_head_weight=lm_head_weight,
            device=device,
            dtype=dtype,
        )

        # Step 2: vision activations [n_v, n_unique]
        vision_activations_list = compute_vision_activations(
            hidden_states=hidden_states,
            vision_blocks=vision_blocks,
            w_unique=w_unique,
        )

        assert len(vision_blocks) > 0, "No vision blocks found in input_ids."
        v_block = vision_blocks[0]
        vision_activations = vision_activations_list[0]   # [n_v, n_unique]

        # Step 3: extract F^v (visual token hidden states), upcast to float32
        # to avoid bfloat16/float16 overflow in subsequent matmuls
        v_pos = v_block["v_pos"]
        F_v = hidden_states[v_pos].float()                # [n_v, d] in float32

        # Step 4: offline gaze-weighted visual feature
        #   v_G = Σ_j  G_j * F^v_j
        gaze_map = gaze_map.to(device=F_v.device, dtype=F_v.dtype)
        gaze_map_norm = gaze_map / (gaze_map.sum() + 1e-8)
        v_G = (gaze_map_norm.unsqueeze(-1) * F_v).sum(0)  # [d] float32

        # Static calibration
        v_norm_avg = F_v.norm(dim=-1).mean()
        v_G_calibrated = v_G * (v_norm_avg / (v_G.norm() + 1e-8))

        # Step 5: offline gaze logit bias  (compute in float32 to avoid overflow)
        #   b_t = v_G · w_t  for all vocabulary tokens
        lm_head_w = lm_head_weight.to(device=v_G.device, dtype=torch.float32)
        gaze_bias = lm_head_w @ v_G_calibrated                       # [V] float32

        # Step 6: compute tau_ref — mean a_raw.sum() across answer positions.
        # F_v and h_ans are in the same float32 / SAFE_CLIP space, so tau_ref
        # is on the exact same scale as a_raw inside compute_eci_gate.
        tau_ref: Optional[float] = tau   # skip if user provided explicit tau
        if tau is None:
            struct = parse_seq_qwenvl(ids, special_token_ids=special_token_ids)
            ans_pos_list = struct["ans_pos"]
            raw_sums = []
            for ap in ans_pos_list:
                h_idx = ap - 1
                if 0 <= h_idx < hidden_states.shape[0]:
                    h_ans = hidden_states[h_idx]          # float32, clamped
                    raw = torch.relu(F_v @ h_ans).sum().item()
                    raw_sums.append(raw)
            tau_ref = float(np.median(raw_sums)) if raw_sums else None

        # Step 7: normalize gaze_bias for coverage decay
        gaze_bias_normalized = gaze_bias / (gaze_bias.norm() + 1e-8)

        return cls(
            gaze_bias=gaze_bias,
            visual_hidden_states=F_v,
            vision_activations=vision_activations,
            id_to_idx=id_to_idx,
            alpha=alpha,
            tau=tau,
            tau_ref=tau_ref,
            v_G=v_G_calibrated,
            lm_head_weight=lm_head_weight,
            tau_coverage=tau_coverage,
            gaze_bias_normalized=gaze_bias_normalized,
        )

    # ------------------------------------------------------------------
    # Online interface
    # ------------------------------------------------------------------

    def compute_gate(
        self,
        h_last: torch.Tensor,            # [d]
        ctx_hidden_states: torch.Tensor, # [n_ctx, d]
        ctx_ids: List[int],
    ) -> float:
        """Return ECI gate gamma_i for the current decoding step."""
        # Use explicit tau if set; otherwise use tau_ref calibrated at
        # construction time from the actual answer hidden states.
        effective_tau = self.tau if self.tau is not None else self.tau_ref
        return compute_eci_gate(
            h_last=h_last,
            visual_hidden_states=self.visual_hidden_states,
            vision_activations=self.vision_activations,
            ctx_hidden_states=ctx_hidden_states,
            ctx_ids=ctx_ids,
            id_to_idx=self.id_to_idx,
            tau=effective_tau,
        )

    def compute_coverage_decay(self, generated_token_id: int) -> float:
    # 用归一化后的 gaze_bias 值
    # self.gaze_bias_normalized 在 from_encode_output 中预计算:
    #   self.gaze_bias_normalized = self.gaze_bias / (self.gaze_bias.norm() + eps)
        score = self.gaze_bias_normalized[generated_token_id].item()
        
        if score > 0:
            self.coverage += score
        
        delta_i = np.exp(-self.coverage / self.tau_coverage)
        return delta_i

    def reset_coverage(self):
        """Reset coverage state before each new generate() call."""
        self.coverage = 0.0
        self.step_count = 0
        self.delta_history = []

    @torch.no_grad()
    def inject(
        self,
        logits: torch.Tensor,            # [V]
        h_last: torch.Tensor,            # [d]
        ctx_hidden_states: torch.Tensor, # [n_ctx, d]
        ctx_ids: List[int],
        last_generated_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """Apply gaze-gated bias to a single-step logit vector.

        Args:
            last_generated_token_id: Token id generated at the previous step.
                First step should pass ``None`` (no decay applied, δ_i = 1.0).

        Returns:
            Tuple of (injected_logits, gamma_i, delta_i).
            Formula: ``logits + alpha * γ_i * δ_i * gaze_bias``
        """
        gamma = self.compute_gate(h_last, ctx_hidden_states, ctx_ids)
        if last_generated_token_id is not None and self.v_G is not None:
            delta_i = self.compute_coverage_decay(last_generated_token_id)
        else:
            delta_i = 1.0
        bias = self.gaze_bias.to(device=logits.device, dtype=logits.dtype)
        return logits + self.alpha * gamma * delta_i * bias, gamma, delta_i

    @torch.no_grad()
    def compute_logit_bias(
        self,
        h_last: torch.Tensor,            # [d]
        ctx_hidden_states: torch.Tensor, # [n_ctx, d]
        ctx_ids: List[int],
        last_generated_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute the effective logit bias *without* modifying anything.

        Returns:
            (effective_bias [V] in float32, gamma float, delta_i float)
        """
        gamma = self.compute_gate(h_last, ctx_hidden_states, ctx_ids)
        if last_generated_token_id is not None and self.v_G is not None:
            import torch.nn.functional as F
            w_t = self.lm_head_weight[last_generated_token_id].float()
            v_G = self.v_G.float()
            cos_sim = F.cosine_similarity(
                w_t.unsqueeze(0),
                v_G.to(device=w_t.device).unsqueeze(0),
            ).item()
            temp_coverage = self.coverage + (cos_sim if cos_sim > 0 else 0.0)
            delta_i = float(np.exp(-temp_coverage / self.tau_coverage))
        else:
            delta_i = 1.0
        # gaze_bias is already float32; keep it that way for accurate reporting
        return self.alpha * gamma * delta_i * self.gaze_bias, gamma, delta_i
