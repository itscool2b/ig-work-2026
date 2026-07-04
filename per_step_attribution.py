"""
Per-step IG for RDT inside a ManiSkill episode loop.

Exports three functions:

- `prepare_ig_context(...)`: runs all per-step setup (SigLIP-encode the obs,
  stitch the 6-slot image, format proprio, adapt language/image/state through
  their adaptors under no_grad, define the seeded `conditional_sample` wrapper,
  compute the reference action) and returns a dict of tensors and closures that
  downstream modules consume.
- `build_forward_fns(ctx)`: returns (forward_fn_vision, forward_fn_language,
  forward_fn_state) from a prepared context. Each is a scalar function of its
  one free modality used by `integrated_gradients` and by perturbation
  ablations (Δlog p top-k, insertion/deletion AUC, sanity C2 input shuffle).
- `compute_ig_for_step(...)`: thin wrapper that calls the two above plus the
  shared `integrated_gradients` core and returns the per-modality summary dict
  per_step_ig.py consumes. API and return shape unchanged from the pre-refactor
  version so existing callers (per_step_ig.py:249) work without modification.

Design choices baked in here (all decided during the per-step IG planning pass):
- Target is `log π(a_t | o_t)` under a unit-variance Gaussian around the final
  denoised mean μ, with the constant entropy term dropped (does not affect IG
  gradients). RDT has `prediction_type="sample"` so `conditional_sample` output
  IS the mean μ; the Gaussian variance σ² is a hyperparameter (default 1.0).
- Vision baseline holds the five non-camera image slots equal to the input's
  five non-camera slots (both sides of the IG integration see the same
  `bg_image` tokens there); only slot 3 interpolates obs→gray. Completeness
  therefore reports attribution for the external camera alone, matching what
  downstream visualization plots.
- State IG interpolates the raw 128-dim pre-adaptor `state_vec`. `state_adaptor`
  runs inside `forward_fn_state` so autograd pulls gradients back to the raw
  input and attribution at `MANISKILL_INDICES` is readable per joint. (Dims
  outside MANISKILL_INDICES are identical between input and baseline so their
  attribution is zero by construction; `sum(state_attr_flat)` equals the
  per-joint sum up to float noise.)
- Seeded wrapper around `conditional_sample` re-seeds the global RNG inside
  every IG forward so the initial noise in the DPM-Solver++ chain is identical
  across all m+1 interpolation steps AND across the three modalities within a
  single step. This is the invariant that makes ref_action = F(real_input)
  exactly (f_input = 0) and lets completeness actually pass.
- Gradient checkpointing is NOT applied here; the caller applies it once on
  the runner at script start. Applying checkpointing inside this function would
  double-wrap block.forward on repeated calls.
- SigLIP encoding of the constant `bg_image` slot and the full baseline tensor
  are cached by the caller (passed in as `bg_image_encoded` and
  `img_tokens_baseline`). Only the new observation (slot 3 of input) is SigLIP
  encoded per step here. This saves ~83% of SigLIP time across a full episode
  pass.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/rdt-repo"))
from configs.state_vec import STATE_VEC_IDX_MAPPING

from integrated_gradients import integrated_gradients

#Constants, ManiSkill Panda arm mapping into RDT's 128-dim unified space.
#MANISKILL_INDICES = [0, 1, 2, 3, 4, 5, 6, 10] maps to 7 arm joints + gripper.
#The 120 dims outside this list stay at zero at both input and baseline, so
#their attribution is zero by construction (diff=0 along the whole IG path).
MANISKILL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
JOINT_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]

#Proprioception normalization bounds (same as ig_rdt.py). These are
#precomputed from the ManiSkill training data; they normalize the 8
#physical joints to [-1, 1] before slotting into the 128-dim unified vec.
STATE_MIN = torch.tensor([-0.7463, -0.0801, -0.4976, -2.6578, -0.5743, 1.8310, -2.2424, 0.0])
STATE_MAX = torch.tensor([0.7645,  1.4967,  0.4651, -0.3867,  0.5506, 3.2901,  2.5738, 0.04])


def prepare_ig_context(
    runner,
    vision_model,
    obs_image,
    proprio,
    lang_tokens,
    lang_attn_mask,
    lang_tokens_baseline,
    bg_image_encoded,
    img_tokens_baseline,
    action_mask,
    ctrl_freqs,
    seed=42,
    sigma_sq=1.0,
    frozen_ref_action=None,
    target="logpi",
):
    """
    Run all per-step IG setup and return a context dict.

    The returned dict holds everything an IG or perturbation run needs without
    re-doing SigLIP encoding, adaptor forwards, or reference-action computation.
    Used by `compute_ig_for_step` to drive IG and by `faithfulness.py` /
    `sanity.py` to run perturbed forwards.

    Args: same as `compute_ig_for_step` minus `m`.
        frozen_ref_action: if provided, use this as ref_action instead of
            computing it from the model. Used by frozen-target C1 sanity check
            to decouple the IG target from the (re-initialized) model.

    Returns dict with keys:
        runner, vision_model
        img_tokens                   (1, 4374, 1152) bf16, pre-adaptor, stitched
        img_tokens_baseline          (1, 4374, 1152) bf16, pre-adaptor baseline
        lang_adapted                 (1, 1024, H) bf16, runner.lang_adaptor(lang_tokens)
        lang_adapted_bl              (1, 1024, H) bf16
        img_adapted                  (1, 4374, H) bf16
        img_adapted_bl               (1, 4374, H) bf16
        state_input_actual           (1, 1, 128) bf16, raw pre-adaptor
        state_input_baseline         (1, 1, 128) bf16, zeros
        state_traj_actual            (1, 1, H) bf16, POST-adaptor token for
                                     the frozen-state path used in vision/lang IG
        ref_action                   (1, 64, 128) bf16, seeded conditional_sample
                                     output on real inputs (the Gaussian's mean μ)
        seeded_conditional_sample    callable(lang_c, img_c, state_t) -> (1,64,128)
        log_pi                       callable(pred_action) -> scalar
        action_mask, ctrl_freqs, lang_attn_mask: passthrough
        seed, sigma_sq               ints/floats
    """
    device = "cuda"
    dtype = torch.bfloat16
    image_processor = vision_model.image_processor

    #Encode the new observation through SigLIP.
    obs_processed = image_processor.preprocess(obs_image, return_tensors="pt")["pixel_values"][0]
    obs_batched = obs_processed.unsqueeze(0).to(device, dtype=dtype)  # (1, 3, 384, 384)
    with torch.no_grad():
        obs_tokens = vision_model(obs_batched).detach()  # (1, 729, 1152)

    #Stitch 6-slot input: [bg, bg, bg, obs, bg, bg] -> (1, 4374, 1152).
    bg = bg_image_encoded.squeeze(0)  # (729, 1152)
    obs = obs_tokens.squeeze(0)       # (729, 1152)
    img_tokens = torch.stack([bg, bg, bg, obs, bg, bg]).reshape(
        1, -1, vision_model.hidden_size)

    #Format proprioception to the 128-dim unified state vector.
    proprio_norm = (proprio - STATE_MIN) / (STATE_MAX - STATE_MIN) * 2 - 1
    state_vec = torch.zeros(1, 1, 128, device=device, dtype=dtype)
    state_vec[0, 0, MANISKILL_INDICES] = proprio_norm.to(device=device, dtype=dtype)
    state_input_actual = state_vec
    state_input_baseline = torch.zeros_like(state_vec)

    #Adapt lang+img once; state_traj_actual is the post-adaptor state token used
    #in the frozen-state path of vision/lang IG. state_adaptor is intentionally
    #NOT pre-applied to the IG-interpolated state; forward_fn_state calls it
    #inside so gradients flow back to the raw 128-dim state_vec.
    with torch.no_grad():
        lang_adapted = runner.lang_adaptor(lang_tokens)
        img_adapted = runner.img_adaptor(img_tokens)
        state_traj_actual = runner.state_adaptor(
            torch.cat([state_vec, action_mask], dim=2))
        lang_adapted_bl = runner.lang_adaptor(lang_tokens_baseline)
        img_adapted_bl = runner.img_adaptor(img_tokens_baseline)

    #Seeded wrapper, re-seeds inside each call so diffusion noise is identical
    #across all m+1 interpolation forwards and across the three modalities.
    def seeded_conditional_sample(lang_c, img_c, state_t):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return runner.conditional_sample(
            lang_c, lang_attn_mask, img_c, state_t, action_mask, ctrl_freqs)

    #Reference action, the Gaussian's mean μ and the chunk the policy would execute.
    #When frozen_ref_action is provided (frozen-target C1), skip the forward pass
    #and use the caller's ref_action so log_pi targets the original model's action.
    if frozen_ref_action is not None:
        ref_action = frozen_ref_action.to(device=device, dtype=dtype)
    else:
        with torch.no_grad():
            ref_action = seeded_conditional_sample(
                lang_adapted, img_adapted, state_traj_actual).detach()

    #IG target (scalar function of the predicted action chunk). Five variants,
    #selectable via `target`, all reduced over the 8 MANISKILL_INDICES joints:
    #  logpi   -0.5/σ²·mean((a-μ)²)   Gaussian log-density (default).
    #          Deletion magnitudes are structurally small under this target: the
    #          perturbation does move μ (4-8% of the action norm at top-5%
    #          deletion, Month 5 measurement), but the quadratic per-entry-mean
    #          readout maps that displacement to ~1e-3 nats (paper Sec. 6.1).
    #          Kept as the canonical/comparison target.
    #  l2      -‖a-μ‖₂                 raw L2 distance (not squared); larger
    #          gradient magnitude away from μ than the squared form.
    #  l2sq    -‖a-μ‖₂²               squared L2 sum (logpi without the 0.5/σ²/mean
    #          averaging) — sharper than logpi, same minimum.
    #  maxdev  -max_j|a_j-μ_j|        worst-joint deviation; sensitive to the
    #          single most-perturbed joint rather than the average.
    #  cosine  cos(a, μ)              directional similarity of the action chunk,
    #          scale-invariant; tests whether perturbations rotate the action.
    #All are maximized at a=μ (cosine at 1, the rest at 0) so the reference action
    #remains F's argmax and completeness still references F(input)-F(baseline).
    def _target(pred_action):
        a = pred_action[..., MANISKILL_INDICES].float()
        mu = ref_action[..., MANISKILL_INDICES].float()
        if target == "logpi":
            return -0.5 / sigma_sq * ((a - mu) ** 2).mean()
        if target == "l2":
            return -torch.sqrt(((a - mu) ** 2).sum() + 1e-12)
        if target == "l2sq":
            return -((a - mu) ** 2).sum()
        if target == "maxdev":
            return -(a - mu).abs().max()
        if target == "cosine":
            return torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0), mu.flatten().unsqueeze(0)).squeeze()
        raise ValueError(f"unknown target: {target}")

    log_pi = _target  # name kept for downstream callers (build_forward_fns, etc.)

    return {
        "runner": runner,
        "vision_model": vision_model,
        "img_tokens": img_tokens,
        "img_tokens_baseline": img_tokens_baseline,
        "lang_adapted": lang_adapted,
        "lang_adapted_bl": lang_adapted_bl,
        "img_adapted": img_adapted,
        "img_adapted_bl": img_adapted_bl,
        "state_input_actual": state_input_actual,
        "state_input_baseline": state_input_baseline,
        "state_traj_actual": state_traj_actual,
        "ref_action": ref_action,
        "seeded_conditional_sample": seeded_conditional_sample,
        "log_pi": log_pi,
        "action_mask": action_mask,
        "ctrl_freqs": ctrl_freqs,
        "lang_attn_mask": lang_attn_mask,
        "seed": seed,
        "sigma_sq": sigma_sq,
    }


def build_forward_fns(ctx):
    """
    Return (forward_fn_vision, forward_fn_language, forward_fn_state) for a
    prepared context. Each is a scalar function of one modality with the other
    two frozen, used both by `integrated_gradients` during IG and by
    perturbation studies (e.g. faithfulness.py replaces the top-|IG| tokens
    with baseline embeddings and re-evaluates forward_fn_vision to get
    Δlog π).
    """
    runner = ctx["runner"]
    lang_adapted = ctx["lang_adapted"]
    img_adapted = ctx["img_adapted"]
    state_traj_actual = ctx["state_traj_actual"]
    action_mask = ctx["action_mask"]
    seeded = ctx["seeded_conditional_sample"]
    log_pi = ctx["log_pi"]

    def forward_fn_vision(img_cond):
        return log_pi(seeded(lang_adapted, img_cond, state_traj_actual))

    def forward_fn_language(lang_cond):
        return log_pi(seeded(lang_cond, img_adapted, state_traj_actual))

    def forward_fn_state(state_vec_in):
        state_with_mask = torch.cat([state_vec_in, action_mask], dim=2)
        state_token = runner.state_adaptor(state_with_mask)
        return log_pi(seeded(lang_adapted, img_adapted, state_token))

    return forward_fn_vision, forward_fn_language, forward_fn_state


def compute_ig_for_step(
    runner,
    vision_model,
    obs_image,
    proprio,
    lang_tokens,
    lang_attn_mask,
    lang_tokens_baseline,
    bg_image_encoded,
    img_tokens_baseline,
    action_mask,
    ctrl_freqs,
    seed=42,
    m=64,
    sigma_sq=1.0,
    target="logpi",
):
    """
    Run per-modality IG for a single observation.

    Args:
        runner: preloaded, frozen, gradient-checkpointed RDTRunner
        vision_model: preloaded SigLIP (stays on GPU, encodes new obs per call)
        obs_image: PIL Image 384x384, freshly rendered env observation
        proprio: (8,) tensor, raw joint positions from ManiSkill qpos[:, :8]
        lang_tokens: (1, 1024, 4096) bf16, T5-XXL encoding of the task instruction
        lang_attn_mask: (1, 1024) bool, real task's attention mask
        lang_tokens_baseline: (1, 1024, 4096) bf16, [PAD, EOS] two-token T5 encoding
            (file name says bos_eos, but T5 has no BOS; the tokens are PAD and EOS)
        bg_image_encoded: (1, 729, 1152) bf16 on GPU, SigLIP tokens for ONE
            bg_image slot (used as 5 of 6 input slots and 5 of 6 baseline slots)
        img_tokens_baseline: (1, 4374, 1152) bf16 on GPU, full 6-slot baseline
            tensor pre-assembled as [bg, bg, bg, gray, bg, bg]
        action_mask: (1, 1, 128) bf16 on GPU, 1 at MANISKILL_INDICES else 0
        ctrl_freqs: (1,) bf16 on GPU, RDT control frequency tensor
        seed: int, shared across all forward passes within this step
        m: int, number of Riemann integration points (m+1 forwards per modality)
        sigma_sq: float, Gaussian variance for log π target

    Returns:
        dict with keys "ref_action", "ref_norm_maniskill", "vision", "language",
        "state". Each modality dict has "attribution" (torch.Tensor), "ig_sum",
        "expected_gap", "completeness_err" (decimal, not percent). The "state"
        dict additionally has "per_joint" (np.ndarray (8,) extracted at
        MANISKILL_INDICES).
    """
    ctx = prepare_ig_context(
        runner, vision_model, obs_image, proprio,
        lang_tokens, lang_attn_mask, lang_tokens_baseline,
        bg_image_encoded, img_tokens_baseline,
        action_mask, ctrl_freqs,
        seed=seed, sigma_sq=sigma_sq, target=target,
    )
    fwd_vision, fwd_lang, fwd_state = build_forward_fns(ctx)

    #IG per modality (integrated_gradients prints its own completeness check).
    vision_attr = integrated_gradients(
        fwd_vision, ctx["img_adapted"], ctx["img_adapted_bl"], m=m)
    lang_attr = integrated_gradients(
        fwd_lang, ctx["lang_adapted"], ctx["lang_adapted_bl"], m=m)
    state_attr = integrated_gradients(
        fwd_state, ctx["state_input_actual"], ctx["state_input_baseline"], m=m)

    #Per-modality summary stats. For logpi/l2/l2sq/maxdev, f_input is 0 by
    #construction (target is 0 at a=μ); for cosine it is 1. Compute f_input
    #explicitly per modality so completeness is correct for every target
    #(one extra cheap forward per modality).
    with torch.no_grad():
        f_in_vision = fwd_vision(ctx["img_adapted"]).item()
        f_in_lang = fwd_lang(ctx["lang_adapted"]).item()
        f_in_state = fwd_state(ctx["state_input_actual"]).item()
        f_base_vision = fwd_vision(ctx["img_adapted_bl"]).item()
        f_base_lang = fwd_lang(ctx["lang_adapted_bl"]).item()
        f_base_state = fwd_state(ctx["state_input_baseline"]).item()

    def _stats(attr, f_in, f_base):
        actual = attr.sum().item()
        expected = f_in - f_base
        rel_err = abs(expected - actual) / abs(expected) if expected != 0 else float("inf")
        return {
            "attribution": attr.detach(),
            "ig_sum": actual,
            "expected_gap": expected,
            "completeness_err": rel_err,
        }

    vision_stats = _stats(vision_attr, f_in_vision, f_base_vision)
    lang_stats = _stats(lang_attr, f_in_lang, f_base_lang)
    state_stats = _stats(state_attr, f_in_state, f_base_state)

    #Extract per-joint state attribution at MANISKILL_INDICES (real per-joint
    #values because IG ran on the raw 128-dim input, before state_adaptor).
    state_flat = state_attr.squeeze(0).squeeze(0).detach().cpu().float()
    state_stats["per_joint"] = state_flat[MANISKILL_INDICES].numpy()

    return {
        "ref_action": ctx["ref_action"],
        "ref_norm_maniskill": ctx["ref_action"][..., MANISKILL_INDICES].float().norm().item(),
        "vision": vision_stats,
        "language": lang_stats,
        "state": state_stats,
    }
