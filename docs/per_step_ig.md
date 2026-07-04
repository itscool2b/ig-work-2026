# per_step_ig.py / per_step_attribution.py

Lifts Integrated Gradients out of the one-shot `ig_rdt.py` demo and runs it at every policy call during an actual ManiSkill episode (PickCube, StackCube, PegInsertionSide, PickSingleYCB). Writes one summary JSONL row per call plus a sidecar `.pt` file with the full per-modality attribution tensors. This is the raw data every downstream metric (O1 correctness per-step, O2 faithfulness, O3 sanity) reads from. Phase A of Month 3 refactored `per_step_attribution.py` to expose two public helpers (`prepare_ig_context` and `build_forward_fns`) so `faithfulness.py` and `sanity.py` can rebuild the IG context without replaying the env loop.

## What's in each file

- `per_step_attribution.py` — three public surfaces: (1) `compute_ig_for_step(runner, vision_model, obs_image, proprio, lang_tokens, lang_attn_mask, lang_tokens_baseline, bg_image_encoded, img_tokens_baseline, action_mask, ctrl_freqs, seed, m, sigma_sq, target)` returns a dict with `ref_action`, `ref_norm_maniskill`, and per-modality `{attribution, ig_sum, expected_gap, completeness_err}` (`target` defaults to `"logpi"`; the Month 4 ablation passes `l2` or `maxdev`). (2) `prepare_ig_context(...)` returns a context dict that `compute_ig_for_step` and the post-processing scripts share. (3) `build_forward_fns(ctx)` returns the three modality-specific closures (`forward_fn_vision`, `forward_fn_language`, `forward_fn_state`) used by IG. Reuses `integrated_gradients.integrated_gradients()` unchanged.
- `per_step_ig.py` — episode loop. Creates the env with `render_mode='rgb_array'`, `control_mode="pd_joint_pos"`, `max_episode_steps=400` (matches RDT's official eval; the Bug 1 + Bug 2 fixes landed in Phase C; the original m=64 pass predates this fix and ran under the per-task ManiSkill default caps, 50 steps on three tasks, with PegInsertionSide's larger default yielding its seven policy calls per episode). Pre-encodes the constant SigLIP tokens (bg slots + baseline), loads RDT (170M or 1B per `--model`; for 1B prefers the **authors' fully fine-tuned weights** at `checkpoints/rdt_maniskill_authors/mp_rank_00_model_states.pt` if present, falls back to HF base + our LoRA), applies gradient checkpointing once (skippable via `--no-checkpoint` on cards with ≥ 24 GB VRAM for ~1.4× per-call speedup; mandatory on 12 GB), then runs `--episodes` episodes of policy-call IG with JSONL + sidecar output.

CLI flags added in Phase A: `--seed-base` (replaces the module-constant `SEED_BASE`; episode `e` uses `seed = seed_base + e`), `--resume` (skips episodes whose `episode_end` row exists in the `--out` JSONL; requires explicit `--out`; an interrupted episode without that row is re-run from scratch and its earlier partial rows remain in the appended file, which is the source of the duplicate rows documented in `data/README.md`), `--video-dir <path>` (wraps env with `RecordEpisode` and saves one mp4 per episode). Sidecar gained a `proprio` field so the post-processing scripts don't need to replay the env.

## Usage

```bash
# Smoke test (1 episode, m=32, ~30s/call)
.venv/bin/python per_step_ig.py --episodes 1 --m 32 --model 170m --max-policy-calls 3 --out data/metrics_smoke.jsonl

# Dev pass (10 episodes, m=64, ~4h wall at 25 policy calls/episode)
.venv/bin/python per_step_ig.py --episodes 10 --m 64 --model 170m

# 1B scale cross-check (slow — one episode is enough to confirm)
.venv/bin/python per_step_ig.py --episodes 1 --m 64 --model 1b
```

Output paths:

- `data/metrics_{task}_{model}_seed{seed_base}[_{target}][_T{T}]_{timestamp}.jsonl` — one summary row per policy call, plus one `event=episode_end` row per episode (`--out` overrides the auto name)
- `data/per_step_attr/{task}_{model}_s{seed_base}[_{target}][_m{m}][_T{T}]_ep{E:03d}_t{T:02d}.pt` — sidecar with full attribution tensors, ref_action, and observation image (seed, target, integration-budget, and solver-step tags prevent overwrites across runs)

## Why per-policy-call, not per-env-step

RDT predicts a 64-step action chunk per inference, subsampled 4× to 16 actual env steps. Under the original 50-step TimeLimit a PickCube episode was four policy calls; after the `max_episode_steps=400` fix a failing episode truncates at 25 policy calls. One policy-call attribution costs about a minute on 170M at m=64, so per-env-step IG would multiply a multi-hour 50-episode pass by the 16 subsampled steps per chunk. And semantically, the chunk is the unit the policy decides — the 16 env steps are its own interpolation. If downstream ever wants per-env-step granularity, the chunk attributions can be interpolated at read time.

## Why RDT-170M by default

The protocol uses RDT-170M for dev and RDT-1B for scale confirmation. 170M is ~2.7× faster per IG call on matched hardware (the committed runs, recorded on different machines, show ~2.2×) (smaller hidden_size and depth, same 5-step DPM chain). No ManiSkill LoRA checkpoint exists for 170M — we run pretrained-only. Task success drops versus 1B+LoRA (the model has never seen ManiSkill), but IG attributes whatever the model does, success or fail. 1B stays behind the `--model 1b` flag for the eventual scale confirmation pass.

### 170M config is not what the initial spec said

An earlier internal spec listed `hidden_size=1024, depth=14, num_heads=16` for RDT-170M. The **actual** published checkpoint at `robotics-diffusion-transformer/rdt-170m` has `num_heads=32` (head_dim=32). Verified by loading the state dict and reading parameter shapes:

```
model.blocks.0.attn.q_norm.weight: (32,)      → head_dim = 32
model.blocks.0.attn.qkv.weight:    (3072, 1024) → hidden_size = 1024
model.blocks.0.attn.proj.weight:   (1024, 1024) → hidden_size = 1024
max block index: 13                            → depth = 14
```

`configs/base_170m.yaml` uses `num_heads=32`. If you see a `q_norm size mismatch: (32) vs (64)` error during weight load, the config has drifted back to the old 16-head number.

## The loop structure

Forked from `eval_maniskill.py` lines 67–100. Per episode:

1. `env.reset(seed=SEED_BASE + ep)`.
2. Render first frame with `env.render()`, push into a 2-frame deque (`[None, frame_0]` at t=0; no history yet).
3. Until terminated/truncated or `call_idx == max_policy_calls`:
    - Extract proprio: `obs["agent"]["qpos"][0, :8].cpu()` (7 arm + 1 gripper, same as RDT eval reference).
    - Call `compute_ig_for_step(...)` with the current `obs_image` and `proprio`, the cached `bg_image_encoded` and `img_tokens_baseline`, the task's `lang_tokens`, and the BOS/EOS `lang_tokens_baseline`.
    - Save sidecar `.pt` with all four tensors (`vision_attr`, `lang_attr`, `state_attr`, `ref_action`) plus the `obs_image`.
    - Append one JSONL summary row.
    - Take the returned `ref_action`, slice `MANISKILL_INDICES` to (64, 8), denormalize to physical joint space with `ACTION_MIN`/`ACTION_MAX`, subsample 4× to (16, 8), step the env 16 times.
    - Push each new rendered frame onto the deque.
4. After the episode ends, write one `event=episode_end` row with the outcome.

## Why the observation image must go through `env.render()`

The first working version of `ig_rdt.py` used `obs_mode="state_dict"` without `render_mode`, then substituted a synthetic solid-color PIL placeholder "because ManiSkill rendering has sapien/mplib issues on this system." That comment was stale — rendering works fine with `render_mode="rgb_array"` + `env.render()`. The same path is used by `generate_demos.py` and `eval_maniskill.py`, so compatibility was already exercised. `per_step_ig.py` uses this path from the start.

## The action conversion

`ref_action` returned by `runner.conditional_sample` lives in RDT's 128-dim unified state space, normalized to `[-1, 1]`. ManiSkill's Panda controller expects an `(1, 8)` action in physical joint units. The conversion mirrors `RoboticDiffusionTransformerModel._unformat_action_to_joint` in `~/rdt-repo/scripts/maniskill_model.py:187`:

```python
action_unified = attr["ref_action"].squeeze(0).float().cpu()      # (64, 128)
action_joints = action_unified[:, MANISKILL_INDICES]                # (64, 8), in [-1, 1]
action_denorm = (action_joints + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
actions_np = action_denorm[::4, :].numpy()                          # (16, 8)
for action in actions_np:
    env.step(action.reshape(1, 8))
```

`ACTION_MIN`/`ACTION_MAX` are the same 8-dim bounds RDT uses internally (`maniskill_model.py:28`, `DATA_STAT['action_min'/'action_max']`).

## Why `.float().cpu().numpy()` and not `.cpu().numpy()`

numpy has no native bfloat16 dtype. `ref_action` is bf16 to match the rest of the pipeline, so converting to numpy directly throws `TypeError: Got unsupported ScalarType BFloat16`. Upcasting to fp32 first (`.float()`) is essentially free on small tensors.

## JSONL schema

Two row types share one file, distinguished by the `event` field. Summary rows (one per policy call) carry per-modality `vision/lang/state` `_err`/`_gap`/`_ig_sum` fields plus `ref_norm_maniskill`, `attr_file`, and `wall_seconds`. `event=episode_end` rows (one per episode) carry `success`, `terminated`, `truncated`, `env_steps`, and `policy_calls`. `data/README.md` maps every released JSONL to the run that produced it.

Full attribution tensors are NOT inlined — a single vision attribution is `(1, 4374, 2048) × bf16 ≈ 18 MB` (36 MB in fp32). Inlining 250 of them yields a ~5 GB JSONL file and breaks `jq`/`grep`/`tail`. Sidecars keep the JSONL tiny (~100 KB for a full dev pass) and the notebook loads sidecars only when building figures.

## Known convergence caveat

Completeness at t=0 of each episode is noise-dominated. The first policy call has `||ref[MANISKILL]|| ≈ 7–9` and corresponding F-gaps of ~0.004 nats — at that scale, bf16 accumulation precision and the ~65-point Riemann sum dominate the relative error. Later calls in the same episode have `|ref|` growing to 20+ and F-gaps growing proportionally, and completeness drops into the 1–6% range. This is not a bug in the IG code; it's a signal-to-noise issue in a nearly-flat log-density region around the seed observation. If the ≤3% threshold becomes binding, the fix is to run on 1B (stronger signal) or raise m, not to change the integration.

## File dependencies

```
per_step_ig.py
  imports: per_step_attribution.py
           configs/base_170m.yaml  (or ~/rdt-repo/configs/base.yaml for --model 1b)
           ~/rdt-repo/models/rdt_runner.py
           ~/rdt-repo/models/multimodal_encoder/siglip_encoder.py
  reads: data/lang_embeds/{task}.pt, data/lang_embeds/baseline_bos_eos.pt
         checkpoints/rdt_maniskill_authors/mp_rank_00_model_states.pt  (1B, preferred when present; used for the paper's 1B results)
         checkpoints/rdt_maniskill_lora/final.pt  (1B fallback with HF base; skipped on 170m)
         HF: robotics-diffusion-transformer/rdt-{170m,1b}/pytorch_model.bin
  writes: data/metrics_*.jsonl, data/per_step_attr/*.pt

per_step_attribution.py
  imports: integrated_gradients.py
           ~/rdt-repo/configs/state_vec.py (for STATE_VEC_IDX_MAPPING)
```
