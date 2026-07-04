# ig_rdt.py

Integrated Gradients on RDT-1B (Robotics Diffusion Transformer). A 1.2B-parameter Vision-Language-Action model that predicts robot actions via diffusion. Attributes the **full 5-step DPM-Solver++ denoising chain's** action prediction to three input modalities: vision, language, and proprioceptive state.

This is fundamentally different from the other four models. RDT outputs continuous robot actions, not class logits or next-token probabilities. There is no softmax. The scalar target for IG is **log π(a_t | o_t) under a Gaussian around the final denoised action**, restricted to the active ManiSkill joint dimensions: `log π(a | o) = -½·mean((a − μ)²)/σ²`, equivalently `-‖a − μ‖²/(2σ²D)` with `D = 512` active entries — the per-entry-mean convention of the paper's Eq. 3, implemented by the code's `.mean()` (dropping the constant entropy term, which does not affect IG gradients; a sum-form target without the 1/D factor would scale every reported Δlog π by 512). With σ²=1 this is numerically proportional to the earlier −MSE placeholder target but is now grounded in the standard predict-x₀ diffusion-policy log-probability, which the downstream faithfulness metric (Δlog p in nats) depends on.

## RDT-1B architecture

RDT-1B is a conditional diffusion model. During inference it starts from random noise and iteratively denoises over 5 steps to produce a 64-step action trajectory. The denoising is conditioned on three inputs:

- Vision: 6 images (2 timesteps x 3 cameras) encoded through SigLIP-SO400M into 4374 tokens (6 x 729 patches per image)
- Language: task instruction encoded through T5-v1_1-XXL into up to 1024 tokens
- State: 8-dim robot proprioception (7 arm joints + 1 gripper) in a 128-dim unified vector

These conditions pass through adaptor MLPs (img_adaptor, lang_adaptor, state_adaptor) into the DiT backbone (28 transformer layers, 2048-dim hidden size, 32 attention heads). The backbone alternates cross-attention between language and image conditions every other layer.

The model predicts clean actions directly (prediction_type="sample"), not noise.

## How IG works on a diffusion model

Standard IG requires a single differentiable forward pass from input to scalar output. RDT's full inference involves a 5-step DPM-Solver++ denoising chain — but the entire chain (`runner.conditional_sample` in `~/rdt-repo/models/rdt_runner.py`) is fully differentiable: the scheduler does pure tensor math with no `.detach()` calls, and `prediction_type="sample"` keeps the gradient path clean. So we backprop through all 5 steps.

The forward function calls `runner.conditional_sample` directly, bypassing `runner.predict_action` (which is wrapped in `no_grad` in inference scripts), runs the full chain, and returns:

```
log π(a_t | o_t) = -½ · mean((pred_action[..., MANISKILL_INDICES] − ref_action[..., MANISKILL_INDICES])²) / σ²
```

with σ²=1. The `ref_action` is computed once at script start with the *real* (non-interpolated) inputs, under `no_grad`, with a fixed seed — it is the mean μ of the Gaussian. All IG perturbations are then measured as log-probability of μ under the Gaussian centered at `conditional_sample(perturbed_input)`. This makes attributions answer "which inputs steer the model toward the action it actually produces" in a unit-correct log-probability, so downstream faithfulness metrics (Δlog p under top-k removal) read in nats.

### Why a fixed seed

`conditional_sample` initializes its denoising loop with `noisy_action = torch.randn(...)`, drawing from the global RNG. To make repeated calls deterministic (so the IG completeness check holds and so all 301 forward passes per modality use the same starting noise), we wrap the call in a helper that re-seeds CPU and CUDA generators immediately before each invocation:

```python
def seeded_conditional_sample(lang_c, img_c, state_t):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return runner.conditional_sample(
        lang_c, lang_attn_mask, img_c, state_t, action_mask, ctrl_freqs)
```

`DPMSolverMultistepScheduler.set_timesteps` is called at the top of every `conditional_sample` invocation and resets the scheduler's internal state (`model_outputs`, `lower_order_nums`), so back-to-back calls don't accumulate stale state.

## Per-modality IG

We run IG three separate times. Each time, one modality's adapted embeddings are interpolated from baseline to actual, while the other two are held frozen and the seeded `conditional_sample` is called to produce a fresh action prediction at each interpolation point.

1. Vision IG: interpolate `img_adapted` from gray_baseline to actual, freeze lang + state
2. Language IG: interpolate `lang_adapted` from the encoded BOS/EOS baseline to actual, freeze img + state (see Baselines below; the original zeros baseline was retired)
3. State IG: interpolate the **raw 128-dim `state_vec`** (pre-adaptor) from zeros to actual; `state_adaptor` runs inside `forward_fn_state`; freeze img + lang. See the dedicated subsection below for why this must happen pre-adaptor.

Each call uses m=300 Riemann steps. The denoising noise inside `conditional_sample` is held fixed across all interpolation points via the seeded wrapper, so the only thing varying is the modality under attribution.

## Baselines

- Vision: gray image (128,128,128) in **slot 3 only** (external camera at time t), with the five other image slots holding the same SigLIP mean-color `bg_image` as the input. This means the only non-trivial interpolation in vision IG is `obs_image → gray_image` on the external camera; the five non-camera slots contribute zero attribution by construction, and the reported vision completeness number matches what the heatmap visualizes. All six slots are encoded through SigLIP + img_adaptor.
- Language: **BOS/EOS minimal sequence** (`[PAD, EOS]` = `[0, 1]`) encoded through T5-v1_1-XXL + lang_adaptor, pre-generated offline by `encode_task_lang.py` and saved to `data/lang_embeds/baseline_bos_eos.pt`
- State: zeros in the raw 128-dim `state_vec` (pre-adaptor). `state_adaptor` is a plain MLP with GELU (no RMSNorm) so zeros are numerically safe, and interpolating pre-adaptor is required to get per-joint attribution (see State IG subsection below).

The language baseline was originally zeros through lang_adaptor, which is numerically safe (adaptor is an MLP with GELU, no RMSNorm) but semantically degenerate — it sat the adaptor output at a point that produced very different action predictions than any real T5 encoding, inflating the `F(input) − F(baseline)` gap and dragging Riemann-sum completeness error up. Switching to a minimal BOS/EOS sequence through the real T5 encoder gives a baseline that's close to a true "empty prompt" in embedding space, and in practice cut language completeness error nearly in half (13.71% → 6.97%).

The BOS/EOS file and the real task embeddings are both saved as `(1, 1024, 4096)` bf16 tensors padded with zeros. The per-baseline attention mask is also saved but not used during IG — the forward function uses the real task's `lang_attn_mask` for both baseline and real calls, because IG completeness requires the same function (same mask) evaluated at both endpoints and along the interpolation path.

## Script walkthrough

### Loading sequence (VRAM management)

The script loads models in sequence to stay under 12 GB:

1. Load SigLIP (~1.8 GB). Encode the observation image and gray baseline image into tokens. Unload SigLIP. VRAM drops to 0.03 GB.
2. Load RDT-1B backbone (~2.5 GB). Load LoRA fine-tuned weights if available. VRAM at ~2.5 GB.
3. Load pre-encoded T5-XXL embeddings from disk (no T5 model in VRAM, just the embedding tensors).

Peak VRAM during IG: **does not fit** on a 12 GB card without gradient checkpointing — vision IG OOMs during backward at ~10.8 GB allocated (measured: tried to allocate 36 MiB with 46 MiB free). The full 5-step chain keeps activations from all 5 DiT forward passes in the autograd graph, and with the 4374-token image condition flowing through 28 cross-attention layers, the activation memory overflows. `ENABLE_GRADIENT_CHECKPOINTING` is therefore **enabled by default** at the top of the script. With checkpointing on, the script fits comfortably.

### Observation

Both proprioception and the camera frame are pulled from a ManiSkill3 PickCube-v1 environment. The env is created with `obs_mode="state_dict"` for proprio plus `render_mode="rgb_array"` so `env.render()` returns a real scene:

```python
env = gym.make("PickCube-v1", obs_mode="state_dict", num_envs=1, render_mode="rgb_array")
obs, _ = env.reset(seed=42)
proprio = obs["agent"]["qpos"][0, :8].cpu()                     # 7 arm joints + 1 gripper finger
img_array = env.render().squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (512, 512, 3)
obs_image = Image.fromarray(img_array).resize((384, 384))       # match SigLIP patch-14 input
env.close()
```

Proprio extraction matches `~/rdt-repo/eval_sim/eval_rdt_maniskill.py` (`obs['agent']['qpos'][:, :-1]`). Earlier versions used `obs_mode="state"` and sliced `obs[0, :8]` off the 42-dim flat state vector, which actually mixes joint positions with joint velocities and object pose — the fix is to go through the dict so you get qpos specifically.

The render path is the same one `generate_demos.py` and `eval_maniskill.py` use, so compatibility is already exercised on this system. An earlier version of this script substituted a solid-color PIL placeholder on the assumption that rendering was broken by sapien/mplib compatibility issues; in practice, `env.render()` with `render_mode="rgb_array"` works fine and the placeholder is gone.

For the 6-camera input that RDT expects, only slot 3 (external camera at time t) gets the rendered observation. The other five slots (wrist cameras, previous timestep) get SigLIP's mean-color `bg_image` in both the input AND the baseline — see the Baselines section above for why this asymmetric layout is load-bearing for the vision completeness number.

### State formatting

ManiSkill's Panda arm exposes 9 qpos values (7 arm + 2 gripper fingers); we keep the first 8 (7 arm + 1 gripper finger). These map into RDT's 128-dim unified state vector at specific indices defined in `configs/state_vec.py`:

```python
MANISKILL_INDICES = [0, 1, 2, 3, 4, 5, 6, 10]  # right_arm_joint_0-6_pos + right_gripper_open
```

Values are normalized to [-1, 1] using precomputed min/max stats from the ManiSkill training data. All other 120 dimensions are zero (unused for single-arm manipulation).

### Adapted embeddings

The RDTRunner has three adaptor MLPs that project each modality into the 2048-dim DiT hidden space. We call `lang_adaptor` and `img_adaptor` directly on their inputs so vision / language IG can interpolate their 2048-dim outputs. `state_adaptor` is treated differently — see the State IG subsection.

```python
lang_adapted = runner.lang_adaptor(lang_tokens)              # (1, lang_len, 2048)
img_adapted = runner.img_adaptor(img_tokens)                 # (1, 4374, 2048)
state_traj_actual = runner.state_adaptor(
    torch.cat([state_vec, action_mask], dim=2))              # (1, 1, 2048), reference-only
```

The state_adaptor takes `state_token_dim * 2 = 256` because the input is the state vector concatenated with a binary mask indicating which of the 128 dims are active. `state_traj_actual` above is built purely so the reference action can be computed with the real state token; state IG itself does NOT interpolate it (it interpolates the pre-adaptor 128-dim vector — see below).

### State IG: why the raw 128-dim vector, not the post-adaptor token

`state_adaptor` is configured as `mlp3x_gelu` in `~/rdt-repo/configs/base.yaml`, which expands to:

```
Linear(256 -> 2048) -> GELU -> Linear(2048 -> 2048) -> GELU -> Linear(2048 -> 2048)
```

Two nonlinear layers. If state IG interpolated the post-adaptor `(1, 1, 2048)` token with baseline zeros, attribution would be computed per output channel of the MLP. Each of the 2048 output channels is a nonlinear mixture of all 128 input dims, so there is no way to project attribution back to the 8 physical joints — the best you could plot was a uniform `state_total / 8` per joint, which has zero per-joint information.

The fix is to interpolate the raw 128-dim `state_vec` (with baseline zeros) and call `state_adaptor` *inside* `forward_fn_state`:

```python
state_input_actual   = state_vec                          # (1, 1, 128)
state_input_baseline = torch.zeros_like(state_vec)        # (1, 1, 128)

def forward_fn_state(state_vec_in):
    state_with_mask = torch.cat([state_vec_in, action_mask], dim=2)  # (1,1,256)
    state_token = runner.state_adaptor(state_with_mask)              # (1,1,2048)
    return _log_pi(seeded_conditional_sample(lang_adapted, img_adapted, state_token))
```

Autograd's chain rule carries gradients back through the adaptor MLP to each of the 128 input dims, and slicing the attribution at `MANISKILL_INDICES` gives one real scalar per joint:

```python
state_attr_flat     = state_attr.squeeze(0).squeeze(0).cpu().float()   # (128,)
state_attr_per_joint = state_attr_flat[MANISKILL_INDICES].numpy()      # (8,)
```

The 120 dims outside `MANISKILL_INDICES` are identical (zero) at both input and baseline, so their path integrand is zero and their attribution is zero by construction. The sum of the 8 per-joint attributions therefore equals `sum(state_attr_flat)` up to float noise, which is what the completeness axiom wants.

### The deeper reason post-projection doesn't work

"The 2048 channels mix input dims nonlinearly" is the intuitive answer but it glosses over why no amount of matrix magic on the 2048-d attribution gives you back the 8 per-joint numbers. The real reason is that IG's attribution is *defined* by the specific path you integrate along, and the two paths live in different spaces:

- Post-adaptor IG integrates along a straight line in 2048-d space, from `T(0)` to `T(state_vec)`, where `T` is the adaptor.
- Pre-adaptor IG integrates along a straight line in 128-d space, from `0` to `state_vec`, with `T` evaluated at each point along the way.

Because `T` is nonlinear (two GELUs), the image of the 128-d straight line is a *curve* in 2048-d space, not a straight line. So the two paths are genuinely different curves. IG attributes along whichever curve you picked; it does not promise that attributions computed on one curve transform nicely into attributions on another.

The completeness axiom makes this concrete. IG guarantees:

```
sum(attr) = F(input) - F(baseline)
```

at whatever parameterization you integrate over. Both parameterizations give the same scalar on the right side (the same F evaluated at the same two points), so both yield the same *total* attribution. What differs is the decomposition. In 2048-d, the total is split across 2048 output channels. In 128-d, it is split across 128 input dims. You cannot reshape or linearly project one decomposition into the other because neither one is a linear transformation of the other — they are separate applications of a theorem that happens to agree on the total.

Concretely: if you computed the Jacobian `J = dT/dx ∈ R^{2048 x 128}` at every alpha along the post-adaptor path and multiplied the 2048-d attributions through it, you would *not* recover the 128-d IG attributions, because you would be composing attributions-along-one-curve with gradients-at-a-different-curve. The only way to get per-joint attribution that respects the completeness axiom is to run IG in the 128-d parameterization from the start. That is what the refactor does.

### Why `action_mask` is held fixed

`action_mask` is structural metadata, not signal. It is the binary indicator that says "these 8 of my 128 dims are valid for the Panda embodiment" and zeros everywhere else. If IG interpolated it, at `alpha = 0.5` the mask would be 0.5, an undefined "half-active" indicator the model was never trained to handle. More importantly: varying the mask means changing which robot you are talking to, which is not the attribution question we are asking. The question is "how much does each joint position matter", not "how much does the fact that we are driving a Panda matter".

The general convention, inherited from IG itself: interpolate only the payload (state values, language embeddings, image tokens), and hold structural scaffolding fixed (attention masks, control frequencies, positional embeddings, embodiment indicators). The same thing is happening in `forward_fn_language`: `lang_attn_mask` comes from the *real* task's encoding and is held fixed for both the baseline (BOS/EOS) call and the real call, even though it is "wrong" for the BOS/EOS baseline. This keeps the two endpoints of the path evaluating "the same function", with only the signal varying.

### Why `state_traj_actual` still exists

After the refactor, `state_traj_actual` (the post-adaptor `(1, 1, 2048)` token) is NOT the IG input. But the script still builds it, because IG's target is `log π(a_t | o_t) = -½·mean((pred_action − ref_action)²)/σ²` (per-entry mean over the D = 512 sliced entries, matching the code below) and we need `ref_action` (the Gaussian's mean μ) before IG can run. `ref_action` is `F(real_input)`, a one-off evaluation of the pipeline with the real state + real language + real image, under `no_grad`, with a fixed seed. That one call needs a post-adaptor state token to feed into `conditional_sample`. So:

- `state_traj_actual` exists only for this single reference call at script start.
- During state IG, every interpolated forward pass rebuilds its own state token from its interpolated 128-d input via `runner.state_adaptor(...)` inside `forward_fn_state`. The outer `state_traj_actual` is never touched by state IG.

You could refactor to compute the state_adaptor inline inside the reference call's `no_grad` block and drop the outer variable entirely. Keeping it named is just for code readability: it says "this is the real state token we used for the reference".

### Forward functions

```python
SIGMA_SQ = 1.0

def _log_pi(pred_action):
    diff = (pred_action[..., MANISKILL_INDICES]
            - ref_action[..., MANISKILL_INDICES]).float()
    return -0.5 / SIGMA_SQ * (diff ** 2).mean()

def forward_fn_vision(img_cond):
    return _log_pi(seeded_conditional_sample(lang_adapted, img_cond, state_traj_actual))
```

`conditional_sample` runs the full 5-step DPM-Solver++ chain, returning a `(1, 64, 128)` action trajectory already masked to the active dimensions. `_log_pi` slices the 8 ManiSkill indices across all 64 horizon steps, casts to fp32 (bf16 loses precision on small differences), and returns `log π(pred | o)` under a unit-variance Gaussian centered at `ref_action` (the Gaussian's mean μ), dropping the constant entropy term because it does not affect IG gradients.

The same pattern applies to `forward_fn_language` (interpolates `lang_cond` in the 2048-dim post-adaptor space). `forward_fn_state` is the exception — it interpolates the raw 128-dim `state_vec` and calls `state_adaptor` inside the forward; see the "State IG: why the raw 128-dim vector" subsection above.

### Gradient checkpointing

An `ENABLE_GRADIENT_CHECKPOINTING` flag near the top of the script wraps each `RDTBlock.forward` in `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, which recomputes block activations during backward instead of storing them. **Default is ON** — vision IG OOMs on a 12 GB card without it (empirically verified: the first run of the current script hit `torch.OutOfMemoryError` at 10.82 GB allocated). Checkpointing roughly doubles backward time but fits the full 5-step chain into memory, which is strictly required here.

Why this works with frozen params: `use_reentrant=False` only requires that one input to the checkpointed function carries grad. For state/lang/vision IG, the modality being interpolated has `requires_grad=True` and flows into every block via either `x` (state token in the state-action input) or `c` (the cross-attention condition tensor). So the gradient path through every checkpoint is non-empty.

## Supporting scripts

### encode_task_lang.py

Pre-encodes T5-v1_1-XXL embeddings offline. T5-XXL has ~4.8B encoder parameters (~9.6 GB in bfloat16), too large to keep loaded during IG.

Uses sentencepiece directly for tokenization to bypass a bug in transformers 5.x where `AutoTokenizer` and `T5Tokenizer` both trigger a failed tiktoken conversion on T5's spiece.model file.

Loads the encoder model with `device_map="auto"` which offloads layers to CPU/disk as needed. Encodes five task instructions (the four paper tasks plus the dropped TurnFaucet-v1, with a BOS/EOS baseline) and pads each to 1024 tokens (RDT's max language length). Saves to `data/lang_embeds/{task}.pt`.

Every saved `.pt` file contains three fields:

- `embeds` — padded T5 last hidden state, `(1, 1024, 4096)` bf16
- `attn_mask` — padded attention mask, `(1, 1024)` bool
- `tokens` — plain Python list of SentencePiece pieces, length equal to the number of real tokens (no padding). For example PickCube's tokens start `['▁', 'Gra', 's', 'p', '▁', 'a', '▁red', '▁cu', 'be', ..., '</s>']`.

`ig_rdt.py` reads the `tokens` field and uses it as the y-axis labels for the language attribution bar chart, replacing earlier `tok_i` placeholders. The load path uses `torch.load(..., weights_only=True)` — the `tokens` list is a plain list of strings and loads safely under the PyTorch 2.6+ safelist. If the file lacks the field (older .pt), the script falls back to `f"tok_{i}"` labels automatically.

Also encodes a **BOS/EOS minimal-sequence baseline** (`[PAD, EOS]` = `[0, 1]`) as `data/lang_embeds/baseline_bos_eos.pt` — consumed by `ig_rdt.py` as the language IG baseline. T5 has no formal BOS token, so we use PAD as a BOS stand-in; both positions have attention mask 1 so the encoder actually processes them. The resulting embedding at positions 0 and 1 is a real T5 encoder output for the minimal valid input, and positions 2-1023 are zero-padded (the model never attends to them because IG uses the real task's attention mask for both endpoints of the interpolation). Baseline tokens are `['<pad>', '</s>']`.

### generate_demos.py

Replays ManiSkill3's pre-built demonstration trajectories with RGB rendering to produce training data.

ManiSkill3 provides downloadable state-only demos via `python -m mani_skill.utils.download_demo PickCube-v1`. These contain actions and environment states but no images. The script creates an environment in RGBD mode, replays each trajectory by restoring the initial state and stepping with the recorded actions, and captures rendered frames.

The motion planning solver (mplib) segfaults on this system due to a C++ library compatibility issue. The replay approach bypasses this entirely since it only steps the environment with pre-recorded actions, no planning needed.

Output: `data/maniskill_demos/PickCube-v1.h5` with 200 trajectories. Each contains:
- images/frame_{t}: 384x384 RGB frames (resized from ManiSkill's 512x512 render)
- states: (T, 128) normalized proprioception in RDT's unified format
- actions: (T, 128) normalized actions in RDT's unified format

### finetune_rdt.py

LoRA fine-tuning of RDT-1B on ManiSkill3 PickCube demonstrations.

The 1B model cannot be fully fine-tuned on 12 GB VRAM (model + optimizer + activations > 14 GB). LoRA freezes the base model and trains small rank-16 adapters on the attention layers.

LoRA target modules (from inspecting the RDT DiT block structure):
- `attn.qkv` (fused self-attention Q/K/V, 6144x2048)
- `attn.proj` (self-attention output, 2048x2048)
- `cross_attn.q`, `cross_attn.kv`, `cross_attn.proj` (cross-attention layers)

SigLIP stays loaded during training for on-the-fly image encoding. Pre-encoding all images would require ~148 GB RAM (200 trajectories x ~74 frames x 10 MB per embedding). On-the-fly encoding uses ~1.8 GB GPU for SigLIP and processes one frame per step.

Total VRAM during training: SigLIP (1.8 GB) + RDT (2.5 GB) + LoRA + activations = ~6-8 GB.

Training: 5000 steps, batch_size=1 with gradient accumulation over 8 steps, lr=1e-4, AdamW. Loss converges to ~0.0001 within 3000 steps. Saves LoRA checkpoints every 1000 steps.

### eval_maniskill.py

Evaluation script that loads the fine-tuned model and runs episodes in ManiSkill3 PickCube-v1 to measure task success rate. Follows the same inference pattern as the RDT repo's eval_rdt_maniskill.py.

## Output

`output/ig_rdt.png`

Three-panel figure:
- Left: vision patch attribution heatmap (27x27 SigLIP patches for the external camera, upscaled and overlaid on the observation image).
- Center: language token attribution bar chart. Each bar is one T5 SentencePiece token (`▁Grasp`, `▁a`, `▁red`, `▁cu`, `be`, ..., `</s>`). Labels come from the `tokens` field saved by `encode_task_lang.py`; bars are sign-colored red/blue.
- Right: per-joint state attribution bar chart (7 joints + gripper). Real IG values extracted at `MANISKILL_INDICES` after running state IG on the raw 128-dim `state_vec`; sign-colored red/blue. Previous versions plotted a uniform `state_total / 8` per joint because IG was on the post-adaptor 2048-dim token (no per-joint information recoverable).

## Completeness

Measured with the `log π(a_t | o_t)` target (σ²=1 Gaussian around the final denoised mean), the real PickCube-v1 rendered observation, and the slot-3-only vision baseline:

| Modality | Error | `expected = f(input) − f(baseline)` | Notes |
|----------|-------|-------------------------------------|-------|
| Vision | 9.06% | 0.0481 | Full 5-step chain has 5× the activations of a single step, and the 4374-token image condition flows through every DiT block; Riemann sum has more surface area to under-integrate. Expected gap is now for `bg_image → gray` on slot 3 alone (the five non-camera slots are identical between input and baseline and contribute zero). |
| Language | 5.11% | 0.0019 | Very small gap: a BOS/EOS T5 encoding produces an action chain close to the real task's chain when conditioned on the real camera image, so the path integral is both short and smooth. Moved from the earlier 6.97% baseline because the reference action itself changed when the image source switched from a flat-color placeholder to a real scene. |
| State | 6.12% | 0.0749 | State IG interpolates the raw 128-dim pre-adaptor `state_vec` (so attribution is recoverable per joint), passing the path integral through the `mlp3x_gelu` adaptor — two GELU nonlinearities added to the path. Gap grew vs the placeholder-image run because a real scene produces a reference action that diverges more sharply from the zeros-state baseline. |

Reference action: `||ref[MANISKILL]|| = 9.4697`. The "expected" values are positive because the target is a log-probability and `f(input) = log π(μ | real) = 0` by construction (evaluating the forward function at the real input produces the Gaussian's mean μ, so the log-density at that point is the constant we dropped, i.e. zero with `SIGMA_SQ=1.0`). Hence `expected = 0 − f(baseline) = −f(baseline)`, which grows with how divergent the baseline's predicted trajectory is from μ.

Language has the smallest gap because a BOS/EOS T5 encoding already produces an action chain close to the real task's chain.

For comparison, the previous run used a synthetic solid-color placeholder image and the `−MSE` target; it reported Vision 8.93% / Language 6.97% / State 7.93% with `||ref|| = 8.6013`. The vision error is roughly unchanged (9.06%), language is now measured against a smaller gap (0.0019 vs 0.0098) because the real image brings the BOS/EOS-conditioned action closer to the reference, and state is measured against a larger gap (0.0749 vs 0.0485) for the opposite reason. Completeness *percentage* is invariant under the target's 0.5× rescaling; the measured shifts come entirely from the reference action and vision baseline changes.

## File dependencies

```
ig_rdt.py
  imports: integrated_gradients.py, ~/rdt-repo/models/*, ~/rdt-repo/configs/*
  reads: data/lang_embeds/PickCube-v1.pt, data/lang_embeds/baseline_bos_eos.pt, checkpoints/rdt_maniskill_lora/final.pt
  writes: output/ig_rdt.png

encode_task_lang.py -> data/lang_embeds/*.pt
generate_demos.py -> data/maniskill_demos/PickCube-v1.h5
finetune_rdt.py -> checkpoints/rdt_maniskill_lora/*.pt
```

The project scripts never modify the RDT repo at ~/rdt-repo. They import from it via sys.path.
