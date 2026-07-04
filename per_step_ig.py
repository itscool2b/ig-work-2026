"""
Per-step IG across a PickCube-v1 episode, writing one JSONL row per policy call
plus sidecar .pt files for the full per-modality attribution tensors.

Usage:
    .venv/bin/python per_step_ig.py --episodes 10 --m 64 --model 170m
    .venv/bin/python per_step_ig.py --episodes 1 --m 16 --model 170m  # smoke

Outputs:
    data/metrics_{task}_{model}_seed{seed_base}[_{target}][_T{n}]_{timestamp}.jsonl
        (step rows + episode_end rows; or the explicit --out path)
    data/per_step_attr/{task}_{model}_s{seed}[_{target}][_m{m}][_T{n}]_ep{E:03d}_t{T:02d}.pt
        (one per policy call; tags appear when non-default)

Granularity is per-policy-call, not per-env-step: RDT predicts a 64-step action
chunk that is subsampled 4x to 16 env steps, so one policy call corresponds to
~16 env steps. The chunk is what the policy "decides"; attributing it once per
chunk cuts the wall cost of any pass by a factor of 16 versus a
per-env-step pass.

Default model is RDT-170M (primary for dev, 1B for scale).
170M has no ManiSkill LoRA so it runs pretrained-only — task success will drop
vs the 1B+LoRA setup but IG attributes whatever the model does, success or
failure. Use `--model 1b` for a scale cross-check when needed.
"""

import sys
import os
import gc
import json
import time
import argparse
from collections import deque

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/rdt-repo"))
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower

from per_step_attribution import compute_ig_for_step, MANISKILL_INDICES

CONTROL_FREQ = 25

#Panda action bounds for denormalizing the 8-dim joint slice out of RDT's 128-dim
#unified action output. Taken verbatim from ~/rdt-repo/scripts/maniskill_model.py
#line 28 (DATA_STAT['action_min'/'action_max']) — RDT's _unformat_action_to_joint
#applies this same denormalization before passing actions to env.step.
ACTION_MIN = torch.tensor([-0.7472005486488342, -0.08631071448326111, -0.4995281398296356,
                           -2.658363103866577, -0.5751323103904724, 1.8290787935256958,
                           -2.245187997817993, -1.0])
ACTION_MAX = torch.tensor([0.7654682397842407, 1.4984270334243774, 0.46786263585090637,
                           -0.38181185722351074, 0.5517147779464722, 3.291581630706787,
                           2.575840711593628, 1.0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickCube-v1",
                   help="ManiSkill task id. Must have data/lang_embeds/{task}.pt.")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of episodes. 1 for smoke test, 10 for dev pass, 50 for the full pass.")
    p.add_argument("--m", type=int, default=64,
                   help="Riemann integration steps per modality. 16 for smoke, 64 for dev, 128 for figures.")
    p.add_argument("--model", choices=["170m", "1b"], default="170m",
                   help="RDT backbone size. Default 170m (dev); 1b for scale cross-check.")
    p.add_argument("--max-policy-calls", type=int, default=30,
                   help="Safety bound on policy calls per episode (prevents runaway episodes).")
    p.add_argument("--out", default=None,
                   help="Output JSONL path. Auto-named data/metrics_{task}_{model}_"
                        "seed{seed_base}[_{target}][_T{n}]_{ts}.jsonl if omitted.")
    p.add_argument("--video-dir", default=None,
                   help="If set, wrap env with RecordEpisode and save an mp4 per episode to this dir.")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing. Faster backward; needs more VRAM.")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Base seed. Episode e uses seed=seed_base+e. Two full-pass seeds "
                        "42 (default) and 142.")
    p.add_argument("--resume", action="store_true",
                   help="If --out JSONL already exists, skip episodes whose episode_end "
                        "row is present. Requires explicit --out so auto-named files "
                        "aren't accidentally resumed across runs.")
    p.add_argument("--target", default="logpi",
                   choices=["logpi", "l2", "l2sq", "maxdev", "cosine"],
                   help="IG target function. logpi (default, Gaussian log-density) "
                        "or an alternative-target ablation (Month 4): l2, l2sq, maxdev, cosine.")
    p.add_argument("--solver-steps", type=int, default=None,
                   help="Override the DPM-Solver++ denoising step count "
                        "(num_inference_timesteps, default 5 in the config). Month 5 "
                        "contraction-mechanism sweep uses {1,2,3,5,10,20}. None keeps the "
                        "config value and the historical output naming byte-for-byte.")
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    #Month 5: a non-default solver-step count tags every output so a T-sweep never
    #collides with the canonical T=5 data. Empty for the default so historical names
    #are unchanged.
    _t_tag = "" if args.solver_steps is None else f"_T{args.solver_steps}"
    #Auto-names carry a non-default target the same way the sidecars do, so an
    #alternative-target run launched without --out cannot masquerade as a base
    #logpi run to the downstream globs. Default "logpi" keeps historical naming.
    _tgt_tag_out = "" if args.target == "logpi" else f"_{args.target}"
    if args.out is None:
        if args.resume:
            print("ERROR: --resume requires explicit --out so the right JSONL is "
                  "resumed. Auto-named timestamped files cannot be resumed.")
            sys.exit(1)
        args.out = (f"data/metrics_{args.task}_{args.model}_seed{args.seed_base}"
                    f"{_tgt_tag_out}{_t_tag}_{timestamp}.jsonl")
    os.makedirs("data/per_step_attr", exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    #Resume: scan the JSONL for episode_end rows and skip those episodes below.
    #Partial episodes (episode_end missing) are re-run from scratch; that's simpler
    #than resuming mid-episode and the wall overhead is bounded by one episode.
    resume_episodes = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("event") == "episode_end":
                        resume_episodes.add(row["episode"])
                except json.JSONDecodeError:
                    pass
        print(f"resume: found {len(resume_episodes)} completed episodes in {args.out}")

    #----- SigLIP: load once, stay loaded (encodes new obs per call) -----
    print("loading SigLIP...")
    vision_model = SiglipVisionTower(vision_tower="google/siglip-so400m-patch14-384", args=None)
    vision_model = vision_model.to("cuda", dtype=torch.bfloat16)
    image_processor = vision_model.image_processor
    NUM_PATCHES = vision_model.num_patches  # 729

    background_color = tuple(int(x * 255) for x in image_processor.image_mean)
    bg_image = Image.new("RGB", (384, 384), background_color)
    gray_image = Image.new("RGB", (384, 384), (128, 128, 128))

    #Pre-encode one bg slot (used for 5 of 6 input slots, 5 of 6 baseline slots)
    bg_processed = image_processor.preprocess(bg_image, return_tensors="pt")["pixel_values"][0]
    bg_batched = bg_processed.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        bg_image_encoded = vision_model(bg_batched).detach()  # (1, 729, 1152)

    #Pre-encode full baseline: [bg, bg, bg, gray, bg, bg] → (1, 4374, 1152).
    #Layout matches the [t-1 ext, t-1 wrist_r, t-1 wrist_l, t ext, t wrist_r, t wrist_l]
    #ordering that RDT's image position embedding expects (see ig_rdt.py:54-55).
    gray_processed = image_processor.preprocess(gray_image, return_tensors="pt")["pixel_values"][0]
    baseline_stack = torch.stack([bg_processed, bg_processed, bg_processed,
                                  gray_processed,
                                  bg_processed, bg_processed]).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        baseline_tokens = vision_model(baseline_stack).detach()  # (6, 729, 1152)
        img_tokens_baseline = baseline_tokens.reshape(1, -1, vision_model.hidden_size)
    print(f"encoded bg slot: {bg_image_encoded.shape}, baseline full: {img_tokens_baseline.shape}")

    #----- RDT backbone -----
    if args.model == "170m":
        config_path = "configs/base_170m.yaml"
    else:
        config_path = os.path.expanduser("~/rdt-repo/configs/base.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    #Month 5 contraction-mechanism sweep: override the DPM-Solver++ step count in
    #OUR loaded config dict only, before RDTRunner reads it. The RDT library at
    #~/rdt-repo is never edited. RDTRunner builds the scheduler from
    #config["model"]["noise_scheduler"], so this is honored at construction.
    if args.solver_steps is not None:
        _old_T = config["model"]["noise_scheduler"]["num_inference_timesteps"]
        config["model"]["noise_scheduler"]["num_inference_timesteps"] = args.solver_steps
        print(f"OVERRIDE num_inference_timesteps {_old_T} -> {args.solver_steps}")

    print(f"loading RDT-{args.model} backbone from {config_path}...")
    runner = RDTRunner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=config["common"]["img_history_size"] * config["common"]["num_cameras"] * NUM_PATCHES,
        img_pos_embed_config=[("image", (config["common"]["img_history_size"],
                                         config["common"]["num_cameras"], -NUM_PATCHES))],
        lang_pos_embed_config=[("lang", -config["dataset"]["tokenizer_max_length"])],
        dtype=torch.bfloat16,
    )

    #Two weight paths supported on 1B:
    #  (1) HF base `rdt-1b` + our LoRA at `checkpoints/rdt_maniskill_lora/final.pt`
    #      -- the path that produced the per-step IG demo through Month 2.
    #  (2) Authors' fully fine-tuned ManiSkill weights at
    #      `checkpoints/rdt_maniskill_authors/mp_rank_00_model_states.pt` (DeepSpeed
    #      checkpoint, ~2.45 GB, 1.2B params bf16). We measured 76% PickCube
    #      success with these. Preferred when present because our own LoRA on top of
    #      the HF base does not match this success rate empirically (sub-SNR |ref|
    #      throughout 25-call episodes on seed 42).
    authors_path = "checkpoints/rdt_maniskill_authors/mp_rank_00_model_states.pt"
    if args.model == "1b" and os.path.exists(authors_path):
        print(f"loading authors' fine-tuned 1B weights from {authors_path}")
        ckpt = torch.load(authors_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["module"] if "module" in ckpt else ckpt
        #DeepSpeed checkpoints sometimes prefix keys with "module." — strip if so.
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        missing, unexpected = runner.load_state_dict(state_dict, strict=False)
        print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  first missing: {missing[:3]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:3]}")
        del state_dict, ckpt
        print("skipping LoRA wrap (authors' weights are already a full fine-tune)")
    else:
        from huggingface_hub import hf_hub_download
        model_id = f"robotics-diffusion-transformer/rdt-{args.model}"
        weight_path = hf_hub_download(model_id, "pytorch_model.bin")
        print(f"loading HF pretrained weights from {weight_path}...")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        runner.load_state_dict(state_dict, strict=True)
        del state_dict

        #Optional LoRA on top (1B only). 170M has no LoRA checkpoint in this project.
        if args.model == "1b":
            lora_path = "checkpoints/rdt_maniskill_lora/final.pt"
            if os.path.exists(lora_path):
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=16, lora_alpha=32,
                    target_modules=["attn.qkv", "attn.proj",
                                    "cross_attn.q", "cross_attn.kv", "cross_attn.proj"],
                    lora_dropout=0.0, bias="none")
                runner.model = get_peft_model(runner.model, lora_config)
                lora_ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
                runner.model.load_state_dict(lora_ckpt["lora_state_dict"], strict=False)
                print(f"loaded LoRA checkpoint (step {lora_ckpt.get('step', '?')})")
            else:
                print("no LoRA checkpoint — using pretrained 1B")
        else:
            print(f"skipping LoRA on {args.model} (no 170M ManiSkill LoRA available)")

    runner = runner.to("cuda", dtype=torch.bfloat16)
    runner.train(False)
    for p in runner.parameters():
        p.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM with RDT: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    #Gradient checkpointing — applied ONCE here. Must not be re-applied inside
    #compute_ig_for_step or each call would double-wrap block.forward.
    if args.no_checkpoint:
        print(f"gradient checkpointing DISABLED ({len(runner.model.blocks)} blocks)")
    else:
        from torch.utils.checkpoint import checkpoint as ckpt_fn
        def _make_wrapped(orig_fn):
            def wrapped(x, c, mask=None):
                return ckpt_fn(orig_fn, x, c, mask, use_reentrant=False)
            return wrapped
        for block in runner.model.blocks:
            block.forward = _make_wrapped(block.forward)
        print(f"gradient checkpointing enabled on {len(runner.model.blocks)} blocks")

    #----- Language embeddings (per task) -----
    lang_path = f"data/lang_embeds/{args.task}.pt"
    if not os.path.exists(lang_path):
        print(f"ERROR: no language embeddings at {lang_path}. Run encode_task_lang.py first.")
        sys.exit(1)
    lang_data = torch.load(lang_path, weights_only=True)
    lang_tokens = lang_data["embeds"].to("cuda", dtype=torch.bfloat16)
    lang_attn_mask = lang_data["attn_mask"].to("cuda", dtype=torch.bool)
    print(f"loaded language embeddings: {lang_tokens.shape}")

    baseline_path = "data/lang_embeds/baseline_bos_eos.pt"
    lang_baseline_data = torch.load(baseline_path, weights_only=True)
    lang_tokens_baseline = lang_baseline_data["embeds"].to("cuda", dtype=torch.bfloat16)

    #----- Constants passed to compute_ig_for_step every call -----
    state_elem_mask = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
    state_elem_mask[0, MANISKILL_INDICES] = 1.0
    action_mask = state_elem_mask.unsqueeze(1)                                       # (1, 1, 128)
    ctrl_freqs = torch.tensor([CONTROL_FREQ], device="cuda", dtype=torch.bfloat16)  # (1,)

    #----- Env -----
    import gymnasium as gym
    import mani_skill.envs
    #control_mode must match what RDT's LoRA was fine-tuned on. eval_rdt_maniskill.py:58
    #in the RDT repo explicitly sets pd_joint_pos. ManiSkill's default for PickCube-v1 is
    #pd_joint_delta_pos with action space [-1,1]; ACTION_MIN/ACTION_MAX in this file are
    #for pd_joint_pos (absolute joint positions, ~[-2.9, 2.9]). Without this control_mode
    #override the env silently clips + reinterprets each RDT action as a tiny delta,
    #which drives 1B+LoRA from 76% success down to 0%.
    #
    #max_episode_steps must also be set: ManiSkill wraps PickCube-v1 in a TimeLimitWrapper
    #that truncates at 50 env steps by default (one chunk of policy actions barely fits).
    #RDT's official eval (eval_rdt_maniskill.py) uses MAX_EPISODE_STEPS=400. Without this,
    #the policy is truncated before it can grasp; this is the second of two bugs that
    #stopped 1B+LoRA from succeeding.
    env = gym.make(args.task, obs_mode="state_dict", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   max_episode_steps=400)

    if args.video_dir:
        from mani_skill.utils.wrappers import RecordEpisode
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=args.video_dir, save_trajectory=False,
                            save_video=True, video_fps=30, info_on_video=False,
                            trajectory_name=f"{args.task}_{args.model}_ep")
        print(f"=== recording videos to {args.video_dir} ===")

    def render_pil():
        arr = env.render().squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return Image.fromarray(arr).resize((384, 384))

    #----- Episode loop -----
    print(f"\n=== running {args.episodes} episodes, m={args.m}, model={args.model} ===")
    print(f"=== output: {args.out} ===\n")
    jsonl = open(args.out, "a")

    for ep in range(args.episodes):
        if ep in resume_episodes:
            print(f"--- episode {ep}: SKIPPED (resume) ---")
            continue
        print(f"\n--- episode {ep} (seed={args.seed_base + ep}) ---")
        obs, info = env.reset(seed=args.seed_base + ep)
        frame_deque = deque(maxlen=2)
        frame_deque.append(None)               # no t-1 history at the very first call
        frame_deque.append(render_pil())

        call_idx = 0
        env_step_count = 0
        terminated = truncated = False

        while not (terminated or truncated) and call_idx < args.max_policy_calls:
            obs_image = frame_deque[-1]
            proprio = obs["agent"]["qpos"][0, :8].cpu()

            t0 = time.time()
            attr = compute_ig_for_step(
                runner, vision_model, obs_image, proprio,
                lang_tokens, lang_attn_mask, lang_tokens_baseline,
                bg_image_encoded, img_tokens_baseline,
                action_mask, ctrl_freqs,
                seed=args.seed_base + ep, m=args.m, target=args.target,
            )
            wall = time.time() - t0

            #Sidecar: full attribution tensors + reference action + obs image
            #+ raw proprio (needed by faithfulness.py to re-run perturbed
            #forwards without replaying the env from seed).
            #Sidecar name carries seed + target so different seeds / IG targets
            #(Month 4 alternative-target ablation) do not overwrite each other.
            #Default target "logpi" keeps the historical naming for that target.
            #A non-default m is tagged the same way so an m=128 pass cannot
            #overwrite the m=64 sidecars the sanity variants read as originals.
            #Default m=64 keeps the historical naming.
            _seed_tag = f"_s{args.seed_base}"
            _tgt_tag = "" if args.target == "logpi" else f"_{args.target}"
            _m_tag = "" if args.m == 64 else f"_m{args.m}"
            attr_path = (f"data/per_step_attr/{args.task}_{args.model}"
                         f"{_seed_tag}{_tgt_tag}{_m_tag}{_t_tag}_ep{ep:03d}_t{call_idx:02d}.pt")
            torch.save({
                "vision_attr": attr["vision"]["attribution"].cpu(),
                "lang_attr":   attr["language"]["attribution"].cpu(),
                "state_attr":  attr["state"]["attribution"].cpu(),
                "ref_action":  attr["ref_action"].cpu(),
                "obs_image":   np.array(obs_image),
                "proprio":     proprio.cpu(),
            }, attr_path)

            #JSONL summary row — summary stats only, safe for jq/grep.
            #State per-joint (8 floats) stays inline; large attribution tensors
            #are in the sidecar above.
            row = {
                "event": "step",
                "task": args.task, "model": args.model,
                "episode": ep, "seed": args.seed_base + ep,
                "policy_call_idx": call_idx, "env_step_at_call": env_step_count,
                "solver_steps": (args.solver_steps if args.solver_steps is not None
                                 else config["model"]["noise_scheduler"]["num_inference_timesteps"]),
                "ref_norm_maniskill": attr["ref_norm_maniskill"],
                "vision_err":   attr["vision"]["completeness_err"],
                "vision_gap":   attr["vision"]["expected_gap"],
                "vision_ig_sum": attr["vision"]["ig_sum"],
                "lang_err":     attr["language"]["completeness_err"],
                "lang_gap":     attr["language"]["expected_gap"],
                "lang_ig_sum":  attr["language"]["ig_sum"],
                "state_err":    attr["state"]["completeness_err"],
                "state_gap":    attr["state"]["expected_gap"],
                "state_ig_sum": attr["state"]["ig_sum"],
                "state_per_joint": attr["state"]["per_joint"].tolist(),
                "wall_seconds": wall,
                "attr_file": attr_path,
            }
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            print(f"  call {call_idx}: "
                  f"v={attr['vision']['completeness_err']:.4f} "
                  f"l={attr['language']['completeness_err']:.4f} "
                  f"s={attr['state']['completeness_err']:.4f} "
                  f"|ref|={attr['ref_norm_maniskill']:.2f} "
                  f"wall={wall:.1f}s")
            call_idx += 1

            #Apply the chunk — same 4x subsampling pattern as eval_maniskill.py:87.
            #ref_action is the Gaussian's mean μ = conditional_sample(real_inputs),
            #which IS the action the policy would execute. No duplicate forward.
            #
            #RDT outputs a (1, 64, 128) unified action; ManiSkill's Panda controller
            #wants a (1, 8) denormalized joint-space action per env.step. Replicate
            #the RoboticDiffusionTransformerModel._unformat_action_to_joint path from
            #~/rdt-repo/scripts/maniskill_model.py: slice MANISKILL_INDICES then
            #denormalize from [-1, 1] to [ACTION_MIN, ACTION_MAX] per-dim.
            action_unified = attr["ref_action"].squeeze(0).float().cpu()  # (64, 128)
            action_joints = action_unified[:, MANISKILL_INDICES]           # (64, 8), still in [-1, 1]
            action_denorm = (action_joints + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
            actions_np = action_denorm[::4, :].numpy()                     # (16, 8)
            for action in actions_np:
                obs, _r, terminated, truncated, info = env.step(action.reshape(1, 8))
                env_step_count += 1
                frame_deque.append(render_pil())
                if terminated or truncated:
                    break

        #Episode-end row — each episode has one outcome record keyed on event
        success = bool(info.get("success", False))
        jsonl.write(json.dumps({
            "task": args.task, "model": args.model, "episode": ep,
            "seed": args.seed_base + ep, "event": "episode_end",
            "env_steps": env_step_count, "policy_calls": call_idx,
            "success": success,
            "terminated": bool(terminated), "truncated": bool(truncated),
        }) + "\n")
        jsonl.flush()
        print(f"  episode {ep}: {call_idx} calls, {env_step_count} env steps, success={success}")

    jsonl.close()
    env.close()
    print(f"\n=== done. wrote {args.out} ===")


if __name__ == "__main__":
    main()
