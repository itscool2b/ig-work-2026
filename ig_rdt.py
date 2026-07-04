#imports
import sys
import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/rdt-repo"))
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from configs.state_vec import STATE_VEC_IDX_MAPPING
from integrated_gradients import integrated_gradients

#constants — ManiSkill Panda arm mapping into RDT's 128-dim unified space
MANISKILL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
JOINT_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]
CONTROL_FREQ = 25
SEED = 42  # fixed RNG seed for the initial noise inside conditional_sample
ENABLE_GRADIENT_CHECKPOINTING = True  # vision IG OOMs at ~10 GB without this on a 12 GB card

#model — load SigLIP for image encoding
print("loading SigLIP...")
vision_model = SiglipVisionTower(vision_tower="google/siglip-so400m-patch14-384", args=None)
vision_model = vision_model.to("cuda", dtype=torch.bfloat16)
image_processor = vision_model.image_processor
NUM_PATCHES = vision_model.num_patches  # 729 (27x27)

#observation — render a real frame from ManiSkill PickCube-v1
import gymnasium as gym
import mani_skill.envs
env = gym.make("PickCube-v1", obs_mode="state_dict", num_envs=1, render_mode="rgb_array")
obs, _ = env.reset(seed=42)
#obs["agent"]["qpos"] is (1, 9) for Franka Panda (7 arm joints + 2 gripper fingers)
#[:8] keeps 7 arm joints + 1 gripper finger — matches eval_rdt_maniskill.py reference
proprio = obs["agent"]["qpos"][0, :8].cpu()

#env.render() returns (1, 512, 512, 3) uint8 tensor — same path used by
#generate_demos.py and eval_maniskill.py. Resize to 384 for SigLIP's patch-14 input.
img_array = env.render().squeeze(0).detach().cpu().numpy().astype(np.uint8)
obs_image = Image.fromarray(img_array).resize((384, 384))
env.close()
print(f"proprio from ManiSkill: shape={proprio.shape} values={proprio.float().numpy()}")
print(f"observation image: {obs_image.size} (rendered from PickCube-v1)")

#encode image through SigLIP
#RDT expects 6 image frames: [ext_t-1, wrist_r_t-1, wrist_l_t-1, ext_t, wrist_r_t, wrist_l_t]
#ManiSkill has 1 camera — fill missing with background
background_color = tuple(int(x * 255) for x in image_processor.image_mean)
bg_image = Image.new("RGB", (384, 384), background_color)

image_list = [bg_image, bg_image, bg_image,  # t-1: no history yet
              obs_image, bg_image, bg_image]  # t: only external camera

image_tensors = []
for img in image_list:
    processed = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
    image_tensors.append(processed)
image_tensor = torch.stack(image_tensors).to("cuda", dtype=torch.bfloat16)

with torch.no_grad():
    img_tokens = vision_model(image_tensor).detach()  # (6, 729, 1152)
    img_tokens = img_tokens.reshape(1, -1, vision_model.hidden_size)  # (1, 4374, 1152)
print(f"image tokens: {img_tokens.shape}")

#encode baseline images through SigLIP
#only slot 3 (external camera at time t) interpolates obs→gray. The five other slots
#hold bg_image in BOTH input and baseline, so IG does not integrate over bg→gray there.
#Completeness then reports attribution for the external-camera slot alone, matching
#what the heatmap visualizes in the plot.
gray_image = Image.new("RGB", (384, 384), (128, 128, 128))
baseline_list = [bg_image, bg_image, bg_image,   # t-1 slots: same bg as input
                 gray_image,                      # t ext-cam: only slot that interpolates
                 bg_image, bg_image]              # t wrist slots: same bg as input
gray_tensors = []
for img in baseline_list:
    processed = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
    gray_tensors.append(processed)
gray_tensor = torch.stack(gray_tensors).to("cuda", dtype=torch.bfloat16)

with torch.no_grad():
    img_tokens_baseline = vision_model(gray_tensor).detach()
    img_tokens_baseline = img_tokens_baseline.reshape(1, -1, vision_model.hidden_size)
print(f"baseline image tokens: {img_tokens_baseline.shape}")

#unload SigLIP — not needed during IG
del vision_model, image_tensor, gray_tensor, image_processor
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM after SigLIP unload: {torch.cuda.memory_allocated()/1e9:.2f} GB")

#load RDTRunner backbone
print("loading RDT-1B backbone...")
config_path = os.path.expanduser("~/rdt-repo/configs/base.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

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

#load pretrained weights from HuggingFace
from huggingface_hub import hf_hub_download
weight_path = hf_hub_download("robotics-diffusion-transformer/rdt-1b", "pytorch_model.bin")
print(f"loading weights from {weight_path}...")
state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
runner.load_state_dict(state_dict, strict=True)
del state_dict

#load fine-tuned LoRA weights if available
lora_path = "checkpoints/rdt_maniskill_lora/final.pt"
if os.path.exists(lora_path):
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=16, lora_alpha=32,
        target_modules=["attn.qkv", "attn.proj", "cross_attn.q", "cross_attn.kv", "cross_attn.proj"],
        lora_dropout=0.0, bias="none")
    runner.model = get_peft_model(runner.model, lora_config)
    ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
    runner.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    print(f"loaded LoRA checkpoint (step {ckpt.get('step', '?')})")
else:
    print("no LoRA checkpoint — using pretrained weights")

runner = runner.to("cuda", dtype=torch.bfloat16)
runner.train(False)
for p in runner.parameters():
    p.requires_grad_(False)
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM with RDT loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB")

#optional gradient checkpointing — wraps each RDTBlock.forward in torch.utils.checkpoint
#use_reentrant=False works with frozen params as long as one input carries grad,
#which is true for all three IG modalities (state_traj / lang_cond / img_cond)
if ENABLE_GRADIENT_CHECKPOINTING:
    from torch.utils.checkpoint import checkpoint as ckpt
    def _make_wrapped(orig_fn):
        def wrapped(x, c, mask=None):
            return ckpt(orig_fn, x, c, mask, use_reentrant=False)
        return wrapped
    for block in runner.model.blocks:
        block.forward = _make_wrapped(block.forward)
    print(f"gradient checkpointing enabled on {len(runner.model.blocks)} RDT blocks")

#language embeddings — load pre-encoded T5-XXL from disk
lang_path = "data/lang_embeds/PickCube-v1.pt"
if os.path.exists(lang_path):
    lang_data = torch.load(lang_path, weights_only=True)
    lang_tokens = lang_data["embeds"].to("cuda", dtype=torch.bfloat16)
    lang_attn_mask = lang_data["attn_mask"].to("cuda", dtype=torch.bool)
    lang_token_labels = lang_data.get("tokens", None)  # SentencePiece pieces, length == num real tokens
    print(f"loaded language embeddings: {lang_tokens.shape}")
else:
    #fallback — zeros if T5 encoding hasn't run yet
    print("WARNING: no pre-encoded language embeddings, using zeros")
    lang_tokens = torch.zeros(1, 20, 4096, device="cuda", dtype=torch.bfloat16)
    lang_attn_mask = torch.ones(1, 20, device="cuda", dtype=torch.bool)
    lang_token_labels = None

#language baseline — load the [PAD, EOS] minimal-sequence T5 encoding from disk
#(the file name says bos_eos but T5 has no BOS token; the encoded pair is PAD and EOS,
# more semantically meaningful than zeros, which sat the lang_adaptor at a degenerate point)
baseline_path = "data/lang_embeds/baseline_bos_eos.pt"
baseline_data = torch.load(baseline_path, weights_only=True)
lang_tokens_baseline = baseline_data["embeds"].to("cuda", dtype=torch.bfloat16)
lang_attn_mask_baseline = baseline_data["attn_mask"].to("cuda", dtype=torch.bool)
print(f"loaded language baseline: {lang_tokens_baseline.shape}")

#format proprioception into RDT's 128-dim unified state vector
state_min = torch.tensor([-0.7463, -0.0801, -0.4976, -2.6578, -0.5743, 1.8310, -2.2424, 0.0])
state_max = torch.tensor([0.7645, 1.4967, 0.4651, -0.3867, 0.5506, 3.2901, 2.5738, 0.04])
proprio_norm = (proprio - state_min) / (state_max - state_min) * 2 - 1  # normalize to [-1, 1]

state_vec = torch.zeros(1, 1, 128, device="cuda", dtype=torch.bfloat16)
state_vec[0, 0, MANISKILL_INDICES] = proprio_norm.to(device="cuda", dtype=torch.bfloat16)

state_elem_mask = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
state_elem_mask[0, MANISKILL_INDICES] = 1.0

#state IG inputs — pre-adaptor (raw 128-dim). state_adaptor is mlp3x_gelu (nonlinear),
#so IG interpolating the post-adaptor (1,1,2048) token would mix all 128 input dims
#across the 2048 output channels and lose per-joint structure. Interpolating pre-adaptor
#lets the chain rule recover per-joint attribution at MANISKILL_INDICES.
state_input_actual = state_vec                              # (1, 1, 128) — proprio at MANISKILL_INDICES, zeros elsewhere
state_input_baseline = torch.zeros_like(state_vec)          # (1, 1, 128)

ctrl_freqs = torch.tensor([CONTROL_FREQ], device="cuda", dtype=torch.bfloat16)
action_mask = state_elem_mask.unsqueeze(1)  # (1, 1, 128) — held fixed across IG interpolation

#adapt lang+img once (not interpolated for state IG). state_adaptor is intentionally
#NOT pre-applied for state IG — forward_fn_state runs it inside so gradients flow back
#to the raw 128-dim input. state_traj_actual is still built here for the reference action.
with torch.no_grad():
    lang_adapted = runner.lang_adaptor(lang_tokens)              # (1, lang_len, 2048)
    img_adapted = runner.img_adaptor(img_tokens)                 # (1, 4374, 2048)
    state_traj_actual = runner.state_adaptor(
        torch.cat([state_vec, action_mask], dim=2))              # (1, 1, 2048) — for ref action only

    lang_adapted_bl = runner.lang_adaptor(lang_tokens_baseline)
    img_adapted_bl = runner.img_adaptor(img_tokens_baseline)

print(f"adapted shapes: lang={lang_adapted.shape}, img={img_adapted.shape}, "
      f"state_ref={state_traj_actual.shape}, state_ig_input={state_input_actual.shape}")

#seeded wrapper around the full 5-step DPM-Solver++ chain
#bypasses predict_action (which is wrapped in no_grad in inference scripts) and
#calls conditional_sample directly. Re-seeds RNG immediately before each call so the
#initial noisy_action = torch.randn(...) inside conditional_sample is identical across calls.
def seeded_conditional_sample(lang_c, img_c, state_t):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return runner.conditional_sample(
        lang_c, lang_attn_mask, img_c, state_t, action_mask, ctrl_freqs)

#reference executed action — one no_grad call with the real inputs and the fixed seed
#all IG forward passes will compare their interpolated prediction against this trajectory
with torch.no_grad():
    ref_action = seeded_conditional_sample(lang_adapted, img_adapted, state_traj_actual)
ref_action = ref_action.detach()  # (1, 64, 128), already masked by conditional_sample
print(f"reference action: {ref_action.shape}, "
      f"||ref[MANISKILL]||={ref_action[..., MANISKILL_INDICES].float().norm().item():.4f}")

#IG target: y_t = log π(a_t | o_t) under a unit-variance Gaussian around the final
#denoised mean μ = conditional_sample(o_t). RDT has prediction_type="sample" (predict-x₀),
#so the final head output IS the mean of that Gaussian. With σ²=1 and the constant
#entropy term dropped (it does not affect IG gradients), log π reduces to -½‖a − μ‖².
#Keeping .mean() over MANISKILL_INDICES × horizon matches the prior normalization; all
#IG attributions scale by a constant factor vs the old -MSE target (completeness % invariant).
SIGMA_SQ = 1.0

def _log_pi(pred_action):
    diff = (pred_action[..., MANISKILL_INDICES]
            - ref_action[..., MANISKILL_INDICES]).float()
    return -0.5 / SIGMA_SQ * (diff ** 2).mean()

def forward_fn_vision(img_cond):
    return _log_pi(seeded_conditional_sample(lang_adapted, img_cond, state_traj_actual))

def forward_fn_language(lang_cond):
    return _log_pi(seeded_conditional_sample(lang_cond, img_adapted, state_traj_actual))

def forward_fn_state(state_vec_in):
    #state_adaptor runs INSIDE forward_fn_state: IG interpolates the 128-dim state_vec,
    #and autograd pulls gradients back through the mlp3x_gelu adaptor to the raw input
    state_with_mask = torch.cat([state_vec_in, action_mask], dim=2)  # (1, 1, 256)
    state_token = runner.state_adaptor(state_with_mask)              # (1, 1, 2048)
    return _log_pi(seeded_conditional_sample(lang_adapted, img_adapted, state_token))

#ig — vision attribution (m=300, largest input)
print("\n--- Vision IG ---")
vision_attr = integrated_gradients(forward_fn_vision, img_adapted, img_adapted_bl, m=300)

#ig — language attribution
print("\n--- Language IG ---")
lang_attr = integrated_gradients(forward_fn_language, lang_adapted, lang_adapted_bl, m=300)

#ig — state attribution (IG interpolates the raw 128-dim state_vec, not the post-adaptor token)
print("\n--- State IG ---")
state_attr = integrated_gradients(forward_fn_state, state_input_actual, state_input_baseline, m=300)

#per-modality attribution scores
#vision: per-patch attribution (sum over hidden dim)
vis_attr = vision_attr.squeeze(0).sum(dim=-1).detach().cpu().float().numpy()  # (4374,)
#only use external camera patches at time t (patches 3*729 to 4*729)
ext_cam_attr = vis_attr[3*NUM_PATCHES:4*NUM_PATCHES]  # (729,)
grid_size = int(np.sqrt(NUM_PATCHES))  # 27

#language: per-token attribution
n_lang = int(lang_attn_mask.sum().item())
lang_attr_per_token = lang_attr.squeeze(0)[:n_lang].sum(dim=-1).detach().cpu().float().numpy()

#state: per-joint attribution extracted at MANISKILL_INDICES (IG was on raw 128-dim input)
#dims outside MANISKILL_INDICES are identical (zero) at both input and baseline, so their
#attribution is zero by construction and sum(state_attr_per_joint) ≈ state_total.
state_attr_flat = state_attr.squeeze(0).squeeze(0).detach().cpu().float()  # (128,)
state_attr_per_joint = state_attr_flat[MANISKILL_INDICES].numpy()          # (8,)
state_total = float(state_attr_flat.sum())

#visualize — 3-panel figure
from matplotlib.patches import Patch

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=150,
                                     gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
fig.suptitle(f"RDT-1B  |  PickCube-v1  |  5-step DPM-Solver++ chain  |  "
             f"target: log π(a_t | o_t) under N(μ, σ²=1) over MANISKILL dims",
             fontsize=13, fontweight="bold", y=1.02)

#left: vision heatmap
heatmap = np.abs(ext_cam_attr).reshape(grid_size, grid_size)
heatmap = heatmap / (heatmap.max() + 1e-8)
from scipy.ndimage import zoom
heatmap_up = zoom(heatmap, 384 // grid_size, order=1)
ax1.imshow(np.array(obs_image))
im1 = ax1.imshow(heatmap_up, cmap="hot", alpha=0.5)
ax1.set_title("vision patch attribution (external camera)", fontsize=11)
ax1.axis("off")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="normalized attribution")

#center: language token bar chart
colors_lang = ["#d32f2f" if v > 0 else "#1976d2" for v in lang_attr_per_token]
ax2.barh(range(n_lang), lang_attr_per_token, color=colors_lang)
ax2.set_yticks(range(n_lang))
#per-token labels: SentencePiece pieces from the saved .pt, fallback to tok_i if missing
if lang_token_labels is not None and len(lang_token_labels) >= n_lang:
    token_labels = list(lang_token_labels[:n_lang])
else:
    token_labels = [f"tok_{i}" for i in range(n_lang)]
ax2.set_yticklabels(token_labels, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("attribution (sum over hidden dim)", fontsize=10)
ax2.set_title("language token attribution", fontsize=11)
ax2.axvline(x=0, color="0.3", linewidth=0.8)
legend_elements = [Patch(facecolor="#d32f2f", label="positive"),
                   Patch(facecolor="#1976d2", label="negative")]
ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)

#right: state attribution — real per-joint IG values, sign-colored
colors_state = ["#d32f2f" if v > 0 else "#1976d2" for v in state_attr_per_joint]
ax3.barh(JOINT_NAMES, state_attr_per_joint, color=colors_state)
ax3.invert_yaxis()  # joint_0 at top like the language chart
ax3.axvline(x=0, color="0.3", linewidth=0.8)
ax3.set_xlabel("attribution", fontsize=10)
ax3.set_title("state attribution (per joint)", fontsize=11)
ax3.legend(handles=legend_elements, loc="lower right", fontsize=9)

fig.text(0.5, -0.02,
         "IG params: m=300 steps  |  baseline: gray ext-cam / BOS-EOS lang / zeros state  |  "
         "target: log π(a_t | o_t) = −½‖a − μ‖²/σ² (σ²=1, Gaussian around final denoised mean)",
         ha="center", fontsize=9, fontstyle="italic", color="0.4")

plt.tight_layout()
plt.savefig("output/ig_rdt.png", dpi=150, bbox_inches="tight")
print("\nsaved ig_rdt.png")

#cleanup
del runner
torch.cuda.empty_cache()
