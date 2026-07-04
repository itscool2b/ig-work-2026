#fine-tune RDT-1B with LoRA on ManiSkill3 PickCube demonstrations
#keeps SigLIP loaded for on-the-fly image encoding (avoids 148GB RAM for pre-encoding)
#RDT repo stays pristine — imported via sys.path

import os
import sys
import gc
import h5py
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.expanduser("~/rdt-repo"))
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from configs.state_vec import STATE_VEC_IDX_MAPPING

#constants
MANISKILL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
NUM_PATCHES = 729
CONTROL_FREQ = 25
CHUNK_SIZE = 64
DEVICE = "cuda"
DTYPE = torch.bfloat16

#hyperparameters
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1e-4
MAX_STEPS = 5000
SAVE_EVERY = 1000
CHECKPOINT_DIR = "checkpoints/rdt_maniskill_lora"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#load config
config_path = os.path.expanduser("~/rdt-repo/configs/base.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

#load SigLIP (stays loaded for on-the-fly encoding, frozen)
print("loading SigLIP...")
vision_model = SiglipVisionTower(vision_tower="google/siglip-so400m-patch14-384", args=None)
vision_model = vision_model.to(DEVICE, dtype=DTYPE)
image_processor = vision_model.image_processor
background_color = tuple(int(x * 255) for x in image_processor.image_mean)
bg_image = Image.new("RGB", (384, 384), background_color)
bg_processed = image_processor.preprocess(bg_image, return_tensors="pt")["pixel_values"][0]

#load pre-encoded language embeddings
lang_data = torch.load("data/lang_embeds/PickCube-v1.pt", weights_only=True)
lang_tokens = lang_data["embeds"].to(DEVICE, dtype=DTYPE)
lang_attn_mask = lang_data["attn_mask"].to(DEVICE, dtype=torch.bool)

#index demo trajectories (don't load images into memory)
demo_file = h5py.File("data/maniskill_demos/PickCube-v1.h5", "r")
traj_keys = [k for k in sorted(demo_file.keys(), key=lambda x: int(x.split("_")[1]))
             if demo_file[k].attrs.get("success", False) and demo_file[k].attrs.get("length", 0) >= CHUNK_SIZE + 1]
print(f"found {len(traj_keys)} valid trajectories")

#load RDT-1B backbone
print("loading RDT-1B...")
img_cond_len = config["common"]["img_history_size"] * config["common"]["num_cameras"] * NUM_PATCHES
runner = RDTRunner(
    action_dim=config["common"]["state_dim"],
    pred_horizon=config["common"]["action_chunk_size"],
    config=config["model"],
    lang_token_dim=config["model"]["lang_token_dim"],
    img_token_dim=config["model"]["img_token_dim"],
    state_token_dim=config["model"]["state_token_dim"],
    max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
    img_cond_len=img_cond_len,
    img_pos_embed_config=[("image", (config["common"]["img_history_size"],
                                     config["common"]["num_cameras"], -NUM_PATCHES))],
    lang_pos_embed_config=[("lang", -config["dataset"]["tokenizer_max_length"])],
    dtype=DTYPE,
)

from huggingface_hub import hf_hub_download
weight_path = hf_hub_download("robotics-diffusion-transformer/rdt-1b", "pytorch_model.bin")
state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
runner.load_state_dict(state_dict, strict=True)
del state_dict
runner = runner.to(DEVICE, dtype=DTYPE)
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

#apply LoRA to DiT attention layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["attn.qkv", "attn.proj", "cross_attn.q", "cross_attn.kv", "cross_attn.proj"],
    lora_dropout=0.05,
    bias="none",
)
runner.model = get_peft_model(runner.model, lora_config)
runner.model.print_trainable_parameters()

#optimizer
trainable_params = [p for p in runner.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

#helper: encode one frame on-the-fly through SigLIP
def encode_frame(frame_data):
    pil_img = Image.fromarray(frame_data)
    processed = image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]
    cam_stack = torch.stack([bg_processed, bg_processed, bg_processed,
                             processed, bg_processed, bg_processed])
    with torch.no_grad():
        img_embed = vision_model(cam_stack.to(DEVICE, dtype=DTYPE)).detach()
    return img_embed.reshape(1, -1, vision_model.hidden_size)

#training loop
print(f"\ntraining: {MAX_STEPS} steps, effective_batch={BATCH_SIZE*GRAD_ACCUM}")
runner.train()
# freeze everything except LoRA
for name, param in runner.named_parameters():
    if "lora" not in name:
        param.requires_grad_(False)

action_mask = torch.zeros(1, 1, 128, device=DEVICE, dtype=DTYPE)
action_mask[0, 0, MANISKILL_INDICES] = 1.0
ctrl_freqs = torch.tensor([CONTROL_FREQ], device=DEVICE, dtype=DTYPE)

running_loss = 0.0
optimizer.zero_grad()

for step in range(MAX_STEPS):
    #sample random episode and chunk
    traj_key = traj_keys[np.random.randint(len(traj_keys))]
    traj = demo_file[traj_key]
    n_steps = traj.attrs["length"]
    start = np.random.randint(0, n_steps - CHUNK_SIZE)

    #encode image on-the-fly (no grad through SigLIP)
    frame = traj["images"][f"frame_{start}"][:]
    img_tokens = encode_frame(frame)

    #get state and action chunk
    state = torch.tensor(traj["states"][start:start+1], dtype=DTYPE, device=DEVICE).unsqueeze(0)
    action_gt = torch.tensor(traj["actions"][start:start+CHUNK_SIZE], dtype=DTYPE, device=DEVICE).unsqueeze(0)

    #compute loss
    loss = runner.compute_loss(
        lang_tokens=lang_tokens,
        lang_attn_mask=lang_attn_mask,
        img_tokens=img_tokens,
        state_tokens=state,
        action_gt=action_gt,
        action_mask=action_mask,
        ctrl_freqs=ctrl_freqs,
    )
    loss = loss / GRAD_ACCUM
    loss.backward()
    running_loss += loss.item()

    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if (step + 1) % 100 == 0:
        avg_loss = running_loss / 100
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"step {step+1}/{MAX_STEPS}  loss={avg_loss:.4f}  vram={vram:.1f}GB")
        running_loss = 0.0

    if (step + 1) % SAVE_EVERY == 0:
        save_path = os.path.join(CHECKPOINT_DIR, f"step_{step+1}.pt")
        lora_state = {k: v.cpu() for k, v in runner.model.state_dict().items() if "lora" in k}
        torch.save({"lora_state_dict": lora_state, "step": step + 1}, save_path)
        print(f"  saved: {save_path}")

    #periodic cache cleanup
    if (step + 1) % 50 == 0:
        torch.cuda.empty_cache()

#save final
final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
lora_state = {k: v.cpu() for k, v in runner.model.state_dict().items() if "lora" in k}
torch.save({"lora_state_dict": lora_state, "step": MAX_STEPS}, final_path)
print(f"\ndone — saved {final_path}")

demo_file.close()
del runner, optimizer, vision_model
torch.cuda.empty_cache()
