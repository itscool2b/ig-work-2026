#evaluate fine-tuned RDT-1B on ManiSkill3 PickCube-v1
#loads LoRA checkpoint and runs episodes to measure success rate

import os
import sys
import torch
import numpy as np
import yaml
from PIL import Image
from collections import deque

sys.path.insert(0, os.path.expanduser("~/rdt-repo"))
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel

import gymnasium as gym
import mani_skill.envs

#config
TASK = "PickCube-v1"
NUM_EPISODES = 50
MAX_STEPS = 400
CHECKPOINT = "checkpoints/rdt_maniskill_lora/final.pt"

task2lang = {
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
}

#load model (this loads T5-XXL — needs ~10GB VRAM, may need CPU offload)
config_path = os.path.expanduser("~/rdt-repo/configs/base.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

print("loading model...")
policy = create_model(
    args=config,
    dtype=torch.bfloat16,
    pretrained=None,  # we'll load LoRA weights separately
    pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl",
    pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
)

#load LoRA checkpoint
if os.path.exists(CHECKPOINT):
    print(f"loading LoRA checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    lora_state = ckpt["lora_state_dict"]
    # Merge LoRA weights into policy
    policy_state = policy.policy.model.state_dict()
    policy_state.update(lora_state)
    policy.policy.model.load_state_dict(policy_state, strict=False)
    print(f"  loaded step {ckpt.get('step', '?')}")
else:
    print(f"WARNING: no checkpoint at {CHECKPOINT}, using pretrained weights")

#encode language
if os.path.exists(f"data/lang_embeds/{TASK}.pt"):
    lang_data = torch.load(f"data/lang_embeds/{TASK}.pt", weights_only=True)
    text_embed = lang_data["embeds"].to("cuda", dtype=torch.bfloat16)
else:
    text_embed = policy.encode_instruction(task2lang[TASK])

#create environment
#control_mode must match RDT's training + eval_rdt_maniskill.py:58 (pd_joint_pos).
#max_episode_steps=400 matches RDT's eval; the env's 50-step default truncates before
#the policy can finish.
#obs_mode must be "state_dict": the loop below reads obs["agent"]["qpos"], and
#plain "state" returns a flattened tensor (same convention as per_step_ig.py).
env = gym.make(TASK, obs_mode="state_dict", control_mode="pd_joint_pos",
               render_mode="rgb_array", max_episode_steps=400)

#run evaluation
success_count = 0
for episode in range(NUM_EPISODES):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed=episode + 20241201)
    policy.reset()

    img = env.render().squeeze(0).detach().cpu().numpy()
    obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs["agent"]["qpos"][:, :-1]

    done = False
    for step in range(MAX_STEPS):
        image_arrs = []
        for window_img in obs_window:
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]

        actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
        actions = actions[::4, :]  # subsample to match control frequency

        for idx in range(actions.shape[0]):
            obs, reward, terminated, truncated, info = env.step(actions[idx])
            img = env.render().squeeze(0).detach().cpu().numpy()
            obs_window.append(img)
            proprio = obs["agent"]["qpos"][:, :-1]
            if terminated or truncated:
                if info.get("success", False):
                    success_count += 1
                done = True
                break
        if done:
            break

    print(f"episode {episode+1}/{NUM_EPISODES}  success={info.get('success', False)}")

env.close()
print(f"\nSuccess rate: {success_count}/{NUM_EPISODES} = {success_count/NUM_EPISODES*100:.1f}%")
