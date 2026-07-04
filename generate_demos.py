#replay ManiSkill3 downloaded demos to get RGB observations + convert to RDT format
#ManiSkill demos are state-only — we replay them to capture images

import os
import sys
import h5py
import torch
import numpy as np
from PIL import Image
import gymnasium as gym
import mani_skill.envs

sys.path.insert(0, os.path.expanduser("~/rdt-repo"))
from configs.state_vec import STATE_VEC_IDX_MAPPING

#constants
MANISKILL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]

TASK = "PickCube-v1"
DEMO_PATH = os.path.expanduser(f"~/.maniskill/demos/{TASK}/motionplanning/trajectory.h5")
OUTPUT_DIR = "data/maniskill_demos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#normalization stats (from RDT repo's maniskill_model.py)
STATE_MIN = np.array([-0.7463, -0.0801, -0.4976, -2.6578, -0.5743, 1.8310, -2.2424, 0.0])
STATE_MAX = np.array([0.7645, 1.4967, 0.4651, -0.3867, 0.5506, 3.2901, 2.5738, 0.04])
ACTION_MIN = np.array([-0.7472, -0.0863, -0.4995, -2.6584, -0.5751, 1.8291, -2.2452, -1.0])
ACTION_MAX = np.array([0.7655, 1.4984, 0.4679, -0.3818, 0.5517, 3.2916, 2.5758, 1.0])

#load source demos
print(f"loading demos from {DEMO_PATH}...")
src = h5py.File(DEMO_PATH, "r")
traj_keys = sorted([k for k in src.keys() if k.startswith("traj_")],
                   key=lambda x: int(x.split("_")[1]))
print(f"found {len(traj_keys)} trajectories")

#create env for replay
env = gym.make(TASK, obs_mode="rgbd", num_envs=1, render_mode="rgb_array")

#replay trajectories and save with images
output_path = os.path.join(OUTPUT_DIR, f"{TASK}.h5")
out = h5py.File(output_path, "w")

max_trajs = min(200, len(traj_keys))  # start with 200 for speed
success_count = 0

for idx in range(max_trajs):
    traj_key = traj_keys[idx]
    traj = src[traj_key]
    actions = traj["actions"][:]  # (T, 8)
    env_states = traj["env_states"]

    #reset env to the initial state of this trajectory
    initial_state = {}
    for group_name in env_states.keys():
        initial_state[group_name] = {}
        for key in env_states[group_name].keys():
            initial_state[group_name][key] = torch.tensor(env_states[group_name][key][0:1])

    obs, _ = env.reset()
    env.unwrapped.set_state_dict(initial_state)
    obs = env.unwrapped.get_obs()

    #collect trajectory data
    images = []
    qpos_list = []
    action_list = []

    for step in range(len(actions)):
        #capture observation
        img = env.render().squeeze(0).detach().cpu().numpy()  # (512, 512, 3)
        images.append(img)
        qpos = obs["agent"]["qpos"][0, :8].cpu().numpy()  # 7 arm + 1 gripper
        qpos_list.append(qpos)

        #step environment
        action = torch.tensor(actions[step:step+1]).float()
        obs, reward, terminated, truncated, info = env.step(action)
        action_list.append(actions[step])

    #check success
    was_success = bool(traj["success"][-1])
    if was_success:
        success_count += 1

    #save to output HDF5
    grp = out.create_group(f"traj_{idx}")

    #images: resize to 384x384, save as uint8
    img_grp = grp.create_group("images")
    for t, img in enumerate(images):
        pil_img = Image.fromarray(img.astype(np.uint8)).resize((384, 384))
        img_grp.create_dataset(f"frame_{t}", data=np.array(pil_img))

    #states: normalize to [-1,1], zero-pad to 128D
    qpos_arr = np.array(qpos_list)  # (T, 8)
    qpos_norm = (qpos_arr - STATE_MIN) / (STATE_MAX - STATE_MIN) * 2 - 1
    states_128 = np.zeros((len(qpos_list), 128), dtype=np.float32)
    states_128[:, MANISKILL_INDICES] = qpos_norm.astype(np.float32)
    grp.create_dataset("states", data=states_128)

    #actions: normalize to [-1,1], zero-pad to 128D
    action_arr = np.array(action_list)  # (T, 8)
    action_norm = (action_arr - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) * 2 - 1
    actions_128 = np.zeros((len(action_list), 128), dtype=np.float32)
    actions_128[:, MANISKILL_INDICES] = action_norm.astype(np.float32)
    grp.create_dataset("actions", data=actions_128)

    #metadata
    grp.attrs["success"] = was_success
    grp.attrs["length"] = len(actions)

    if (idx + 1) % 20 == 0:
        print(f"  processed {idx+1}/{max_trajs} trajectories ({success_count} successful)")

out.close()
src.close()
env.close()

print(f"\ndone — saved {max_trajs} trajectories to {output_path}")
print(f"  successful: {success_count}/{max_trajs}")
print(f"  states shape: (T, 128), actions shape: (T, 128)")
