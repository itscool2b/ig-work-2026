"""
Qualitative artifact renderers for per-step IG sidecars.

Pure functions: load a sidecar .pt, extract the per-modality attributions,
render to three file layouts:

    out/overlays/<task>/ep{EP}_t{T}.png     render_overlay_only_png
    out/tokens/<task>/ep{EP}_t{T}.png       render_tokens_only_figure
    out/episodes/<task>/ep{EP}_summary.png  render_episode_summary

render_step_figure produces the combined three-panel layout used by the
single-shot demo `ig_rdt.py`.

No RDT or SigLIP import here, these run on any machine with matplotlib
(the 150 GB of sidecars never need to leave the pod; PNGs come home).
"""

import os
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import zoom

from per_step_attribution import MANISKILL_INDICES, JOINT_NAMES

NUM_PATCHES = 729
GRID_SIZE = 27  # sqrt(NUM_PATCHES)
EXT_CAM_SLOT = 3  # [t-1 ext, t-1 wrist_r, t-1 wrist_l, t ext, t wrist_r, t wrist_l]
POS_COLOR = "#d32f2f"  # red
NEG_COLOR = "#1976d2"  # blue


def extract_vision_heatmap(vision_attr):
    """(1, 4374, H) -> (GRID_SIZE, GRID_SIZE) normalized to [0, 1] from the
    external-camera slot. vision_attr can be bf16 or float, on any device."""
    per_pos = vision_attr.squeeze(0).sum(dim=-1).detach().cpu().float().numpy()
    ext = per_pos[EXT_CAM_SLOT * NUM_PATCHES:(EXT_CAM_SLOT + 1) * NUM_PATCHES]
    grid = np.abs(ext).reshape(GRID_SIZE, GRID_SIZE)
    return grid / (grid.max() + 1e-8)


def extract_language_per_token(lang_attr, lang_attn_mask=None, n_real=None):
    """
    (1, 1024, H) -> (n_real,) per-token attribution (sum over hidden dim).
    Pass either `lang_attn_mask` (1, 1024) bool tensor, or `n_real` int.
    """
    if n_real is None:
        assert lang_attn_mask is not None, "need lang_attn_mask or n_real"
        n_real = int(lang_attn_mask.sum().item())
    per_tok = lang_attr.squeeze(0)[:n_real].sum(dim=-1).detach().cpu().float().numpy()
    return per_tok


def extract_state_per_joint(state_attr):
    """(1, 1, 128) -> (8,) per-joint attribution at MANISKILL_INDICES."""
    flat = state_attr.squeeze(0).squeeze(0).detach().cpu().float().numpy()
    return flat[MANISKILL_INDICES]


def render_vision_panel(ax, obs_image, heatmap_grid, title="vision patch attribution"):
    """
    Render the vision heatmap overlaid on the observation image on a given
    matplotlib axis. `heatmap_grid` is the GRID_SIZE x GRID_SIZE normalized
    heatmap returned by extract_vision_heatmap.
    """
    target_size = 384
    zoom_factor = target_size / heatmap_grid.shape[0]
    heatmap_up = zoom(heatmap_grid, (zoom_factor, zoom_factor), order=1)
    #Fallback: if scipy produces a slightly off-size array, crop or pad.
    if heatmap_up.shape != (target_size, target_size):
        zoomed = np.zeros((target_size, target_size))
        h, w = min(target_size, heatmap_up.shape[0]), min(target_size, heatmap_up.shape[1])
        zoomed[:h, :w] = heatmap_up[:h, :w]
        heatmap_up = zoomed
    ax.imshow(np.array(obs_image))
    im = ax.imshow(heatmap_up, cmap="hot", alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return im


def render_language_panel(ax, lang_attr_per_token, token_labels,
                          title="language token attribution"):
    n_lang = len(lang_attr_per_token)
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in lang_attr_per_token]
    ax.barh(range(n_lang), lang_attr_per_token, color=colors)
    ax.set_yticks(range(n_lang))
    if token_labels is not None and len(token_labels) >= n_lang:
        labels = list(token_labels[:n_lang])
    else:
        labels = [f"tok_{i}" for i in range(n_lang)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("attribution (sum over hidden dim)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axvline(x=0, color="0.3", linewidth=0.8)
    legend = [Patch(facecolor=POS_COLOR, label="positive"),
              Patch(facecolor=NEG_COLOR, label="negative")]
    ax.legend(handles=legend, loc="lower right", fontsize=9)


def render_state_panel(ax, state_attr_per_joint,
                       title="state attribution (per joint)"):
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in state_attr_per_joint]
    ax.barh(JOINT_NAMES, state_attr_per_joint, color=colors)
    ax.invert_yaxis()
    ax.axvline(x=0, color="0.3", linewidth=0.8)
    ax.set_xlabel("attribution", fontsize=10)
    ax.set_title(title, fontsize=11)
    legend = [Patch(facecolor=POS_COLOR, label="positive"),
              Patch(facecolor=NEG_COLOR, label="negative")]
    ax.legend(handles=legend, loc="lower right", fontsize=9)


def render_step_figure(sidecar, lang_attn_mask, token_labels, out_path,
                       title=None, subtitle=None, dpi=150):
    """
    Three-panel layout (vision | language | state) for one policy call,
    saved to out_path. Mirrors output/ig_rdt.png.

    Args:
        sidecar: dict as saved by per_step_ig.py (keys: vision_attr, lang_attr,
            state_attr, obs_image, ref_action, proprio).
        lang_attn_mask: (1, 1024) bool tensor or None. If None, inferred from
            len(token_labels).
        token_labels: list[str] of SentencePiece pieces, or None.
        out_path: str. Parent dirs created if missing.
    """
    heatmap = extract_vision_heatmap(sidecar["vision_attr"])
    if lang_attn_mask is not None:
        n_real = int(lang_attn_mask.sum().item()) if torch.is_tensor(lang_attn_mask) \
                 else int(np.asarray(lang_attn_mask).sum())
    else:
        n_real = len(token_labels) if token_labels is not None else sidecar["lang_attr"].shape[1]
    lang_per_tok = extract_language_per_token(sidecar["lang_attr"], n_real=n_real)
    state_per_joint = extract_state_per_joint(sidecar["state_attr"])
    obs_image = Image.fromarray(sidecar["obs_image"]) \
                if isinstance(sidecar["obs_image"], np.ndarray) else sidecar["obs_image"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 6), dpi=dpi,
        gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    im = render_vision_panel(ax1, obs_image, heatmap,
                             title="vision patch attribution (external camera)")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="normalized attribution")
    render_language_panel(ax2, lang_per_tok, token_labels)
    render_state_panel(ax3, state_per_joint)

    if subtitle:
        fig.text(0.5, -0.02, subtitle, ha="center", fontsize=9,
                 fontstyle="italic", color="0.4")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_overlay_only_png(sidecar, out_path, dpi=150):
    """
    Just the vision heatmap over the observation, no axes or titles.
    Output: H x W x 3 RGB, channel-summed IG, normalized, blended PNG.
    """
    heatmap = extract_vision_heatmap(sidecar["vision_attr"])
    obs_image = Image.fromarray(sidecar["obs_image"]) \
                if isinstance(sidecar["obs_image"], np.ndarray) else sidecar["obs_image"]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    render_vision_panel(ax, obs_image, heatmap, title="")
    ax.set_title("")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_tokens_only_figure(sidecar, token_labels, out_path, n_real=None, dpi=150):
    """Just the language bar chart. n_real defaults to len(token_labels)."""
    if n_real is None:
        n_real = len(token_labels) if token_labels is not None else sidecar["lang_attr"].shape[1]
    lang_per_tok = extract_language_per_token(sidecar["lang_attr"], n_real=n_real)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * n_real)), dpi=dpi)
    render_language_panel(ax, lang_per_tok, token_labels)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_episode_summary(sidecars_in_order: Iterable[dict], out_path,
                           n_frames=4, dpi=150):
    """
    Stitch key frames (default 4) from an episode into a single figure.
    Picks evenly-spaced indices from the full list of sidecars.
    """
    sidecars = list(sidecars_in_order)
    if len(sidecars) == 0:
        return
    if len(sidecars) <= n_frames:
        picks = list(range(len(sidecars)))
    else:
        picks = [int(round(i * (len(sidecars) - 1) / (n_frames - 1)))
                 for i in range(n_frames)]
    picks = sorted(set(picks))

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), dpi=dpi)
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, idx in enumerate(picks):
        sc = sidecars[idx]
        obs_image = Image.fromarray(sc["obs_image"]) \
                    if isinstance(sc["obs_image"], np.ndarray) else sc["obs_image"]
        heatmap = extract_vision_heatmap(sc["vision_attr"])
        axes[0, col].imshow(np.array(obs_image))
        axes[0, col].set_title(f"call {idx}", fontsize=10)
        axes[0, col].axis("off")
        render_vision_panel(axes[1, col], obs_image, heatmap,
                            title=f"attribution (call {idx})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
