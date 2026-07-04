#imports
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from integrated_gradients import integrated_gradients

#model — 4-bit quantized to fit 12GB VRAM (8-bit OOMs during backward pass)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", quantization_config=bnb_config, device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
for p in model.parameters():
    p.requires_grad_(False)
model.gradient_checkpointing_enable()

#process image + text
image = Image.open("image.jpg")
prompt = "USER: <image>\nWhat is this?\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
input_ids = inputs.input_ids

#embeddings — construct combined vision+text manually
embed_layer = model.get_input_embeddings()
img_token_id = model.config.image_token_id

with torch.no_grad():
    text_embeds = embed_layer(input_ids).float()
    # get_image_features returns raw CLIP output — select layer, remove CLS, project
    clip_out = model.get_image_features(inputs.pixel_values)
    selected = clip_out.hidden_states[model.config.vision_feature_layer]
    if getattr(model.config, 'vision_feature_select_strategy', 'default') == "default":
        selected = selected[:, 1:]  # remove CLS token
    img_feats = model.model.multi_modal_projector(selected).squeeze(0).float()

n_image_tokens = img_feats.shape[0]
img_mask = (input_ids[0] == img_token_id)
img_start = img_mask.nonzero()[0].item()
img_end = img_start + n_image_tokens

combined = text_embeds.clone()
combined[0, img_start:img_end, :] = img_feats
input_embeds = combined.detach()

print(f"sequence: {input_embeds.shape[1]} tokens ({n_image_tokens} image + {input_embeds.shape[1] - n_image_tokens} text)")

#baseline — PAD embeddings (zeros cause RMSNorm gradient singularity)
with torch.no_grad():
    pad_ids = torch.full((1, input_embeds.shape[1]), processor.tokenizer.pad_token_id,
                         dtype=torch.long, device="cuda")
    baseline = embed_layer(pad_ids).float()

#target token
attn_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model(inputs_embeds=input_embeds.half(), attention_mask=attn_mask)
    next_token_id = out.logits[0, -1, :].argmax().item()
    target_name = processor.tokenizer.decode(next_token_id)
    prob = torch.softmax(out.logits[0, -1, :].float(), dim=-1)[next_token_id].item() * 100
    print(f"next token: '{target_name}' (id={next_token_id}, {prob:.1f}%)")
del out

#free everything not needed for IG
del text_embeds, img_feats, combined, inputs
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM before IG: {torch.cuda.memory_allocated()/1e9:.2f} GB")

#forward_fn — float16 forward, float32 attributions
def forward_fn(embeds):
    out = model(inputs_embeds=embeds.half(), attention_mask=attn_mask)
    logprobs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return logprobs[next_token_id]

#ig — m=300, float32
attr = integrated_gradients(forward_fn, input_embeds, baseline, m=300)

#split attributions
attr_squeezed = attr.squeeze(0).detach()
image_attr = attr_squeezed[img_start:img_end].sum(dim=-1).cpu().numpy()
text_before = attr_squeezed[:img_start].sum(dim=-1).cpu().numpy()
text_after = attr_squeezed[img_end:].sum(dim=-1).cpu().numpy()
text_attr = np.concatenate([text_before, text_after])

text_ids_before = input_ids[0, :img_start]
text_ids_after = input_ids[0, img_end:]
text_ids = torch.cat([text_ids_before, text_ids_after])
token_labels = [processor.tokenizer.decode(t) for t in text_ids]

#visualize — side-by-side image heatmap + token bars
from matplotlib.patches import Patch

grid_size = int(np.sqrt(n_image_tokens))
heatmap = np.abs(image_attr).reshape(grid_size, grid_size)
heatmap = heatmap / (heatmap.max() + 1e-8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150,
                                gridspec_kw={"width_ratios": [1.2, 1]})

question = prompt.split('\n')[1].strip()
fig.suptitle(f"LLaVA-1.5-7B (4-bit)  |  '{question}' -> '{target_name}'  ({prob:.1f}%)",
             fontsize=13, fontweight="bold", y=1.02)

#left: image heatmap overlay
img_resized = image.resize((grid_size * 14, grid_size * 14))
from scipy.ndimage import zoom
heatmap_upscaled = zoom(heatmap, 14, order=1)
ax1.imshow(img_resized)
im1 = ax1.imshow(heatmap_upscaled, cmap="hot", alpha=0.5)
ax1.set_title("image patch attribution (|sum over embed dim|)", fontsize=11)
ax1.axis("off")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="normalized attribution")

#right: token bar chart
colors = ["#d32f2f" if v > 0 else "#1976d2" for v in text_attr]
ax2.barh(range(len(token_labels)), text_attr, color=colors)
ax2.set_yticks(range(len(token_labels)))
ax2.set_yticklabels(token_labels, fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("attribution (sum over embed dim)", fontsize=10)
ax2.set_title(f"text token attribution -> '{target_name}'", fontsize=11)
ax2.axvline(x=0, color="0.3", linewidth=0.8, linestyle="-")

legend_elements = [Patch(facecolor='#d32f2f', label='positive attribution'),
                   Patch(facecolor='#1976d2', label='negative attribution')]
ax2.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

fig.text(0.5, -0.02,
         f"IG params: m=300 steps  |  baseline: PAD-token embeddings  |  target: log_softmax(next token)  |  {n_image_tokens} image patches",
         ha="center", fontsize=9, fontstyle="italic", color="0.4")

plt.tight_layout()
plt.savefig("output/ig_llava.png", dpi=150, bbox_inches="tight")
print("saved ig_llava.png")

#cleanup
del model
torch.cuda.empty_cache()
