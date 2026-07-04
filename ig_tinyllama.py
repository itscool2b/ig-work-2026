#imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from integrated_gradients import integrated_gradients

#model
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32,
    device_map="auto",
)
for p in model.parameters():
    p.requires_grad_(False)

#tokenize
prompt = "The capital of France is"
tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = tokens.input_ids

#embeddings — attribute to embedding layer output
embed_layer = model.get_input_embeddings()
with torch.no_grad():
    input_embeds = embed_layer(input_ids).float()

#target token
with torch.no_grad():
    out = model(input_ids=input_ids)
    next_token_id = out.logits[0, -1, :].argmax().item()
    target_name = tokenizer.decode(next_token_id)
    prob = torch.softmax(out.logits[0, -1, :], dim=-1)[next_token_id].item() * 100
    print(f"prompt: '{prompt}'")
    print(f"next token: '{target_name}' (id={next_token_id})")

#forward_fn — log_softmax of the next token
def forward_fn(embeds):
    out = model(inputs_embeds=embeds)
    logprobs = torch.log_softmax(out.logits[0, -1, :], dim=-1)
    return logprobs[next_token_id]

#ig — PAD token baseline (zeros cause RMSNorm gradient singularity), m=1000
with torch.no_grad():
    pad_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    baseline = embed_layer(pad_ids).float()
attr = integrated_gradients(forward_fn, input_embeds, baseline, m=1000)

#per-token attribution — sum over embedding dim
token_attr = attr.squeeze(0).sum(dim=-1).detach().cpu().numpy()
token_labels = [tokenizer.decode(t) for t in input_ids[0]]

#visualize
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(8, max(3, len(token_labels) * 0.5)), dpi=150)
fig.suptitle(f"TinyLlama-1.1B-Chat  |  '{prompt}' -> '{target_name}'  ({prob:.1f}%)",
             fontsize=12, fontweight="bold", y=1.03)

colors = ['#d32f2f' if v > 0 else '#1976d2' for v in token_attr]
ax.barh(range(len(token_labels)), token_attr, color=colors)
ax.set_yticks(range(len(token_labels)))
ax.set_yticklabels(token_labels, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("attribution (sum over embedding dim)", fontsize=10)
ax.axvline(x=0, color="0.3", linewidth=0.8, linestyle="-")

legend_elements = [Patch(facecolor='#d32f2f', label='positive attribution'),
                   Patch(facecolor='#1976d2', label='negative attribution')]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

fig.text(0.5, -0.04,
         "IG params: m=1000 steps  |  baseline: PAD-token embeddings  |  target: log_softmax(next token)",
         ha="center", fontsize=9, fontstyle="italic", color="0.4")

plt.tight_layout()
plt.savefig("output/ig_tinyllama.png", dpi=150, bbox_inches="tight")
print("saved ig_tinyllama.png")

#cleanup
del model
torch.cuda.empty_cache()
