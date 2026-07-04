# ig_llava.py

Integrated Gradients on LLaVA-1.5-7B (multimodal vision-language model). 4-bit quantized to fit 12 GB VRAM. Attributes next-token prediction to combined image+text embeddings. Outputs a side-by-side image heatmap and token bar chart.

This is the most complex script. It combines the vision-model approach (image heatmap) with the LLM approach (token bars) in a single attribution.

## Architecture

LLaVA-1.5-7B combines a CLIP ViT-L/14 vision encoder with a Vicuna-7B language model. The image is encoded into 576 patch tokens (24x24 grid, from 336px / 14px patches), projected through a multi-modal projector to 4096-dim (matching the LM), and spliced into the text token sequence at `<image>` placeholder positions. Total sequence: ~592 tokens.

## Quantization and memory

4-bit quantized via BitsAndBytes. The model uses ~4.4 GB in 4-bit.

8-bit was attempted but OOMs during the backward pass on 12 GB VRAM (model is ~7.4 GB in 8-bit, leaving insufficient room for gradient storage).

Gradient checkpointing trades compute for memory by recomputing activations during backward instead of storing them. Essential for fitting forward+backward in VRAM.

## Sections

### #embeddings -- constructing the combined input

This is the trickiest part. The processor creates input_ids with 576 image placeholder tokens (id=32000). We need to replace those with actual CLIP features.

```python
clip_out = model.get_image_features(inputs.pixel_values)
selected = clip_out.hidden_states[model.config.vision_feature_layer]
selected = selected[:, 1:]  # remove CLS token
img_feats = model.model.multi_modal_projector(selected).squeeze(0).float()
```

The pipeline: CLIP ViT hidden states (second-to-last layer) -> remove CLS token -> multi-modal projector -> (576, 4096).

The combined embedding is built by replacing the image placeholder positions with the projected features:
```python
combined = text_embeds.clone()
combined[0, img_start:img_end, :] = img_feats
```

### #baseline

PAD token embeddings for the full sequence (both image and text positions). Same rationale as TinyLlama: zeros cause RMSNorm gradient singularity. PAD is numerically stable and semantically represents "no information."

### #forward_fn

```python
def forward_fn(embeds):
    out = model(inputs_embeds=embeds.half(), attention_mask=attn_mask)
    logprobs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return logprobs[next_token_id]
```

Embeddings cast to float16 for the forward pass (matching 4-bit compute dtype). Log_softmax computed in float32 for precision.

### #split attributions

After IG on the combined 592-token sequence, attributions are split into image and text regions by index:

```python
image_attr = attr_squeezed[img_start:img_end].sum(dim=-1)  # (576,)
text_attr = concat(attr[:img_start], attr[img_end:]).sum(dim=-1)
```

### #visualize

Left: image heatmap (24x24 grid upscaled 14x with scipy.ndimage.zoom, overlaid at 50% opacity).
Right: text token bar chart (red=positive, blue=negative).

## Output

`output/ig_llava.png`

For "What is this?" on a German Shepherd image, the model predicts "The" as the first response token (70.0% probability).

## Completeness

28.26% error at m=300. This is the worst completeness across all five models.

The cause is 4-bit quantization. Each forward pass produces slightly different outputs for the same input due to stochastic rounding in the dequantization step. When IG accumulates gradients over m steps, this noise accumulates instead of averaging out. Higher m makes it worse:

| m | error |
|---|-------|
| 300 | 28.26% |
| 1000 | 78.89% |

This is a hardware limitation, not a code bug. On a GPU with 24+ GB VRAM, running in float16 or 8-bit would give much better completeness.

## Bug fix: get_image_features

The original code called:
```python
img_feats = model.get_image_features(inputs.pixel_values, return_dict=True).pooler_output[0]
```

This was wrong in two ways:

1. `get_image_features` in this transformers version returns the raw CLIP `BaseModelOutputWithPooling`, not projected features. The `return_dict=True` parameter is not valid for this method.

2. `.pooler_output` is the CLS token output (shape [batch, 1024]), not the 576-patch feature sequence. Indexing with `[0]` gave a 1024-dim vector that got broadcast-assigned to 576 embedding positions, producing nonsensical image representations.

The fix: manually extract the correct hidden layer from CLIP, remove the CLS token, and project through `model.model.multi_modal_projector` to get the proper (576, 4096) features.

## VRAM budget

- 4-bit model: ~4.4 GB
- Gradient checkpointed forward+backward: ~3 GB
- Embeddings + gradient accumulation: ~0.5 GB
- Total: ~8 GB (fits in 12 GB with room for desktop apps)
