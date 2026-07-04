# ig_vit.py

Integrated Gradients on ViT-B/16 (Vision Transformer). Attributes the predicted class log-probability to input pixels. Structurally identical to ig_resnet.py but produces patch-shaped attribution patterns and achieves dramatically better completeness.

## How ViT differs from ResNet for IG

ViT-B/16: 12 transformer layers, 12 attention heads, 768-dim embeddings. The input 224x224 image is split into 14x14 = 196 patches of 16x16 pixels each. Each patch is linearly projected to a 768-dim embedding, then processed as a token through the transformer.

Because IG operates on the raw pixel input (not the patch embeddings), the attribution heatmap shows a visible 14x14 grid pattern. Pixels within the same patch tend to get similar attribution values since they share the same linear projection.

## Why ViT gets 0.14% completeness vs ResNet's 13.13%

GELU is smooth (infinitely differentiable). ReLU is piecewise linear with sharp kinks. The IG integral approximation depends on the smoothness of the gradient along the interpolation path:

- ReLU: gradient is a step function that changes abruptly at every activation boundary. With 50+ layers, the path crosses thousands of boundaries. The Riemann sum can't capture each transition, and convergence is non-monotonic (increasing m sometimes makes it worse due to aliasing).
- GELU: gradient is smooth and continuous. The Riemann sum converges monotonically and quickly.

| m | ViT error | ResNet error |
|---|-----------|-------------|
| 64 | 15.08% | 44.01% |
| 128 | 1.08% | 48.09% |
| 300 | 0.14% | 13.13% |

ViT converges monotonically. ResNet oscillates.

## Sections

### #model

```python
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights).cuda()
model.train(False)
```

ImageNet V1 weights. `model.train(False)` sets inference mode (disables dropout, locks batch norm to running stats).

### #forward_fn

```python
def forward_fn(x):
    logits = model(x).squeeze(0)
    return torch.log_softmax(logits, dim=-1)[class_id]
```

Same log_softmax target as ResNet.

### #ig

Same black image baseline (zeros in pixel space, preprocessed through ImageNet normalization). m=300 steps.

### #visualize

Same 3-panel layout as ResNet (original, heatmap, overlay). The heatmap shows the characteristic 14x14 patch grid because ViT processes patches as tokens. Patches overlapping the dog are bright. Background patches are dark.

## Output

`output/ig_vit.png`

Predicts "German shepherd" at 89.7% confidence (logit=9.15). Much higher than ResNet's 32.7%.

## Changes from original

- Switched image loading from `torchvision.io.decode_image` to `PIL.Image.open`
- Changed forward_fn from raw logit to log_softmax
- Same baseline fix as ResNet (PIL black image through preprocess pipeline)
