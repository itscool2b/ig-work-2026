# ig_resnet.py

Integrated Gradients on ResNet50 (ImageNet classifier). Attributes the predicted class log-probability to individual input pixels. Outputs a 3-panel visualization.

## Sections

### #model

```python
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).cuda()
model.train(False)
for p in model.parameters():
    p.requires_grad_(False)
```

Loads pretrained ResNet50 with ImageNet V2 weights. `model.train(False)` disables dropout and locks batch norm to running stats. `requires_grad_(False)` on all params prevents gradient accumulation in the model. Only the input gets gradients.

### #preprocess

```python
img = Image.open('image.jpg')
preprocess = weights.transforms()
input_tensor = preprocess(img).unsqueeze(0).cuda()
```

Uses PIL `Image.open` (not `torchvision.io.decode_image`). The weights' transforms resize to 232px, center-crop to 224px, scale to [0,1] float, and normalize with ImageNet mean/std. `.unsqueeze(0)` adds the batch dimension.

### #predicted class

Runs the model in `no_grad` mode to find which class gets the highest logit. For `image.jpg` (German Shepherd photo), class_id=235, logit=6.24, 32.7% confidence.

### #forward_fn

```python
def forward_fn(x):
    logits = model(x).squeeze(0)
    return torch.log_softmax(logits, dim=-1)[class_id]
```

Returns the log-softmax of the predicted class. This is the log-probability, not the raw logit. log_softmax couples all 1000 class outputs through the softmax denominator, which makes the gradient landscape more complex than a raw logit target.

### #ig

```python
black_img = Image.new('RGB', (224, 224), (0, 0, 0))
baseline = preprocess(black_img).unsqueeze(0).cuda()
attr = integrated_gradients(forward_fn, input_tensor, baseline, m=300)
```

The baseline is a black image -- all-zero pixels in RGB space, then preprocessed through the same ImageNet normalization pipeline as the input. In normalized space this becomes approximately [-2.12, -2.04, -1.80] per channel, not zero. Using `torch.zeros_like(input_tensor)` would represent a gray image in pixel space, which is semantically wrong.

m=300 steps with Riemann sum approximation.

### #visualize

```python
attr_map = attr.squeeze(0).sum(dim=0).abs().detach().cpu().numpy()
attr_map = attr_map / attr_map.max()
```

1. `.squeeze(0)` removes batch dim: [3, 224, 224]
2. `.sum(dim=0)` sums over RGB channels: [224, 224]
3. `.abs()` takes absolute value (magnitude matters, not sign for pixel attribution)
4. Normalize to [0, 1] for visualization

Three panels:
- Left: Original image resized to 224x224
- Middle: Attribution heatmap (cmap="hot", black=low, red/yellow=high)
- Right: Overlay, heatmap at 50% opacity on the original

## Output

`output/ig_resnet50.png`

The heatmap highlights the dog's body, face, and ears. Background (grass) gets low attribution.

## Completeness

13.13% error with log_softmax target. This is expected for deep ReLU networks. The piecewise linear structure of ReLU creates thousands of gradient discontinuities along the interpolation path. Combined with log_softmax (which couples all 1000 classes through the softmax denominator), the Riemann sum converges erratically. Increasing m does not reliably improve the error -- convergence is non-monotonic due to aliasing between the uniform sample grid and the ReLU breakpoint locations. With raw logit target the error is 7.85%.

For comparison, ViT (which uses smooth GELU) gets 0.14% with the same m=300 and log_softmax target.

## Changes from original

- Switched image loading from `torchvision.io.decode_image` to `PIL.Image.open`
- Changed forward_fn from raw logit to log_softmax
- Baseline now created as a PIL black image passed through the same preprocess pipeline (not `normalize(torch.zeros(...))`)
- m changed from 300 (original) to various values during testing, settled back on 300
