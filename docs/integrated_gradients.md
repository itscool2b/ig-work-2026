# integrated_gradients.py

Core module. One function, no classes, no model-specific code. All five model scripts import from this file.

## The formula

Integrated Gradients for input dimension i:

```
IG_i(x) = (x_i - x'_i) * integral from 0 to 1 of (dF/dx_i)(x' + alpha * (x - x')) d_alpha
```

Where:
- x is the actual input
- x' is the baseline (uninformative reference)
- F is the model's scalar output (a logit, log-prob, action norm, etc.)
- alpha sweeps from 0 (baseline) to 1 (input)

The key property (completeness axiom): `sum of all IG_i = F(x) - F(x')`. The attributions add up to exactly the difference in model output.

## Code walkthrough

### Function signature

```python
def integrated_gradients(forward_fn, input_tensor, baseline_tensor, m=300):
```

- `forward_fn`: any callable that takes a tensor and returns a scalar. The caller builds this to target a specific output (e.g., a class log-prob, next-token log-prob, action norm).
- `input_tensor` / `baseline_tensor`: same shape, same device.
- `m=300`: number of interpolation steps. Default of 300 works well for most models. TinyLlama needs m=1000.

### Interpolation loop

```python
diff = input_tensor - baseline_tensor
sum_gradients = torch.zeros_like(input_tensor)

for k in range(m + 1):
    alpha = k / m
    interpolated = (baseline_tensor + alpha * diff).detach().requires_grad_(True)
    output = forward_fn(interpolated)
    grad = torch.autograd.grad(output, interpolated)[0]
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    sum_gradients += grad
    del interpolated, output, grad
    torch.cuda.empty_cache()
```

Step by step:
1. Create interpolated input at position alpha along the straight-line path from baseline to input. `.detach().requires_grad_(True)` makes it a fresh leaf tensor for gradient computation.
2. Forward pass through the model via forward_fn.
3. Compute gradient with `torch.autograd.grad()`. Returns dF/d(interpolated).
4. NaN protection. Zeroes out any NaN/inf gradients. Handles edge cases like zero embeddings through RMSNorm layers in LLMs.
5. Accumulate into a running sum.
6. Free memory. Critical for large models.

This is a Riemann sum with m+1 uniformly-spaced evaluation points (alpha = 0, 1/m, 2/m, ..., 1).

### Why autograd.grad instead of .backward()

During development, `.backward()` produced 280% completeness error on ResNet50, while `autograd.grad` + `model.requires_grad_(False)` gave 7.85%. The issue: `.backward()` computes gradients for all leaf tensors (including model parameters), which can interfere with the input gradient computation on deep models with many ReLU transitions.

### Attribution computation

```python
avg_gradients = sum_gradients / (m + 1)
attributions = diff * avg_gradients
```

Multiply the average gradient by (input - baseline) element-wise. This is the Riemann sum approximation of the path integral.

### Completeness check

```python
expected = (f_input - f_baseline).item()
actual = attributions.sum().item()
rel_error = abs(error / expected) * 100
```

Verifies `sum(attributions) = F(input) - F(baseline)`. Prints absolute and relative error. A low error means the Riemann sum is well-converged.

## Return value

Returns attributions tensor, same shape as input_tensor. Each element tells you how much that dimension of the input contributed to the output difference F(input) - F(baseline).

## Quadrature experiments

Gauss-Legendre quadrature was tested as an alternative to the Riemann sum. GL uses optimally-placed sample points and non-uniform weights. For smooth functions (ViT with GELU), GL at m=300 gave 0.18% error vs Riemann's 0.14%. For non-smooth functions (ResNet with ReLU), GL showed wildly non-monotonic convergence (0.25% at m=256 but 14.46% at m=500). For LLMs (TinyLlama), GL was much worse than Riemann (85.92% vs 25.91% at m=300).

Conclusion: the Riemann sum is more reliable across model architectures. GL's optimal point placement can alias with ReLU gradient discontinuities and transformer internal structure.

## Completeness results across all models

| Model | m | Error | Notes |
|-------|---|-------|-------|
| ViT-B/16 | 300 | 0.14% | Best. Smooth GELU activations |
| TinyLlama 1.1B | 1000 | 0.55% | Needs high m due to sharp SiLU regions |
| RDT-1B (state) | 300 | 6.12% | IG on raw 128-dim `state_vec` (pre-adaptor), path runs through `state_adaptor` (mlp3x_gelu) + 5-step chain so attribution at `MANISKILL_INDICES` is per-joint. Target is `log π` (σ²=1 Gaussian around final denoised mean). |
| RDT-1B (language) | 300 | 5.11% | BOS/EOS T5 baseline is close to real encodings (was 13.71% with the earlier zeros baseline); under real-render conditioning the gap is now tiny (0.0019) |
| RDT-1B (vision) | 300 | 9.06% | Slot-3-only gray baseline (five non-camera slots identical between input and baseline); real PickCube-v1 render as input |
| ResNet50 | 300 | 13.13% | ReLU gradient discontinuities + log_softmax |
| LLaVA 1.5-7B | 300 | 28.26% | 4-bit quantization gradient noise |
