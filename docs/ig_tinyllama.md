# ig_tinyllama.py

Integrated Gradients on TinyLlama-1.1B-Chat. Attributes next-token prediction to input token embeddings. Outputs a per-token bar chart.

## How this differs from vision models

Vision IG attributes to raw pixels. LLM IG attributes to embedding vectors, the continuous representation that the model actually works with. You cannot take a gradient of a discrete token ID, so we use `inputs_embeds` instead of `input_ids`.

## Sections

### #model

```python
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32,
    device_map="auto",
)
```

Loaded in float32, not float16. This is critical. Float16 produces gradient overflow and NaN near the baseline due to limited precision through RMSNorm layers. TinyLlama is only 1.1B params, so float32 (4.4 GB) fits easily in VRAM.

### #embeddings

```python
embed_layer = model.get_input_embeddings()
with torch.no_grad():
    input_embeds = embed_layer(input_ids).float()
```

Converts token IDs to embedding vectors by looking them up in the model's embedding table. Shape: [1, seq_len, 2048] (TinyLlama's hidden dim is 2048).

### #target token

The model predicts "Paris" (id=3681) as the most likely next token after "The capital of France is".

### #forward_fn

```python
def forward_fn(embeds):
    out = model(inputs_embeds=embeds)
    logprobs = torch.log_softmax(out.logits[0, -1, :], dim=-1)
    return logprobs[next_token_id]
```

Feeds embeddings directly to the model (bypassing the embedding lookup). Returns the log-probability of "Paris" at the last position.

### #baseline -- why not zeros

Zero vectors cause a gradient singularity through RMSNorm:

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps)
```

At x=0, the gradient scales as 1/sqrt(eps), which is about 316 for eps=1e-5 (TinyLlama's rms_norm_eps, verified via AutoConfig; a model with eps=1e-6 would see ~1000). Through 22 transformer blocks this compounds to astronomical values. Even float32 overflows. The `nan_to_num` protection in integrated_gradients.py clips these to zero, but that destroys the gradient information at low alpha values and corrupts the integral.

The fix: PAD token embeddings as baseline.

```python
pad_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
baseline = embed_layer(pad_ids).float()
```

PAD (id=2, same as EOS in TinyLlama) has a learned embedding that represents "no information." Semantically similar to what zeros are supposed to mean, but numerically stable.

### #per-token attribution

```python
token_attr = attr.squeeze(0).sum(dim=-1).detach().cpu().numpy()
```

The raw attributions have shape [1, seq_len, 2048]. To get a single attribution per token, sum over the embedding dimension (dim=-1). This gives [seq_len] values. Sign is preserved (not abs) to show whether each token pushes toward or away from the prediction.

### #visualize

Horizontal bar chart. Red bars = positive attribution (pushes toward predicting "Paris"). Blue bars = negative (pushes away).

## Output

`output/ig_tinyllama.png`

Bar chart for "The capital of France is" -> "Paris":
- "France" has the highest positive attribution
- "capital" is second
- Other tokens have smaller contributions

## Completeness

0.55% error at m=1000. This model needs more integration steps than the vision models because:

1. The PAD baseline is not perfectly "uninformative" -- it has its own learned semantics
2. The embedding space path crosses many sharp regions (SiLU activations in the MLP layers)
3. The 2048-dimensional embedding space means small per-element errors accumulate

Convergence (Riemann sum, PAD baseline):

| m | error |
|---|-------|
| 300 | 25.91% |
| 500 | 9.96% |
| 1000 | 0.55% |

Convergence is approximately O(1/m^2), consistent with SiLU being a smooth activation. m=1000 is needed to get under 3%.
