import torch

def integrated_gradients(forward_fn, input_tensor, baseline_tensor, m=300):
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

    avg_gradients = sum_gradients / (m + 1)
    attributions = diff * avg_gradients

    #completeness check
    with torch.no_grad():
        f_input = forward_fn(input_tensor)
        f_baseline = forward_fn(baseline_tensor)
    expected = (f_input - f_baseline).item()
    actual = attributions.sum().item()
    error = abs(expected - actual)
    rel_error = abs(error / expected) * 100 if expected != 0 else float('inf')
    print(f"completeness: expected={expected:.4f}, actual={actual:.4f}, error={error:.4f} ({rel_error:.2f}%)")

    return attributions
