import torch
import math

# Define both implementations
def quatermap_with_max_pooling(x, interval, n_out_of_interval=1):
    H, W = x.shape
    
    # Generate index-based mask for the height
    indices_h = torch.arange(0, math.ceil(H / interval) * interval, device=x.device)
    indices_h = indices_h.view(-1, interval)[:, :n_out_of_interval].flatten()
    indices_h = indices_h[indices_h < H]

    # Generate index-based mask for the width
    indices_w = torch.arange(0, math.ceil(W / interval) * interval, device=x.device)
    indices_w = indices_w.view(-1, interval)[:, :n_out_of_interval].flatten()
    indices_w = indices_w[indices_w < W]

    # Construct a mask
    mask = torch.zeros(H, W, device=x.device)
    mask[indices_h[:, None], indices_w] = 1
    print(mask)

    # Apply the mask
    masked_x = x * mask  # Apply mask (retains gradient flow)

    # Perform max pooling over the masked values
    print(masked_x)
    pooled_x = torch.nn.functional.avg_pool2d(masked_x.unsqueeze(0).unsqueeze(0), kernel_size=interval, stride=interval)
    pooled_x = pooled_x * (interval ** 2)
    return pooled_x.squeeze(0).squeeze(0)

def quatermap(x, interval, n_out_of_interval=1):
    H, W = x.shape
    if n_out_of_interval == 1:
        return x[::interval, ::interval].contiguous()
    else:
        indices = torch.arange(
            0, math.ceil(H / interval) * interval, device=x.device
        )
        indices = indices.view(-1, interval)[:, :n_out_of_interval].flatten()
        indices = indices[indices < H]
        return x[indices, :][:, indices].contiguous()

# Test script
def test_quatermap():
    # Test parameters
    H, W = 4, 4  # Small dimensions for easier debugging
    interval = 2
    n_out_of_interval = 1
    
    # Generate a random tensor
    x = torch.randn(H, W)
    print(x)

    # Apply both implementations
    output_pooling = quatermap_with_max_pooling(x, interval, n_out_of_interval)
    print(output_pooling)
    output_indexing = quatermap(x, interval, n_out_of_interval)
    print(output_indexing)
    
    # Compare outputs
    assert torch.allclose(output_pooling, output_indexing), "Outputs do not match!"

    print("Test passed: Outputs are the same for both implementations.")

# Run the test
if __name__ == "__main__":
    test_quatermap()
