from torch.autograd import Function
import torch.nn.functional as F
import torch

class NearestUpsample(Function):
    @staticmethod
    def forward(ctx, x, target_size):
        """
        Forward pass for nearest neighbor upsampling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            target_size (tuple): Target size (target_H, target_W).
        Returns:
            torch.Tensor: Upsampled tensor of shape (B, C, target_H, target_W).
        """
        B, C, H, W = x.shape
        target_H, target_W = target_size

        # Save input shape and target size for backward
        ctx.save_for_backward(torch.tensor([H, W], device=x.device), torch.tensor([target_H, target_W], device=x.device))

        # Perform nearest upsampling
        grid_H = torch.linspace(0, H - 1, target_H, device=x.device).round().long()
        grid_W = torch.linspace(0, W - 1, target_W, device=x.device).round().long()
        out = x[:, :, grid_H, :][:, :, :, grid_W]  # Nearest neighbor indexing
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for nearest neighbor upsampling.
        Args:
            grad_output (torch.Tensor): Gradient from the next layer, of shape (B, C, target_H, target_W).
        Returns:
            torch.Tensor: Gradient with respect to the input tensor, of shape (B, C, H, W).
        """
        # Retrieve saved tensors
        input_shape, target_size = ctx.saved_tensors
        H, W = input_shape.tolist()
        target_H, target_W = target_size.tolist()

        # Perform bilinear downsampling to approximate the gradient
        grad_input = F.interpolate(
            grad_output, size=(H, W), mode='bilinear', align_corners=False
        )
        return grad_input, None  # No gradients w.r.t. target_size

# Wrapper function
def nearest_upsample(x, target_size):
    return NearestUpsample.apply(x, target_size)
