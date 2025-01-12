import torch
import torch.nn as nn
import torch.nn.functional as F
torch.random.manual_seed(0)
B = 128
L = 3136
D = 256
K = 4

xs = torch.randn(B, 4, D, L)
x_proj_weight = torch.randn(10, D)
# x_proj_weight = torch.stack([x_proj_weight for _ in range(4)], dim=0)
x_proj_bias = None

print(xs.shape)
print(xs.view(B, -1, L).shape)
print(x_proj_weight.view(-1, D, 1).shape)
input_d = D
output_d = x_proj_weight.size(0)                    
linear_layer = nn.Linear(
    in_features=input_d, 
    out_features=output_d, 
    bias=(x_proj_bias is not None)
)
linear_layer.weight = nn.Parameter(x_proj_weight.view(-1, D), requires_grad=False)
x_dbl_linear = []
for i in range(4):
    x_dbl_linear.append(linear_layer(xs[:, i, :].transpose(1, 2)))
x_dbl_linear = torch.concat(x_dbl_linear, dim=2).transpose(1, 2)
x_proj_weight = torch.stack([x_proj_weight for _ in range(4)], dim=0)
x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
print(x_dbl.shape) # 128, 40, 3136
print(x_dbl_linear.shape)
print("Outputs match:", torch.allclose(x_dbl, x_dbl_linear, atol=1e-6))
# dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
# dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)