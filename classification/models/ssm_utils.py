import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
import record_utils

def sparsity(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return 0.0
    return (x == 0).sum() / x.numel()

def selective_scan_ref_v2(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False, scan_using_mean=False, layer_name=None):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = torch.float32
    A = A.to(dtype_in)
    B = B.to(dtype_in)
    C = C.to(dtype_in)
    D = D.to(dtype_in) if D is not None else None
    z = z.to(dtype_in) if z is not None else None
    delta = delta.to(dtype_in) if delta is not None else None
    delta_bias = delta_bias.to(dtype_in) if delta_bias is not None else None

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    # if A.is_complex():
    #     if is_variable_B:
    #         B = torch.view_as_complex(rearrange(B, "... (L two) -> ... L two", two=2))
    #     if is_variable_C:
    #         C = torch.view_as_complex(rearrange(C, "... (L two) -> ... L two", two=2))
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])

            deltaB = torch.einsum('bdl,bdnl->bdln', delta, B)
            deltaB_u = torch.einsum('bdln,bdl->bdln', deltaB, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    if layer_name is not None:
        # print("Using mean before scan")
        if layer_name not in record_utils.weight_diff_accumulator:
            record_utils.weight_diff_accumulator[layer_name] = {}
        
        mean_deltaA = deltaA.mean(dim=1, keepdim=True)
        if "deltaA" not in record_utils.weight_diff_accumulator[layer_name]:
            record_utils.weight_diff_accumulator[layer_name]["deltaA"] = {
                "l1": torch.zeros_like(mean_deltaA).mean(dim=0, keepdim=True),
                "l2": torch.zeros_like(mean_deltaA).mean(dim=0, keepdim=True)
            }
        record_utils.weight_diff_accumulator[layer_name]["deltaA"]['l1'] += (deltaA - mean_deltaA).abs().mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
        record_utils.weight_diff_accumulator[layer_name]["deltaA"]['l2'] += torch.sqrt(((deltaA - mean_deltaA) ** 2).mean(dim=1, keepdim=True)).mean(dim=0, keepdim=True)
        
        mean_deltaB_u = deltaB_u.mean(dim=1, keepdim=True)
        if "deltaB_u" not in record_utils.weight_diff_accumulator[layer_name]:
            record_utils.weight_diff_accumulator[layer_name]["deltaB_u"] = {
                "l1": torch.zeros_like(mean_deltaB_u).mean(dim=0, keepdim=True),
                "l2": torch.zeros_like(mean_deltaB_u).mean(dim=0, keepdim=True)
            }
        record_utils.weight_diff_accumulator[layer_name]["deltaB_u"]['l1'] += (deltaB_u - mean_deltaB_u).abs().mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
        record_utils.weight_diff_accumulator[layer_name]["deltaB_u"]['l2'] += torch.sqrt(((deltaB_u - mean_deltaB_u) ** 2).mean(dim=1, keepdim=True)).mean(dim=0, keepdim=True)
        
        mean_C = C.mean(dim=1, keepdim=True)
        if "C" not in record_utils.weight_diff_accumulator[layer_name]:
            record_utils.weight_diff_accumulator[layer_name]["C"] = {
                "l1": torch.zeros_like(mean_C).mean(dim=0, keepdim=True),
                "l2": torch.zeros_like(mean_C).mean(dim=0, keepdim=True)
            }
        record_utils.weight_diff_accumulator[layer_name]["C"]['l1'] += (C - mean_C).abs().mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
        record_utils.weight_diff_accumulator[layer_name]["C"]['l2'] += torch.sqrt(((C - mean_C) ** 2).mean(dim=1, keepdim=True)).mean(dim=0, keepdim=True)
    
    if scan_using_mean:
        deltaA = deltaA.mean(dim=1, keepdim=True)
        deltaB_u = deltaB_u.mean(dim=1, keepdim=True)
        C = C.mean(dim=1, keepdim=True)
        x = x.mean(dim=1, keepdim=True)
    
    zero_indices = []
    for i in range(u.shape[2]):
        # if torch.all(deltaB_u[:, :, i] == 0):
        #     print("Found sparse deltaB_u")
        #     ys.append(torch.zeros_like(u[:, :, 0]))
        #     ys.append(ys[-1] if len(ys) > 0 else torch.zeros_like(u[:, :, 0]))
        #     zero_indices.append(i)
        #     continue
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        # if y.is_complex():
        #     y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    # print(f"Sparsity of output of SSM: y={sparsity(y)}")
    # if D, let index of 0 in y be 0 in D
    if D is not None:
        u_D = u * rearrange(D, "d -> d 1")
        # print(u_D.shape)
        # u_D[zero_indices] = 0
        out = y + u_D
    # print(f"Sparsity of output of SSM after adding u_D: out={sparsity(out)}")
    # out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state.float())