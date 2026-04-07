import math
from typing import Callable, Tuple

import torch

last_unmerge = None
last_H = None
last_W = None

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    # print("metric", metric.shape)   
    # We can only reduce by a maximum of 50% tokens
    # t = metric.shape[1]
    t = metric.shape[2]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # metric = metric / metric.norm(dim=-1, keepdim=True)
        # a, b = metric[..., ::2, :], metric[..., 1::2, :]
        # scores = a @ b.transpose(-1, -2)

        metric = metric / metric.norm(dim=-2, keepdim=True)
        a, b = metric[..., :, ::2], metric[..., :, 1::2]
        scores = a.transpose(-1, -2) @ b
        
        # print("metric", metric.shape)
        # print("a", a.shape)
        # print("b", b.shape)
        # print("scores", scores.shape)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        # edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        # src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None, :]

        unm_idx = edge_idx[..., :, r:]  # Unmerged Tokens
        src_idx = edge_idx[..., :, :r]  # Merged Tokens
        dst_idx = node_idx[..., None, :].gather(dim=-1, index=src_idx)
        
        # print("node_max", node_max.shape)
        # print("node_idx", node_idx.shape)
        # print("edge_idx", edge_idx.shape)
        # print("unm_idx", unm_idx.shape)
        # print("src_idx", src_idx.shape)
        # print("dst_idx", dst_idx.shape)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=2)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., :, ::2], x[..., :, 1::2]
        n, c, t1 = src.shape
        unm = src.gather(dim=-1, index=unm_idx.expand(n, c, t1 - r))
        src = src.gather(dim=-1, index=src_idx.expand(n, c, r))
        dst = dst.scatter_reduce(-1, dst_idx.expand(n, c, r), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :, :1], dst[:, :, :1], unm[:, :, 1:], dst[:, :, 1:]], dim=2)
        else:
            return torch.cat([unm, dst], dim=2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[2]
        unm, dst = x[..., :, :unm_len], x[..., :, unm_len:]
        n, c, _ = unm.shape

        src = dst.gather(dim=-1, index=dst_idx.expand(n, c, r))

        out = torch.zeros(n, c, metric.shape[2], device=x.device, dtype=x.dtype)

        out[..., :, 1::2] = dst
        out.scatter_(dim=-1, index=(2 * unm_idx).expand(n, c, unm_len), src=unm)
        out.scatter_(dim=-1, index=(2 * src_idx).expand(n, c, r), src=src)

        return out

    return merge, unmerge, r

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[:, 0, None, :])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size

# def calculate_new_hw(h, w, r):
#     # original size h * w
#     # r is the number of tokens to remove

#     token_lengths = h * w
#     token_lengths -= r
#     # remain ratio
#     aspect_ratio = w / h

#     # Calculate new height and new width
#     new_height = math.sqrt(token_lengths / aspect_ratio)
#     new_width = aspect_ratio * new_height

#     return int(new_height), int(new_width)