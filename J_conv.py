"""
## Puzzle 10: Two Dimensional Convolution

A batched 2D convolution.

Uses one program id axis. Block size `B0` represent the batches to process out of `N0`.
Image `x` is size is `H` by `W` with only 1 channel, and kernel `k` is size `KH` by `KW`.

$$z_{i, j, k} = \sum_{oj, ok} k_{oj,ok} \times x_{i,j + oj, k + ok} \text{ for } i = 1\ldots N_0$$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test


def conv2d_spec(x: Float32[Tensor, "4 8 8"], k: Float32[Tensor, "4 4"]) -> Float32[Tensor, "4 8 8"]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    # print(f"{x.shape=}") x.shape=torch.Size([4, 12, 12])
    for i in range(8):
        for j in range(8):
            # print(f"mul: {k[None, :, :].shape=}\t{x[:, i: i+4, j: j + 4].shape=}")
            # mul: k[None, :, :].shape=torch.Size([1, 4, 4])	x[:, i: i+4, j: j + 4].shape=torch.Size([4, 4, 4])
            z[:, i, j] = (k[None, :, :] * x[:, i: i+4, j: j + 4]).sum((1, 2)) # (1, 4, 4) * (4, 4, 4) => (C=4, H=4, W=4) ==.sum(1, 2)=> (C=4,)
    return z


@triton.jit
def conv2d_kernel(
    x_ptr, # (C, H, W)
    k_ptr, # (KH, KW)
    z_ptr, # (C, Ho, Wo)
    N0,    # 
    H, W, # IMAGE SIZE
    KH: tl.constexpr, KW: tl.constexpr, # KERNEL SIZE
    B0: tl.constexpr # block size along CHANNEL DIMENSION
):
    cid = tl.program_id(0)
    Cidx = (tl.arange(0, B0) + cid * B0)[:, None, None] # (B0, 1, 1)

    k_idx = tl.arange(0, KH)[:, None] * KW + tl.arange(0, KW)[None, :] # (KH, KW)
    k = tl.load(k_ptr + k_idx)[None] # (1, KH, KW)
    for h in tl.range(0, H):
        hidx = tl.arange(h, h + KH)[None, :, None] # (1, KH, 1)
        for w in tl.range(0, W):
            widx = tl.arange(w, w + KW)[None, None, :] # (1, 1, KW)

            # indexing
            # [ch, row, col] => [ch * num_rows * num_cols   +   row * num_cols  +  col]
            #           # (B0, 1, 1)                   # (1, KH, 1)                # (1, 1, KW)
            xrange = Cidx * H * W  +  hidx * W  +  widx # (B0, KH, KW)
            xmask = (Cidx < N0) & (hidx < H) & (widx < W)
            xhw = tl.load(x_ptr + xrange, mask=xmask, other=0.0) # (B0=1, KH, KW)

            # conv
            hz_idx = tl.arange(h, h + 1)[None, :, None]
            wz_idx = tl.arange(w, w+1)[None, None, :]

            zhw = (xhw * k).sum(1, keep_dims=True).sum(2, keep_dims=True) # (B0=1, 1, 1)
            zrange = Cidx * H * W  +  hz_idx * W  +  wz_idx # idx = [Cidx, h, w]
            zmask = (Cidx < N0) & (hz_idx < H) & (wz_idx < W)

            tl.store(z_ptr + zrange, zhw, mask=zmask)
    return

if __name__ == "__main__":
    # got this correct on my first run! yey!
    test(conv2d_kernel, conv2d_spec, B={"B0": 1}, nelem={"N0": 4, "H": 8, "W": 8, "KH": 4, "KW": 4}) # DimGrid(4, 1, 1) # 4 for 4 batches