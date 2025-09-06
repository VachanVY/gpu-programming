"""
## Puzzle 11: Matrix Multiplication

A blocked matrix multiplication.

Uses three program id axes. Block size `B2` represent the batches to process out of `N2`.
Block size `B0` represent the rows of `x` to process out of `N0`. Block size `B1` represent the cols of `y` to process out of `N1`. The middle shape is `MID`.

$$z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$$

You are allowed to use `tl.dot` which computes a smaller mat mul.

Hint: the main trick is that you can split a matmul into smaller parts.

$$z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} +  \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k} $$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test

def dot_spec(x: Float32[Tensor, "4 32 32"], y: Float32[Tensor, "4 32 32"]) -> Float32[Tensor, "4 32 32"]:
    return x @ y


# (A , B, C ) @ (A , C , D) => (A, B, D)
# (N2,N0,MID) @ (N2,MID,N1) => (N2,N0,N1)
@triton.jit
def dot_kernel(
    x_ptr, # (N2,N0,MID)
    y_ptr, # (N2,MID,N1)
    z_ptr, # (N2,N0,N1)
    N0, # NUM ROWS      # 32
    N1, # NUM COLS      # 32
    N2, # NUM BATCHES   # 4
    MID,# MID DIMENSION # 32
    B0: tl.constexpr, # BLOCK SIZE ALONG N0 (rows)    # 16
    B1: tl.constexpr, # BLOCK SIZE ALONG N1 (cols)    # 16
    B2: tl.constexpr, # BLOCK SIZE ALONG N2 (batches) # 1
    B_MID: tl.constexpr # BLOCK SIZE ALONG MID        # 16
):
    # print(f"{N0=}, {N1=}, {N2=}, {MID=}, {B0=}, {B1=}, {B2=}, {B_MID=}")
    pid_0 = tl.program_id(0) # for col dim   # B1
    pid_1 = tl.program_id(1) # for row dim   # B0
    pid_2 = tl.program_id(2) # for batch dim # B2

    bidx    = (tl.arange(0, B2) + pid_2 * B2)[:, None, None] # (B2, 1, 1)
    row_idx = (tl.arange(0, B0) + pid_1 * B0)[None, :, None] # (1, B0, 1)
    col_idx = (tl.arange(0, B1) + pid_0 * B1)[None, None, :] # (1, 1 ,B1)

    # always keep the the accumulator in higher dtype to prevent overflow
    accumulator = tl.zeros((B2, B0, B1), dtype=tl.float32) # (B2, B0, B1)
    for mid in tl.range(0, MID, B_MID): # [0:B_MID, B_MID:2*B_MID, 2*B_MID:3*B_MID, ...]
        mid_idx = (tl.arange(0, B_MID) + mid) # (B_MID,)

        # [b, r, c] => b * num_rows * num_cols  +  r * num_cols  +  c
        xrange = bidx * N0 * MID  +  row_idx * MID  +  mid_idx[None, None, :]
        xmask = (bidx < N2) & (row_idx < N0) & (mid_idx[None, None, :] < MID)
        x = tl.load(x_ptr + xrange, mask=xmask) # (B2, B0, B_MID)

        # [b, r, c] => b * num_rows * num_cols  +  r * num_cols  +  c
        yrange = bidx * MID * N1  +  mid_idx[None, :, None] * N1 + col_idx
        ymask  = (bidx < N2) & (mid_idx[None, :, None] < MID) & (col_idx < N1)
        y = tl.load(y_ptr + yrange, mask=ymask) # (B2, B_MID, B1)

        # matmul
        accumulator += tl.dot(x, y) # (B2, B0, B_MID) @ (B2, B_MID, B1) => (B2, B0, B1)

    # [b, r, c] => b * num_rows * num_cols  +  r * num_cols  +  c
    zrange = bidx * N0 * N1 + row_idx * N1 + col_idx
    zmask = (bidx < N2) & (row_idx < N0) & (col_idx < N1)
    tl.store(z_ptr + zrange, value=accumulator, mask=zmask)

if __name__ == "__main__":
    test(dot_kernel, dot_spec, B={"B0": 16, "B1": 16, "B2": 1, "B_MID": 16}, nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 32}) # DimGrid(2, 2, 4)

# pid: [0] [0] [0]
# pid: [0] [0] [1]
# pid: [0] [0] [2]
# pid: [0] [0] [3]
# pid: [0] [1] [0]
# pid: [0] [1] [1]
# pid: [0] [1] [2]
# pid: [0] [1] [3]
# pid: [1] [0] [0]
# pid: [1] [0] [1]
# pid: [1] [0] [2]
# pid: [1] [0] [3]
# pid: [1] [1] [0]
# pid: [1] [1] [1]
# pid: [1] [1] [2]
# pid: [1] [1] [3]