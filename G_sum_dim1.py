"""
## Puzzle 7: Long Sum

Sum of a batch of numbers.

Uses one program blocks. Block size `B0` represents a range of batches of  `x` of length `N0`.
Each element is of length `T`. Process it `B1 < T` elements at a time.  

$$z_{i} = \sum^{T}_j x_{i,j} =  \text{ for } i = 1\ldots N_0$$

Hint: You will need a for loop for this problem. These work and look the same as in Python.
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test


def sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

@triton.jit
def sum_kernel(
    x_ptr, # input pointer  # (N0, T)
    z_ptr, # output pointer # (N0,)
    N0,    # number of Batches
    N1, # NOT USED IGNORE
    T,     # length of each row i.e number of columns
    B0: tl.constexpr, # BLOCK ALONG BATCH DIMENSION
    B1: tl.constexpr  # BLOCK ALONG REDUCTION DIMENSION # sum over iter(B1) if B1_i < T only
):
    xid = tl.program_id(0)

    Bidx = tl.arange(0, B0) + xid * B0 # (B0=1,)
    z = tl.zeros((B0,), tl.float32) # (B0=1,)

    for j in tl.range(0, T, B1):
        Tidx = tl.arange(0, B1) + j # (B1,)

        # (row, col) => (row * num_cols + col)
        BT_idx = Bidx[:, None] * T + Tidx[None, :]
        xmask = (Bidx[:, None] < N0) & (Tidx[None, :] < T)

        x = tl.load(x_ptr + BT_idx, mask=xmask, other=0.0) # (B0=1, B1)
        z += x.sum(1) # (B0=1,)

    tl.store(z_ptr + Bidx, z, mask=Bidx < N0)
    return

if __name__ == "__main__": test(sum_kernel, sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200}) # DimGrid(4, 1, 1)