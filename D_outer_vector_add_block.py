"""
## Puzzle 4: Outer Vector Add Block

Add a row vector to a column vector.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

$$z_{j, i} = x_i + y_j\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test


def add_vec_block_spec(x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]) -> Float32[Tensor, "90 100"]:
    # (1, 100) + (90, 1) -> (90, 100) # (N1, N0)
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr): # col_xrange row_yrange
    xid = tl.program_id(0)
    yid = tl.program_id(1)
    # print(f"pid: {xid} {yid}")

    col_xrange = tl.arange(0, B0) + xid * B0
    row_yrange = tl.arange(0, B1) + yid * B1

    x = tl.load(x_ptr + col_xrange, mask=col_xrange<N0) # (B0,)
    y = tl.load(y_ptr + row_yrange, mask=row_yrange<N1) # (B1,)
    # print(f"\n{x=}, \n{y=}")

    z = x[None, :] + y[:, None] # (B1, B0)
    # print(f"pid: {xid} {yid} => \n{z}")
    zrange = row_yrange[:, None] * N0 + col_xrange[None, :] # (B1, B0) # (row, col) => (row * num_cols + col)
    tl.store(z_ptr + zrange, z, mask=(col_xrange[None, :]<N0) & (row_yrange[:, None] < N1))
    return

if __name__ == "__main__": test(add_vec_block_kernel, add_vec_block_spec, nelem={"N0": 100, "N1": 90}) # BlockDim(4, 3, 1)