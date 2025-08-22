"""
## Puzzle 3: Outer Vector Add

Add two vectors.

Uses one program block axis. Block size `B0` is always the same as vector `x` length `N0`.
Block size `B1` is always the same as vector `y` length `N1`.


$$z_{j, i} = x_i + y_j\text{ for } i = 1\ldots B_0,\ j = 1\ldots B_1$$

"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test

def add_vec_spec(x: Float32[Tensor, "32"], y: Float32[Tensor, "32"]) -> Float32[Tensor, "32 32"]:
    return x[None, :] + y[:, None]

@triton.jit
def add_vec_kernel(
    x_ptr, # input pointer
    y_ptr, # input pointer
    z_ptr, # output pointer
    N0, # length of x
    N1, # length of y
    B0: tl.constexpr, # block size of x; same as N0
    B1: tl.constexpr  # block size of y; same as N1
):
    col_xrange = tl.arange(0, B0) # [0, ..., B0]
    row_yrange = tl.arange(0, B1) # [0, ..., B1]
    z_range = row_yrange[:, None] * N0 + col_xrange[None, :] # (B1, B0) # A[row, col] <=> A[row * n_cols + col]
    x = tl.load(x_ptr + col_xrange, mask=col_xrange < N0) # (32,)
    y = tl.load(y_ptr + row_yrange, mask=row_yrange < N1) # (32,)
    z = x[None, :] + y[:, None] # (1, B0) + (B1, 1) => (B1, B0)=(32, 32)
    tl.store(z_ptr + z_range, z, mask=z_range<N0*N1)
    return

if __name__ == "__main__": test(add_vec_kernel, add_vec_spec, nelem={"N0": 32, "N1": 32}, viz=False)