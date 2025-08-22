"""
## Puzzle 1: Constant Add

Add a constant to a vector. Uses one program id axis. Block size `B0` is always the same as vector `x` with length `N0`.


$$z_i = 10 + x_i \text{ for } i = 1\ldots N_0$$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test


def add_spec(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.


@triton.jit
def add_kernel(
    x_ptr, # input pointer
    z_ptr, # output pointer
    N0,    # length of input & output # 32
    B0: tl.constexpr # block size     # 32
):
    col_idx = tl.arange(0, B0)    # [0, 1, 2, ..., 31]
    x = tl.load(x_ptr + col_idx)
    z = x + 10.
    tl.store(z_ptr + col_idx, z)

if __name__ == "__main__": test(add_kernel, add_spec, nelem={"N0": 32}, viz=False)
