"""
## Puzzle 2: Constant Add Block

Add a constant to a vector. Uses one program block axis (no `for` loops yet). Block size `B0` is now smaller than the shape vector `x` which is `N0`.

$$z_i = 10 + x_i \text{ for } i = 1\ldots N_0$$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test

def add2_spec(x: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    return x + 10.

@triton.jit
def add_mask2_kernel(
    x_ptr, z_ptr, 
    N0, # 200 # length of input and output
    B0: tl.constexpr # 32 # block size
):
    # print(f"{x_ptr=}, {z_ptr=}, {N0=}, {B0=}")
    # 1 BLOCK: [T0, T1, T2, T3, T4, T5, T6]
    # Like this there are 32 BLOCKS
    pid = tl.program_id(0)
    col_idx = tl.arange(0, B0) + pid * B0 # [0, ..., 31] + 0...6 * 32 => [0:31, 32:63, 64:95, 96:127, 128:159, 160:191, 192:223]
    mask = col_idx < N0
    x = tl.load(x_ptr + col_idx, mask, other=0)
    z = x + 10
    tl.store(z_ptr + col_idx, z, mask)
    return z_ptr

if __name__ == "__main__": test(add_mask2_kernel, add2_spec, nelem={"N0": 200}, viz=False)