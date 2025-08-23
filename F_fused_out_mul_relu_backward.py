"""
## Puzzle 6: Fused Outer Multiplication - Backwards


Backwards of a function that multiplies a matrix with a row vector and take a relu.

Uses two program blocks. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`. Chain rule backward `dz`
is of shape `N1` by `N0`

$$f(x, y) = \text{relu}(x_i \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$$

$$dx_{i, j} = f_x'(x, y)_{i, j} \times dz_{i,j}$$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test

@torch.no_grad()
def my_torch_backward(
    x: Float32[Tensor, "90 100"], y: Float32[Tensor, "90"],
    dz: Float32[Tensor, "90 100"]
) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()

    dL_dh = ((x * y[:, None]) > 0).float() * dz.clone()
    dL_dx = dL_dh * y[:, None]
    return dL_dx


def mul_relu_block_back_spec(x: Float32[Tensor, "90 100"], y: Float32[Tensor, "90"],
                             dz: Float32[Tensor, "90 100"]) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    h = x * y[:, None]
    z = torch.relu(h)

    z.backward(dz)
    dx = x.grad
    my_dx = my_torch_backward(x, y, dz)
    torch.testing.assert_close(dx, my_dx)
    return dx

@triton.jit
def mul_relu_block_back_kernel(
    x_ptr,  # input pointer # x: (N1, N0)
    y_ptr,  # input pointer # y: (N1,)
    dz_ptr, # output_grad pointer # dz: (N1, N0)
    dx_ptr, # input_grad pointer # dx: (N1, N0)
    N0,     # number of rows
    N1,     # number of columns
    B0: tl.constexpr,
    B1: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    col_xrange = tl.arange(0, B0) + pid_0 * B0 # (B0,)
    row_yrange = tl.arange(0, B1) + pid_1 * B1 # (B1,)

    # (row, col) => (row * num_cols + col)
    offsets = row_yrange[:, None] * N0 + col_xrange[None, :]    # (B1, B0)
    offsets_mask = (col_xrange[None, :] < N0) & (row_yrange[:, None] < N1) # (B1, B0)

    x = tl.load(x_ptr + offsets, mask=offsets_mask)
    y = tl.load(y_ptr + row_yrange, mask=row_yrange<N1)
    dL_dz = tl.load(dz_ptr + offsets, mask=offsets_mask)
    
    dL_dh = ((x * y[:, None]) > 0) * dL_dz
    dL_dx = dL_dh * y[:, None]

    tl.store(dx_ptr + offsets, value=dL_dx, mask=offsets_mask)
    return

if __name__ == "__main__": test(mul_relu_block_back_kernel, mul_relu_block_back_spec, nelem={"N0": 100, "N1": 90}) # DimGrid(4, 3, 1)