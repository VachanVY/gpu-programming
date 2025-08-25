"""
## Puzzle 8: Long Softmax


Softmax of a batch of logits.

Uses one program block axis. Block size `B0` represents the batch of `x` of length `N0`.
Block logit length `T`.   Process it `B1 < T` elements at a time.  

$$z_{i, j} = \text{softmax}(x_{i,1} \ldots x_{i, T}) \text{ for } i = 1\ldots N_0$$

Note softmax needs to be computed in numerically stable form as in Python. In addition in Triton they recommend not using `exp` but instead using `exp2`. You need the identity

$$\exp(x) = 2^{\log_2(e) x}$$

Advanced: there one way to do this with 3 loops. You can also do it with 2 loops if you are clever. Hint: you will find this identity useful:

$$\exp(x_i - m) =  \exp(x_i - m/2 - m/2) = \exp(x_i - m/ 2) /  \exp(m/2) $$
"""

import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test

def softmax_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4 200"]:
    x_max = x.max(1, keepdim=True)[0] # (4, 1)
    x = x - x_max # (4, 200)
    x_exp = x.exp() # (4, 200)
    return x_exp / x_exp.sum(1, keepdim=True) # (4, 200)

LOG2_E = 1.44269504

# NAIVE VERSION -- 3 for loops
@triton.jit
def softmax_kernel(
    x_ptr, # input pointer
    z_ptr, # output pointer
    N0,    # number of rows
    N1, # NOT USED IGNORE
    T,     # length of each row; number of columns
    B0: tl.constexpr, # block size along batch dimension
    B1: tl.constexpr
):
    pid_0 = tl.program_id(0)

    Bidx = tl.arange(0, B0) + pid_0 * B0 # (B0=1,)
    
    # find max
    x_max = tl.full((B0,), value=float("-inf"), dtype=tl.float32) # (B0=1,)
    for j in tl.range(0, T, B1): # [0, B1, 2*B1, 3*B1, ...]
        # find column indices
        Tidx = tl.arange(0, B1) + j # (B1,)

        # (row, col) => (row * num_cols + col)
        BT_idx = Bidx[:, None] * T + Tidx[None, :] # (B0=1, B1)
        BT_mask = (Bidx[:, None] < N0) & (Tidx[None, :] < T) # (B0=1, B1)

        x = tl.load(x_ptr + BT_idx, mask=BT_mask) # (B0=1, B1)
        # find max
        x_max = tl.maximum(
            x_max,   # (B0=1,)
            x.max(1) # (B0=1,)
        ) # (B0=1,)

    # exp sum
    expf = lambda x: tl.exp2(LOG2_E * x)
    exp_sum = tl.zeros((B0,), dtype=tl.float32)
    for j in tl.range(0, T, B1):
        # find column indices
        Tidx = tl.arange(0, B1) + j # (B1,)

        # (row, col) => (row * num_cols + col)
        BT_idx = Bidx[:, None] * T + Tidx[None, :] # (B0=1, B1)
        BT_mask = (Bidx[:, None] < N0) & (Tidx[None, :] < T) # (B0=1, B1)

        x = tl.load(x_ptr + BT_idx, mask=BT_mask) # (B0=1, B1)
        # compute sum
        exp_sum += expf(x - x_max[:, None]).sum(axis=1)

    # softmax -- normalization with exp_sum
    for j in tl.range(0, T, B1):
        # find column indices
        Tidx = tl.arange(0, B1) + j # (B1,)

        # (row, col) => (row * num_cols + col)
        BT_idx = Bidx[:, None] * T + Tidx[None, :] # (B0=1, B1)
        BT_mask = (Bidx[:, None] < N0) & (Tidx[None, :] < T) # (B0=1, B1)

        x = tl.load(x_ptr + BT_idx, mask=BT_mask) # (B0=1, B1)
        z = expf(x - x_max[:, None])/exp_sum[:, None] # (B0=1, B1)/(B0=1, 1)
        tl.store(z_ptr + BT_idx, z, BT_mask)
    return


# TODO(VachanVY): Implement Optimized Version

if __name__ == "__main__":
    test(softmax_kernel, softmax_spec, B={"B0": 1, "B1":32},
     nelem={"N0": 4, "N1": 32, "T": 200})