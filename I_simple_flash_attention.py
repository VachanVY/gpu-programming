"""
## Puzzle 9: Simple FlashAttention

A scalar version of FlashAttention.

Uses zero programs. Block size `B0` represents `k` of length `N0`.
Block size `B0` represents `q` of length `N0`. Block size `B0` represents `v` of length `N0`.
Sequence length is `T`. Process it `B1 < T` elements at a time.  

$$z_{i} = \sum_{j} \text{softmax}(q_1 k_1, \ldots, q_T k_T)_j v_{j} \text{ for } i = 1\ldots N_0$$

This can be done in 1 loop using a similar trick from the last puzzle.
"""


import torch
import triton
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32

from utils import test

LOG2_E = 1.44269504

def flashatt_spec(q: Float32[Tensor, "200"], k: Float32[Tensor, "200"], v: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft =  x_exp  / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)

# QUESTION WORDING IS HORRIBLE, COULDN"T UNDETSTAND A THING
# DON"T KNOW IF THIS IS RIGHT SOLUTION, IT PASSES THE TESTS THOUGH
# DISCLOSURE: I AM NOT SURE IF THIS IS THE INTENDED SOLUTION
# TODO(VachanVY): Correct this
@triton.jit
def flashatt_kernel(
    q_ptr, # query pointer
    k_ptr, # key pointer
    v_ptr, # value pointer
    z_ptr, # output pointer
    N0, # NOT USED IGNORE
    T,    # length of q, k, v
    B0: tl.constexpr # block size
):
    expf = lambda x : tl.exp2(LOG2_E * x)

    Tidx = tl.arange(0, B0)
    mask_q = Tidx < T

    q = tl.load(q_ptr + Tidx, mask_q, other=0.0)
    k = tl.load(k_ptr + Tidx, mask_q, other=0.0)
    v = tl.load(v_ptr + Tidx, mask_q, other=0.0)
    
    att_scores = q[:, None] * k[None, :]  # Shape: (256, 256)
    
    valid_mask = (Tidx[:, None] < T) & (Tidx[None, :] < T)
    att_scores = tl.where(valid_mask, att_scores, float('-inf'))
    
    row_max = tl.max(att_scores, axis=1, keep_dims=True)
    att_scores = att_scores - row_max
    
    att_scores = tl.where(valid_mask, att_scores, float('-inf'))
    
    att_exp = expf(att_scores)      
    att_sum = tl.sum(att_exp, axis=1, keep_dims=True)
    att_weights = att_exp / att_sum
    
    z = tl.sum(att_weights * v[None, :], axis=1)
    
    tl.store(z_ptr + Tidx, z, mask_q)
    return

if __name__ == "__main__":
    test(flashatt_kernel, flashatt_spec, B={"B0":256},  # Use 256 instead of 200 # error if 200 is used in `tl.arange`
        nelem={"N0": 200, "T": 200})