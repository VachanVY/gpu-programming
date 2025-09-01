import time
import ctypes

import torch
from torch import Tensor, nn

sync = lambda: torch.cuda.synchronize()
seed = lambda s: torch.manual_seed(s)

def torch_softmax(x:Tensor):
    seed(0)
    torch_out = torch.empty_like(x)
    sync()

    t0 = time.time()
    torch.softmax(x, dim=1, out=torch_out)
    sync()
    t1 = time.time()
    return torch_out, (t1 - t0) * 1000

def my_softmax(x:Tensor, fname:str):
    seed(0)
    lib = ctypes.CDLL(fname)
    lib.softmax.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t
    ]

    out = torch.empty_like(x)
    sync()

    A, B = x.shape
    t0 = time.time()
    lib.softmax(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        A, B
    )
    sync()
    t1 = time.time()
    return out, (t1 - t0) * 1000

if __name__ == "__main__":
    (A, B) = SHAPE = (1024, 32768)
    TIMES = 10

    softmax_v0 = lambda x: my_softmax(x, "./libsoftmax_v0.so")
    softmax_v1 = lambda x: my_softmax(x, "./libsoftmax_v1.so")

    x = torch.randn(A, B, device="cuda", dtype=torch.float32)
    torch_out, torch_time = torch_softmax(x)
    v0_out, v0_time = softmax_v0(x)
    v1_out, v1_time = softmax_v1(x)

    print("My softmax_v0 comparison:")
    print("Max abs diff:", (abs_diff:=(v0_out - torch_out).abs()).max().item())
    print("Mean abs diff:", abs_diff.mean().item())
    print(f"v0 Time: {v0_time:<.4f} ms | Torch Time: {torch_time:<.4f} ms\n")

    print("My softmax_v1 comparison:")
    print("Max abs diff:", (abs_diff:=(v1_out - torch_out).abs()).max().item())
    print("Mean abs diff:", abs_diff.mean().item())
    print(f"v1 Time: {v1_time:<.4f} ms | Torch Time: {torch_time:<.4f} ms")
