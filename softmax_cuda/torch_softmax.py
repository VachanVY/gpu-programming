import time
import ctypes

import torch
from torch import Tensor, nn

sync = lambda: torch.cuda.synchronize()
seed = lambda s: torch.manual_seed(s)

@torch.no_grad()
def torch_softmax(x:Tensor, s:int=0):
    seed(s)
    torch_out = torch.empty_like(x)
    sync()

    t0 = time.time()
    torch.softmax(x, dim=1, out=torch_out)
    sync()
    t1 = time.time()
    return torch_out, (t1 - t0) * 1000

@torch.no_grad()
def my_softmax(x:Tensor, fname:str, s:int=0):
    seed(s)
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
    TIMES = 100

    softmax_v0 = lambda x, s: my_softmax(x, "./libsoftmax_v0.so", s)
    softmax_v1 = lambda x, s: my_softmax(x, "./libsoftmax_v1.so", s)
    softmax_v2 = lambda x, s: my_softmax(x, "./libsoftmax_v2.so", s)
    softmax_v3 = lambda x, s: my_softmax(x, "./libsoftmax_v3.so", s)

    
    # warm up torch version
    torch_softmax(torch.randn(A, B, device="cuda", dtype=torch.float32))
    # warmup # is it needed?
    softmax_v0(torch.randn(A, B, device="cuda", dtype=torch.float32), 0)
    softmax_v1(torch.randn(A, B, device="cuda", dtype=torch.float32), 0)
    softmax_v2(torch.randn(A, B, device="cuda", dtype=torch.float32), 0)
    softmax_v3(torch.randn(A, B, device="cuda", dtype=torch.float32), 0)

    v0_time_avg, v1_time_avg, v2_time_avg, v3_time_avg = 0, 0, 0, 0
    torch_time_avg = 0
    for i in range(TIMES):
        seed(i)
        x = torch.randn(A, B, device="cuda", dtype=torch.float32)
        torch_out, torch_time = torch_softmax(x, i)
        v0_out, v0_time = softmax_v0(x, i)
        v1_out, v1_time = softmax_v1(x, i)
        v2_out, v2_time = softmax_v2(x, i)
        v3_out, v3_time = softmax_v3(x, i)

        v0_time_avg += v0_time
        v1_time_avg += v1_time
        v2_time_avg += v2_time
        v3_time_avg += v3_time

        torch_time_avg += torch_time

    torch_time_avg /= TIMES
    v0_time_avg /= TIMES
    v1_time_avg /= TIMES
    v2_time_avg /= TIMES
    v3_time_avg /= TIMES

    print("My softmax_v0 comparison:")
    print("Max abs diff:", (abs_diff:=(v0_out - torch_out).abs()).max().item())
    print("Mean abs diff:", abs_diff.mean().item())
    print(f"v0 Time: {v0_time_avg:<.4f} ms | Torch Time: {torch_time_avg:<.4f} ms\n")

    print("My softmax_v1 comparison:")
    print("Max abs diff:", (abs_diff:=(v1_out - torch_out).abs()).max().item())
    print("Mean abs diff:", abs_diff.mean().item())
    print(f"v1 Time: {v1_time_avg:<.4f} ms | Torch Time: {torch_time_avg:<.4f} ms\n")

    print("My softmax_v2 comparison:")
    print("Max abs diff:", (abs_diff:=(v2_out - torch_out).abs()).max().item())
    print("Mean abs diff:", abs_diff.mean().item())
    print(f"v2 Time: {v2_time_avg:<.4f} ms | Torch Time: {torch_time_avg:<.4f} ms\n")

    print("My softmax_v3 comparison:")
    print("Max abs diff:", (abs_diff:=(v3_out - torch_out).abs()).max().item())
    print("Mean abs diff:", abs_diff.mean().item())
    print(f"v3 Time: {v3_time_avg:<.4f} ms | Torch Time: {torch_time_avg:<.4f} ms\n")


# My softmax_v0 comparison:
# Max abs diff: 1.5366822481155396e-08
# Mean abs diff: 6.319667011922547e-11
# v0 Time: 63.8461 ms | Torch Time: 0.3026 ms

# My softmax_v1 comparison:
# Max abs diff: 2.1187588572502136e-08
# Mean abs diff: 6.52178797078129e-11
# v1 Time: 51.8443 ms | Torch Time: 0.3026 ms

# My softmax_v2 comparison:
# Max abs diff: 4.656612873077393e-10
# Mean abs diff: 8.447319449836344e-13
# v2 Time: 0.3019 ms | Torch Time: 0.3026 ms

# My softmax_v3 comparison:
# Max abs diff: 4.656612873077393e-10
# Mean abs diff: 8.56806325107845e-13
# v3 Time: 0.3049 ms | Torch Time: 0.3026 ms