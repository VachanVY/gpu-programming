import triton, torch
from torch import Tensor
import triton.language as tl
from jaxtyping import Float32, Int32
import time
import matplotlib.pyplot as plt


autotune_configs = [
    triton.Config({'B0': B0, 'B1': B1, 'B2': 1, 'B_MID': BMID}, num_warps=num_warps, num_stages=num_stages)
    for B0 in [16, 32, 64, 128, 256]
    for B1 in [16, 32, 64, 128, 256]
    for BMID in [16, 32, 64, 128, 256]
    for num_warps in [2, 4, 8, 16]
    for num_stages in [2, 3, 4]
]
@triton.autotune(configs=autotune_configs, key=['N0', 'N1', 'MID'])
@triton.jit
def dot_kernel(
        x_ptr, # (N2,N0,MID)
        y_ptr, # (N2,MID,N1)
        z_ptr, # (N2,N0,N1)
        N0, # NUM ROWS
        N1, # NUM COLS
        N2, # NUM BATCHES
        MID,# MID DIMENSION
        B0: tl.constexpr, # BLOCK SIZE ALONG N0 (rows)
        B1: tl.constexpr, # BLOCK SIZE ALONG N1 (cols)
        B2: tl.constexpr, # BLOCK SIZE ALONG N2 (batches)
        B_MID: tl.constexpr # BLOCK SIZE ALONG MID
):
    # print(f"{N0=}, {N1=}, {N2=}, {MID=}, {B0=}, {B1=}, {B2=}, {B_MID=}")
    pid_0 = tl.program_id(0) # for col dim   # B1
    pid_1 = tl.program_id(1) # for row dim   # B0
    pid_2 = tl.program_id(2) # for batch dim # B2

    bidx    = (tl.arange(0, B2) + pid_2 * B2)[:, None, None] # (B2, 1, 1)
    row_idx = (tl.arange(0, B0) + pid_1 * B0)[None, :, None] # (1, B0, 1)
    col_idx = (tl.arange(0, B1) + pid_0 * B1)[None, None, :] # (1, 1 ,B1)

    xrange = bidx * N0 * MID  +  row_idx * MID
    yrange = bidx * MID * N1  + col_idx
    # always keep the the accumulator in higher dtype to prevent overflow
    accumulator = tl.zeros((B2, B0, B1), dtype=tl.float32) # (B2, B0, B1)
    for mid in tl.range(0, MID, B_MID): # [0:B_MID, B_MID:2*B_MID, 2*B_MID:3*B_MID, ...]
        mid_idx = (tl.arange(0, B_MID) + mid) # (B_MID,)

        # [b, r, c] => b * num_rows * num_cols  +  r * num_cols  +  c
        # xrange = bidx * N0 * MID  +  row_idx * MID  +  mid_idx[None, None, :]
        xmask = (bidx < N2) & (row_idx < N0) & (mid_idx[None, None, :] < MID)
        x = tl.load(x_ptr + (xrange + mid_idx[None, None, :]), mask=xmask) # (B2, B0, B_MID)

        # [b, r, c] => b * num_rows * num_cols  +  r * num_cols  +  c
        # yrange = bidx * MID * N1  +  mid_idx[None, :, None] * N1 + col_idx
        ymask  = (bidx < N2) & (mid_idx[None, :, None] < MID) & (col_idx < N1)
        y = tl.load(y_ptr + (yrange + mid_idx[None, :, None] * N1), mask=ymask) # (B2, B_MID, B1)

        # matmul
        accumulator += tl.dot(x, y, input_precision="ieee") # (B2, B0, B_MID) @ (B2, B_MID, B1) => (B2, B0, B1)

    # [b, r, c] => b * num_rows * num_cols  +  r * num_cols  +  c
    zrange = bidx * N0 * N1 + row_idx * N1 + col_idx
    zmask = (bidx < N2) & (row_idx < N0) & (col_idx < N1)
    tl.store(z_ptr + zrange, value=accumulator, mask=zmask)


def matmul(x: Tensor, y: Tensor) -> Tensor:
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape[0] == y.shape[0] and x.shape[2] == y.shape[1]
    N2, N0, MID = x.shape
    _, _, N1 = y.shape
    z = torch.empty((N2, N0, N1), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(N1, meta['B1']),
            triton.cdiv(N0, meta['B0']),
            triton.cdiv(N2, meta['B2']),
        )

    dot_kernel[grid](x, y, z,
                     N0, N1, N2, MID,
                     )
    return z


def benchmark(sizes, device="cuda", num_iters=50):
    triton_times, torch_times = [], []

    for N in sizes:
        x = torch.randn(1, N, N, device=device, dtype=torch.float32)
        y = torch.randn(1, N, N, device=device, dtype=torch.float32)

        # Warmup
        _ = matmul(x, y)
        _ = torch.matmul(x, y)

        # Triton timing
        t0 = time.time()
        for _ in range(num_iters):
            z1 = matmul(x, y)
        torch.cuda.synchronize()
        triton_t = (time.time() - t0) / num_iters * 1000

        # Torch timing
        t0 = time.time()
        for _ in range(num_iters):
            z2 = torch.matmul(x, y)
        torch.cuda.synchronize()
        torch_t = (time.time() - t0) / num_iters * 1000

        print(f"N={N} Triton: {triton_t:.3f}ms | Torch: {torch_t:.3f}ms")
        print(f"mean diff: {torch.abs(z1 - z2).mean():e}, max diff: {torch.abs(z1 - z2).max():e}\n\n")
        assert torch.allclose(z1, z2, atol=1e-2, rtol=1e-2), f"Mismatch for {N}"
        triton_times.append(triton_t)
        torch_times.append(torch_t)
    return triton_times, torch_times


if __name__ == "__main__":
    sizes = [64, 128, 256, 512, 1024, 2048]
    triton_times, torch_times = benchmark(sizes)

    plt.plot(sizes, triton_times, marker='o', label="Triton (autotuned)")
    plt.plot(sizes, torch_times, marker='s', label="Torch matmul")
    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("Time (ms)")
    plt.xticks(sizes)
    plt.title("Batched Matmul Performance: Triton vs Torch")
    plt.legend()
    plt.grid(True)
    plt.savefig("torch_vs_triton_matmul.png")
    plt.show()

# N=64 Triton: 0.021ms | Torch: 0.009ms
# mean diff: 5.897076e-07, max diff: 5.722046e-06


# N=128 Triton: 0.019ms | Torch: 0.007ms
# mean diff: 1.819344e-06, max diff: 2.288818e-05


# N=256 Triton: 0.019ms | Torch: 0.008ms
# mean diff: 3.412233e-06, max diff: 4.196167e-05


# N=512 Triton: 0.019ms | Torch: 0.011ms
# mean diff: 7.226446e-06, max diff: 1.449585e-04


# N=1024 Triton: 0.057ms | Torch: 0.046ms
# mean diff: 0.000000e+00, max diff: 0.000000e+00


# N=2048 Triton: 0.344ms | Torch: 0.319ms
# mean diff: 2.695394e-05, max diff: 5.035400e-04