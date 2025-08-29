# Triton Programming
* <details>
  <summary>Installation</summary>
   
  ```python
  git clone https://github.com/VachanVY/gpu-programming.git
  cd gpu-programming
  
  uv sync
  # or
  uv sync --locked # If you want them to install exactly what’s in uv.lock (no resolver changes):
  ```
   </details>


* General Structure of Triton Program
  * Define `pid` (program id)
  * Using `pid` and `tl.arange` of `block_size`, get `range`/`stride`/`indices` for `tl.load` to get the part of the input tensor using the input pointer
  * Now that you have the loaded tensor, perform operations on it
  * Store the output tensor using `tl.store` in the output pointer

```python
threadIdx.x in CUDA ≈ entries of tl.arange in Triton.
blockIdx.x in CUDA ≈ pid in Triton.

Think of tl.arange as “all the thread IDs in this block at once, in a vector”.
```

* Why blocks?

  <img width="701" height="746" alt="image" src="https://github.com/user-attachments/assets/ce55ab92-69e1-4eb7-bfde-42555feac089" />


---
<details>
  <summary>Notes</summary>
  
# Trash/Notes
  
* Problem 7/G: Long Sum `G_sum_dim1.py`  
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/3e45f097-6e65-4df9-8808-f2515ed1174e" />
* <img width="1829" height="917" alt="image" src="https://github.com/user-attachments/assets/0bae59b3-89a5-4050-ad93-2fecb307103f" />
  <img width="1829" height="917" alt="image" src="https://github.com/user-attachments/assets/7374d3db-0bb3-422b-8e97-8fe4869a54df" />
  
  [yt video](https://youtu.be/zy8ChVd_oTM?t=1627)
  
  <img width="1805" height="1100" alt="image" src="https://github.com/user-attachments/assets/eb9a370f-be34-4385-92f4-5b61ce2c8ea5" />
  <img width="1811" height="1168" alt="image" src="https://github.com/user-attachments/assets/c425b154-2f69-4e2b-a542-6dfce8a742f5" />
  <img width="1811" height="1168" alt="image" src="https://github.com/user-attachments/assets/cba10ebd-9955-46b3-956f-6239f7f864f7" />
  <img width="771" height="657" alt="image" src="https://github.com/user-attachments/assets/1ecbadf4-f22f-43f4-910d-d9c0bda52c93" />
  <img width="767" height="399" alt="image" src="https://github.com/user-attachments/assets/fab9772d-a3e0-4abd-b3ce-69b9f466d87e" />
* Matmul 11
  <img width="1600" height="817" alt="image" src="https://github.com/user-attachments/assets/ab12d199-fee5-43d2-a9f7-69d98afcb7de" />
  <img width="701" height="746" alt="image" src="https://github.com/user-attachments/assets/ce55ab92-69e1-4eb7-bfde-42555feac089" />


</details>
