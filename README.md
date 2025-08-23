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
  * Define `pid`
  * Using `pid` and `tl.arange` of `block_size`, get `range` for `tl.load` to get the input tensor using input pointer
  * You have the load tensor, perform operations on it
  * Store the output tensor using `tl.store` in the output pointer


---
<details>
  <summary>Notes</summary>
  
# Trash/Notes
  
* Problem 7/G: Long Sum `G_sum_dim1.py`  
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/3e45f097-6e65-4df9-8808-f2515ed1174e" />

</details>
