# Triton Programming
* General Structure of Triton Program
  * Define `pid`
  * Using `pid` and `tl.arange` of `block_size`, get `range` for `tl.load` to get the input tensor using input pointer
  * You have the load tensor, perform operations on it
  * Store the output tensor using `tl.store` in the output pointer
