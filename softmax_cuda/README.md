# Compilation and Run
* ```python
  nvcc -arch=sm_86 -Xcompiler -fPIC -shared softmax_v0.cu -o libsoftmax_v0.so \
  && nvcc -arch=sm_86 -Xcompiler -fPIC -shared softmax_v1.cu -o libsoftmax_v1.so \
  && nvcc -arch=sm_86 -Xcompiler -fPIC -shared softmax_v2.cu -o libsoftmax_v2.so \
  && nvcc -arch=sm_86 -Xcompiler -fPIC -shared softmax_v3.cu -o libsoftmax_v3.so
  ```
* Benchmark against PyTorch
  ```python
  python torch_softmax.py
  ```

# Notes Dump
* [Shared Memory Version](https://github.com/VachanVY/gpu-programming/blob/main/softmax_cuda/softmax_v2.cu)
  <img width="1316" height="1600" alt="image" src="https://github.com/user-attachments/assets/a757b90f-7e7b-42b6-be53-53b0d72c3489" />
* [Warp Version](https://github.com/VachanVY/gpu-programming/blob/main/softmax_cuda/softmax_v3.cu)
  <img width="1047" height="1414" alt="image" src="https://github.com/user-attachments/assets/c83e668a-b61c-491b-b233-be90a1b95a13" />
