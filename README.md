# `cuda-device-query`

Rust port of [`deviceQuery.cpp`](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp) written in Rust with [`cudarc`](https://github.com/chelsea0x3b/cudarc).

## Install

Install on either a Windows or Linux machine with at least a CUDA device via `cargo` as:

```bash
cargo install cuda-device-query
```

## Run

Then run it via the `cuda-device-query` binary, e.g., on a NVIDIA L40S looks like:

```console
$ cuda-device-query
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA L40S"
  CUDA Driver Version:                           12.4
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 45373 MBytes (47576711168 bytes)
  (142) Multiprocessors, (128) CUDA Cores/MP:    18176 CUDA Cores
  GPU Max Clock rate:                            2520 MHz (2.52 GHz)
  Memory Clock rate:                             9001 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 100663296 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072) 2D=(131072, 65536) 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768) 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768) 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z):  (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z):  (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 52 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = 12.4

Result = PASS
```
