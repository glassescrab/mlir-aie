# TileFuse

Fused mixed-precision GEMM/GEMV kernels for AMD XDNA2 NPUs, built on the
MLIR-AIE (IRON) flow. TileFuse consumes AWQ-style `W4A16` and LLM.int8-style
`W8A16` weights directly on the NPU, fusing unpacking, dequantization and
GEMM/GEMV into a single kernel pass.

> **TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized
> LLM Inference on AMD NPU.** Wesley Pang\*, Gregory Hyegang Jun\*, Feiyang Liu,
> Deming Chen. ICCAD '26. <https://doi.org/10.1145/3831252.3834036>

This directory is the artifact for that paper. Everything outside it is
unmodified upstream [MLIR-AIE](https://github.com/Xilinx/mlir-aie), except for
the microkernels listed below.

## Requirements

- **An XDNA2 NPU** — Ryzen AI 300-series (Strix Point or Krackan Point). The
  microkernels target `aie2p` / `npu2`; they do not run on Phoenix/Hawk (`npu1`).
- **MLIR-AIE (IRON)** built and on `PATH`, plus XRT. See the
  [top-level README](../../README.md).
- **The AMD Vitis AIE compiler (`xchesscc`).** Every `makefile-common` here sets
  `use_chess?=1`, so a stock Peano-only install will not build these kernels as
  configured.