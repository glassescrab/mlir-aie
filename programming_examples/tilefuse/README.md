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
  configured. Setting `use_chess=0` selects Peano but is untested for these
  designs.

## Layout

| Directory | Precision | Kernel | Paper |
|---|---|---|---|
| [`gemm_w4a16/`](./gemm_w4a16) | AWQ `W4A16`, group size 128 | [`mix_int4_ATB.cc`](../../aie_kernels/aie2p/mix_int4_ATB.cc) | §3.2, §3.3 |
| [`gemm_w8a16/`](./gemm_w8a16) | `W8A16`, per-output-channel | [`mix_int8_ATB.cc`](../../aie_kernels/aie2p/mix_int8_ATB.cc) | §3.3 |
| [`gemv_w4a16/`](./gemv_w4a16) | AWQ `W4A16` | [`vm_mix_int4_64x8.cc`](../../aie_kernels/aie2/vm_mix_int4_64x8.cc) | §3.3, §3.4 |
| [`gemv_w8a16/`](./gemv_w8a16) | `W8A16` | [`vm_mix_int8_r16.cc`](../../aie_kernels/aie2/vm_mix_int8_r16.cc) | §3.3, §3.4 |
| [`ablation/`](./ablation) | — | — | §4.5, Fig. 9 & 10 |

Each example is one directory:

```
gemm_w4a16/
├── test.cpp          host: weight quantization, pre-tiling, reference check
├── common.h          host helpers (shared with the subdirectory)
├── makefile-common   build configuration
├── CMakeLists.txt
└── whole_array/      the design itself
    ├── whole_array.py    IRON design: dataflow, tile shapes, runtime sequence
    ├── Makefile          dimensions and kernel selection
    └── test.cpp          thin wrapper including ../test.cpp
```

GEMV examples use `vector_matrix/` in place of `whole_array/`.

## Quick start

Build and run the W4A16 GEMM at its default shape (M=4096, K=32768, N=4096):

```bash
cd programming_examples/tilefuse/gemm_w4a16/whole_array
NPU2=1 make
NPU2=1 make run
```

`NPU2=1` selects the XDNA2 target and is required. Override the shape on the
command line:

```bash
NPU2=1 M=4096 K=14336 N=4096 make
NPU2=1 M=4096 K=14336 N=4096 make run
```

The host program quantizes a random weight matrix, applies the pre-tiling
layout, runs the kernel, and verifies against a CPU reference. `make run`
defaults to 32 warmup and 64 timed iterations and prints the average NPU time.

**Constraints on the shape.** With the default `m/k/n = 64/128/64` on 8 columns:
`M` must be a multiple of `m * n_aie_rows` = 256, `N` a multiple of
`n * n_aie_cols` = 512, and `K` a multiple of both `k` = 128 and `mtk`, the K
blocking factor set in `whole_array.py` (128 in `gemm_w4a16` and `gemm_w8a16`,
512 in the ablation's `gemm_w4a16`/`gemm_w8a16`). The interleaved pre-tiling
layout supports `K` and `N` up to 32K.

## How the weight layout works

Weights are pre-tiled **offline, on the host** (`test.cpp`); there is no runtime
transformation cost. Two things happen:

1. **Metadata is packed with the weights.** Each 128×64 weight tile is followed
   in memory by its own quantization metadata, so a compute core gets everything
   it needs from a single stream. For `W4A16` a tile is 4096 B of packed INT4
   plus 128 B of BF16 scales plus 128 B of INT8 zero-points (duplicated to keep
   the payload 128-byte aligned for the DMA) = 4352 B, so the B buffer is shaped
   `128 × new_n` with `new_n = 34`. For `W8A16` a tile is 8192 B of INT8 plus
   256 B of scales = 8448 B, giving `new_n = n + 2 = 66`.

2. **Tiles are stored in interleaved column-major order.** The host emits tile
   columns `0, 8, 16, …` then `1, 9, 17, …`, matching the order the eight AIE
   columns consume them. This removes the large DMA stride that otherwise limits
   a single buffer descriptor, and is what lets `K` and `N` reach 32K.

Both scale schemes are **per output channel**: the scale for a weight depends on
its column (the `N` index), not its reduction row. In `W4A16` it additionally
varies per group of 128 reduction rows, which is exactly AWQ with group size 128.

## Known limitations

- **The `W8A16` scale axis was corrected after the paper was measured.** Earlier
  revisions applied the INT8 scale along the reduction (`K`) axis rather than the
  output-channel (`N`) axis, which cannot represent a real per-channel
  checkpoint. The fix changes neither the byte layout nor the instruction count,
  so the reported throughput should be unaffected — but the `W8A16` numbers have
  not been re-measured on hardware since.
- **`n = 64` is assumed by the host.** The `test.cpp` files hardcode `n = 64`
  when packing weights and metadata. The microkernels now derive their metadata
  offsets from the tile shape, so the assumption is confined to the host.
- **`DIV` is defined twice** — in the microkernel and again in `whole_array.py`.
  The two must agree.
- **The host over-allocates the W4A16 weight buffer.** `B_SIZE` reserves 4608 B
  per tile against the 4352 B the layout actually uses. Harmless, but the tail
  is never read.
- **No sweep harness is included.** The per-shape sweeps behind Figures 5, 9 and
  10 were driven by scripts that are not part of this tree.
