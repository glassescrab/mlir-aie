# TileFuse ablation study

These directories isolate the contribution of each TileFuse optimization. They
correspond to Figures 9 (GEMM) and 10 (GEMV) and to §4.5 of the paper.

Each point is a normal TileFuse example — see the
[parent README](../README.md) for build and run instructions. Every directory
below is a complete, independently buildable design; they differ only in the
microkernel they compile and in a small number of constants in the IRON design
file.

## GEMM (Figure 9)

| Directory | Microkernel | What it adds |
|---|---|---|
| [`gemm_baseline/`](./gemm_baseline) | [`mm_optimized.cc`](../../../aie_kernels/aie2p/mm_optimized.cc) | BF16 GEMM, default tiled layout |
| [`gemm_pretiled/`](./gemm_pretiled) | `mm_optimized.cc` | + interleaved pre-tiling (§3.2) |
| [`gemm_w4a16/`](./gemm_w4a16) | [`mix_int4_ATB.cc`](../../../aie_kernels/aie2p/mix_int4_ATB.cc) | + fused W4A16 unpack/dequant (§3.3) |
| [`gemm_w8a16/`](./gemm_w8a16) | [`mix_int8_ATB.cc`](../../../aie_kernels/aie2p/mix_int8_ATB.cc) | + fused W8A16 dequant (§3.3) |

`gemm_baseline` and `gemm_pretiled` differ only in `whole_array/whole_array.py`
(the data movement). `gemm_w4a16` and `gemm_w8a16` differ only in precision:
the microkernel, the packed tile width (`new_n` = 34 vs 66) and the host
quantizer. All four use `m/k/n = 64/128/64` on 8 columns.

The baseline configurations cannot express the DMA stride required for the
largest shapes, which is why Figure 9 omits them at 8K×8K×16K and 8K×8K×32K.

## GEMV (Figure 10)

Two independent axes, crossed: the **microkernel** (how many output elements are
accumulated per iteration, §3.3) and the **dataflow** (how many of the four AIE
rows receive weights, §3.4). `n_rows` is set in `vector_matrix/vector_matrix.py`.

| Directory | Microkernel | Weights per iter. | `n_rows` |
|---|---|---|---|
| [`gemv_baseline/`](./gemv_baseline) | [`vm.cc`](../../../aie_kernels/aie2/vm.cc) (BF16) | 16×8 | 1 |
| [`gemv_pretiled/`](./gemv_pretiled) | `vm.cc` | 16×8 | 1 |
| [`gemv_microkernel/`](./gemv_microkernel) | `vm.cc`, `k=64` | 16×8 | 1 |
| [`gemv_dataflow/`](./gemv_dataflow) | `vm.cc` | 16×8 | 4 |
| [`gemv_w4a16/`](./gemv_w4a16) | [`vm_mix_int4_16x8.cc`](../../../aie_kernels/aie2/vm_mix_int4_16x8.cc) | 16×8 | 1 |
| [`gemv_w4a16_microkernel/`](./gemv_w4a16_microkernel) | [`vm_mix_int4_64x8.cc`](../../../aie_kernels/aie2/vm_mix_int4_64x8.cc) | 64×8 | 1 |
| [`gemv_w4a16_dataflow/`](./gemv_w4a16_dataflow) | `vm_mix_int4_16x8.cc` | 16×8 | 4 |
| [`gemv_w4a16_microkernel_dataflow/`](./gemv_w4a16_microkernel_dataflow) | `vm_mix_int4_64x8.cc` | 64×8 | 4 |

`n_rows = 1` is the baseline dataflow of Figure 4a, which uses only the first
compute row. `n_rows = 4` is the optimized dataflow of Figure 4b, where each
shim core streams a bundle of four weight tiles into its memory core, which then
distributes one tile to each compute row — using all 32 cores.

## Running a comparison

**The default `M`/`K`/`N` differ between directories** — they were left at
whatever shape was last measured. Always pass the shape explicitly so every
point is compared at the same size:

```bash
for d in gemv_baseline gemv_w4a16 gemv_w4a16_microkernel gemv_w4a16_microkernel_dataflow; do
  ( cd "$d/vector_matrix" \
    && NPU2=1 M=1 K=4096 N=4096 make \
    && NPU2=1 M=1 K=4096 N=4096 make run )
done
```

Figure 10 reports the 2K×2K and 4K×4K shapes; Figure 9 reports 2K×2K×2K through
8K×8K×32K.
