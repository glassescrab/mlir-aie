#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import numpy as np
from ml_dtypes import bfloat16
import argparse

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence


def my_matmul(dev, N, K):
 
    # GEVM: A is vector (K,), B is matrix (K, N), C is vector (N,)
    # We treat this as M=1, so each output is a scalar per tile
    # n is the tile size for the output dimension (number of output elements per core)
    m = 1  # Always 1 for GEVM (vector output, not matrix)
    k = 64 # Reduction dimension tile size
    n = 64 # Output dimension tile size
    new_n = n + 2  # Add 2 for scaling factors in packed int8 weights
    # nxk = 1024  # Tile block size for B transformation, assume B is col-major KxN
    # nxk_div_n = nxk // n

    n_cols = 8
    n_rows = 4

    # GEVM dimensions
    A_sz = K
    # B is packed with int8 weights + scaling factors (2 extra elements per n-sized tile)
    B_sz = K * N + K * N // n * 2
    C_sz = N 
    C_sz_div_n_cols = C_sz // n_cols 

    N_div_n = N // n
    N_div_n_div_n_cols = N // (n * n_cols) 
    K_div_k = K // k

    n_x_k = n * k
    n_x_K = n * K

    A_taps = []
    B_taps = []
    C_taps = []
    vectorized = True

    dtype_in = bfloat16
    dtype_in_str = "bf16"
    dtype_in_B = np.int8
    dtype_in_B_str = "i8"
    dtype_out = bfloat16
    dtype_out_str = "bf16"
    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1_4col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # GEVM: A is vector (bf16), B is matrix (int8 packed with scaling factors)
            inA_ty = np.ndarray[(k,), np.dtype[dtype_in]]
            inB_ty = np.ndarray[(k * new_n * n_rows,), np.dtype[dtype_in_B]]
            MemC_ty = np.ndarray[(n * n_rows,), np.dtype[dtype_out]]
            outC_ty = np.ndarray[(n,), np.dtype[dtype_out]]
            B_ty = np.ndarray[(k, new_n), np.dtype[dtype_in_B]]
            # AIE Core Function declarations
            func_type = "vectorized" if vectorized else "scalar"

            zero = external_func(f"zero_{func_type}_{dtype_out_str}", inputs=[outC_ty])
            # For GEVM with mixed precision: vector A (bf16), matrix B (int8), output C (bf16)
            vecmat = external_func(
                f"vecmat_{func_type}_{dtype_in_str}_{dtype_out_str}",
                inputs=[inA_ty, B_ty, outC_ty],
            )

            # Tile declarations
            ShimTiles = [tile(i,0) for i in range(n_cols)]
            MemTiles = [tile(i,1) for i in range(n_cols)]
            memA_fifos = [None] * n_cols
            inA_fifos = [None] * n_cols  # 1D: vector A is broadcast to all tiles
            outC_fifos = [[None] * n_cols for _ in range(n_rows)]
            inB_fifos = [[None] * n_cols for _ in range(n_rows)]  # 2D: matrix B is distributed
            memB_fifos = [None] * n_cols
            memC_fifos = []
            cores = [
                [tile(col, row) for col in range(0, n_cols)] for row in range(2, 2 + n_rows)
            ]
            # AIE-array data movement with object fifos
            # Input A (vector - broadcast to all cores)
            for col in range(n_cols):
                memA_fifos[col] = object_fifo(
                    f"mem_A{col}",
                    ShimTiles[col],
                    MemTiles[col],
                    2,
                    inA_ty,
                )
            for col in range(n_cols):
                inA_fifos[col] = object_fifo(
                    f"inA{col}",
                    MemTiles[col],
                    [cores[j][col] for j in range(n_rows)],
                    2,
                    inA_ty,
                )
                
                object_fifo_link(memA_fifos[col], inA_fifos[col])

                # Output C
                memC_fifos.append(
                    object_fifo(f"memC{col}", MemTiles[col], ShimTiles[col], 2, MemC_ty) 
                )
                for row in range(n_rows):
                    outC_fifos[row][col] = object_fifo(
                        f"outC{row}{col}", 
                        cores[row][col], 
                        MemTiles[col], 
                        2, 
                        outC_ty,
                    )
                    
                object_fifo_link(
                    [outC_fifos[row][col] for row in range(n_rows)],
                    memC_fifos[col], 
                    [n*j for j in range(n_rows)],
                )

                # Input B (matrix - packed int8 with scaling factors, no transformation needed)
                memB_fifos[col] = object_fifo(
                        f"memB{col}", 
                        ShimTiles[col], 
                        MemTiles[col], 
                        2, 
                        inB_ty,
                        # None,
                        # # S2MM in MemTile to convert new_n * nxk block from row-maj
                        # # to tiled k*new_n blocks (transposed at 32-bit word granularity)
                        # [
                        #     [
                        #         (k, new_n),
                        #         (nxk // n, k * new_n),
                        #         (new_n, 1),
                        #     ]
                        # ],
                    )
                
                for row in range(n_rows):
                    inB_fifos[row][col] = object_fifo(
                        f"inB{row}{col}", 
                        MemTiles[col], 
                        cores[row][col], 
                        2, 
                        B_ty,
                        # MM2S pattern to read the transformed B tiles with new_n (includes scaling factors)
                        # [
                        #     (nxk_div_n, k * new_n),
                        #     (new_n // 2, 2),
                        #     (k, new_n),
                        #     (2, 1),
                        # ],
                    )
                
                # Link memB to inB fifos for this column
                object_fifo_link(
                    memB_fifos[col], 
                    [inB_fifos[row][col] for row in range(n_rows)],
                    [],
                    [new_n * k * j for j in range(n_rows)],
                )

            # Set up compute tiles
            for row in range(n_rows):
                for col in range(n_cols):
                # Compute tile col
                    @core(cores[row][col], f"vm_{n}x{k}.o")
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            elem_out = outC_fifos[row][col].acquire(
                                ObjectFifoPort.Produce, 1,
                            )
                            zero(elem_out)
                            
                            for _ in range_(K_div_k):
                                elem_in_a = inA_fifos[col].acquire(ObjectFifoPort.Consume, 1)
                                elem_in_b = inB_fifos[row][col].acquire(ObjectFifoPort.Consume, 1)
                                vecmat(elem_in_a, elem_in_b, elem_out)
                                inA_fifos[col].release(ObjectFifoPort.Consume, 1)
                                inB_fifos[row][col].release(ObjectFifoPort.Consume, 1)

                            outC_fifos[row][col].release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in_B]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
            )
            def sequence(A, B, C):
                # GEVM: A is vector (K,) bf16, B is packed int8 matrix (K, N) with scaling factors, C is vector (N,) bf16
                # B matrix is organized as 64x64 tiles with scaling factors
                # Each tile: 64x64 int8 weights + 64 bf16 scale factors = 4096 + 128 = 4224 bytes
                num_iter = N_div_n_div_n_cols // n_rows // 2
                tile_size_bytes = k * n + k * 2  # 64*64 + 64*2 = 4224 bytes per tile
                
                for j in range(num_iter):
                    for pingpong in [0,1]:
                        bd_id_base = 8 * pingpong
                        for col in range(n_cols):
                            # Transfer vector A (broadcast to all columns)
                            npu_dma_memcpy_nd(
                                metadata=memA_fifos[col],
                                bd_id=bd_id_base + 2,
                                mem=A,
                                sizes=[1, 1, 1, K],
                                strides=[0, 0, 0, 1],
                            )
                            B_base_offset = (j * K * N // num_iter + pingpong * K * N // num_iter // 2) // n * new_n
                            B_col_offset = col * K * new_n * n_rows
                            B_sizes = [K // k, n_rows , new_n, k]
                            B_strides = [k * new_n, K * new_n, k, 1]
                            B_offset = B_base_offset + B_col_offset
                            npu_dma_memcpy_nd(
                                metadata=memB_fifos[col],
                                bd_id=bd_id_base + 1,
                                mem=B,
                                offsets=[0, 0, 0, B_offset],
                                sizes=B_sizes,
                                strides=B_strides,
                            )
                            
                            # Transfer output C
                            C_base_offset = j * N // num_iter + pingpong * N // num_iter // 2
                            C_col_offset = col * N // num_iter // 2 // n_cols
                            C_offset = C_base_offset + C_col_offset
                            C_size0 = N // num_iter // 2 // n_cols
                            npu_dma_memcpy_nd(
                                metadata=memC_fifos[col],
                                bd_id=bd_id_base,
                                mem=C,
                                offsets=[0, 0, 0, C_offset],
                                sizes=[1, 1, 1, C_size0],
                                strides=[0, 0, 0, 1],
                            )
                        if j > 0 or (j == 0 and pingpong > 0):
                            dma_wait(*memC_fifos)
                dma_wait(*memC_fifos)
    print(ctx.module)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE Vector Matrix Multiplication MLIR Design (GEVM)",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-N", type=int, default=4096)
    argparser.add_argument("-K", type=int, default=4096)
    args, _ = argparser.parse_known_args()  # <- ignore the rest args in makefile-common
    dev = args.dev
    N = args.N
    K = args.K
    my_matmul(dev, N, K)
                                                                                                       