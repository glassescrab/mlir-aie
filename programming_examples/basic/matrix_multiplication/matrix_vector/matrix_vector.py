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


def my_matmul(dev, M, K):
 
    m = 64
    k = 64
    mtk = 1024
    mtk_div_k = mtk // k 

    n_cols = 8
    n_rows = 1

    A_sz = M * K
    B_sz = K
    C_sz = M 
    C_sz_div_n_cols = C_sz // n_cols 

    M_div_m = M // m
    M_div_m_div_n_cols = M // (m * n_cols) 
    K_div_k = K // k

    m_x_k = m * k
    m_x_K = m * K

    A_taps = []
    B_taps = []
    C_taps = []
    # FIXME vectorized kernel is currently erroneous
    vectorized = True

    # dtype_in = np.dtype[np.int16]
    # dtype_in_str = "i16"
    # dtype_out = np.dtype[np.int32]
    # dtype_out_str = "i32"

    dtype_in = bfloat16
    dtype_in_str = "bf16"
    dtype_out = bfloat16
    dtype_out_str = "bf16"
    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1_4col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            inA_ty = np.ndarray[(m* mtk_div_k *n_rows,k), np.dtype[dtype_in]]
            inB_ty = np.ndarray[(k,), np.dtype[dtype_in]]
            MemC_ty = np.ndarray[(m * n_rows,), np.dtype[dtype_out]]
            outC_ty = np.ndarray[(m,), np.dtype[dtype_out]]
            A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
            # AIE Core Function declarations
            func_type = "vectorized" if vectorized else "scalar"

            zero = external_func(f"zero_{func_type}_{dtype_out_str}", inputs=[outC_ty])
            matvec = external_func(
                f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
                inputs=[A_ty, inB_ty, outC_ty],
            )

            # Tile declarations
            ShimTiles = [tile(i,0) for i in range(n_cols)]
            MemTiles = [tile(i,1) for i in range(n_cols)]
            memA_fifos = [None] * n_cols
            inA_fifos = [[None] * n_cols for _ in range(n_rows)]
            outC_fifos = [[None] * n_cols for _ in range(n_rows)]
            inB_fifos = [None] * n_cols
            memB_fifos = []
            memC_fifos = []
            cores = [
                [tile(col, row) for col in range(0, n_cols)] for row in range(2, 2 + n_rows)
            ]
            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cols):
                memA_fifos[i] = object_fifo(
                    f"mem_A{i}",
                    (
                        ShimTiles[i]
                    ),  
                    MemTiles[i],
                    2,
                    inA_ty,
                    None,
                    # S2MM in MemTile to convert m * mtk block from row-maj
                    # to tiled m*k blocks
                    
                    [
                        [
                            (m, k),
                            (mtk // k, m * k),
                            (k, 1),
                        ]
                       
                    ],
            )
            for col in range(n_cols):
                for row in range(n_rows):
                    inA_fifos[row][col]= object_fifo(
                            f"inA{row}{col}",
                            MemTiles[col],
                            cores[row][col],
                            2,
                            A_ty,
                            #MM2S
                            # None,
                            
                            [   
                                
                                (mtk_div_k,m*k),
                                (k  // 2, 2),
                                (m, k),
                                (2, 1),
                            ]
                            
                        )
                
                object_fifo_link(memA_fifos[col], [inA_fifos[row][col] for row in range(n_rows)],[],[m*mtk*j for j in range(n_rows)])#

                # Output C
                memC_fifos.append(
                    object_fifo(f"memC{col}", MemTiles[col], ShimTiles[col],2, MemC_ty) 
                )
                for row in range(n_rows):
                    outC_fifos[row][col]=object_fifo(
                            f"outC{row}{col}", 
                            cores[row][col], 
                            MemTiles[col], 
                            2, 
                            outC_ty,
                            []
                        )
                    
                object_fifo_link(
                    [outC_fifos[row][col] for row in range(n_rows)],
                    memC_fifos[col], 
                    [m*j for j in range(n_rows)],
                    # [0,0,0,0],
                    []
                )

                #Input B

                memB_fifos.append(
                    object_fifo(f"memB{col}", ShimTiles[col], MemTiles[col], 2, inB_ty)
                )
                
                
                inB_fifos[col] = object_fifo(
                    f"inB{col}", MemTiles[col], [cores[j][col] for j in range(n_rows)], 2, inB_ty
                )
                
                # Link memB to all inB fifos for this column
                object_fifo_link(
                    memB_fifos[col], 
                    inB_fifos[col],
                    )#

            # Set up compute tiles
            for row in range(n_rows):
                for col in range(n_cols):
                # Compute tile col
                    @core(cores[row][col], f"mv_{m}x{k}.o")
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            elem_out = outC_fifos[row][col].acquire(
                                ObjectFifoPort.Produce,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                1,
                            )
                            zero(elem_out)
                            
                            for _ in range_(K_div_k):
                                elem_in_a = inA_fifos[row][col].acquire(ObjectFifoPort.Consume, 1)
                                elem_in_b = inB_fifos[col].acquire(ObjectFifoPort.Consume, 1)
                                matvec(elem_in_a, elem_in_b, elem_out)
                                inA_fifos[row][col].release(ObjectFifoPort.Consume, 1)
                                inB_fifos[col].release(ObjectFifoPort.Consume, 1)

                            outC_fifos[row][col].release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
            )
            def sequence(A, B, C):



                num_iter = M_div_m_div_n_cols // n_rows // 2
                for j in range(num_iter):
                    for pingpong in [0,1]:
                        bd_id_base = 8 * pingpong
                        for col in range(n_cols):
                            npu_dma_memcpy_nd(
                                metadata=memB_fifos[col],
                                bd_id=bd_id_base + 2,
                                mem=B,
                                sizes=[1, 1, 1, K],
                                strides=[0, 0, 0, 1],
                            )
                            A_base_offset =  j * K * M // num_iter + pingpong * K * M // num_iter // 2
                            A_col_offset = col * K * M // num_iter // 2 // n_cols
                            A_offset = A_base_offset + A_col_offset
                            A_sizes = [1, K//mtk, n_rows * m, mtk]
                            A_strides = [0, mtk, K, 1]
                            npu_dma_memcpy_nd(
                                metadata=memA_fifos[col],
                                bd_id=bd_id_base + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_offset],
                                sizes=A_sizes,
                                strides=A_strides,
                            )
                            C_base_offset =  j * M // num_iter + pingpong * M // num_iter // 2
                            C_col_offset = col * M // num_iter // 2 // n_cols
                            C_offset = C_base_offset + C_col_offset
                            C_size0 = C_sz_div_n_cols//num_iter//2
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
        prog="AIE Matrix Vector Multiplication MLIR Design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    args, _ = argparser.parse_known_args()  # <- ignore the rest args in makefile-common
    dev = args.dev
    M = args.M
    K = args.K
    my_matmul(dev, M, K)
                                                                                                       