    #
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import numpy as np
import argparse
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

def my_matmul(
    dev,
    M,
    K,
    m,
    k,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str
): 

    n_aie_rows = 1 # Because of memory constraints, 1 row has the best performance
    n_cores = n_aie_rows * n_aie_cols

    A_sz = M * K
    B_sz = K
    C_sz = M
    C_sz_div_n_cores = C_sz // n_cores

    M_div_m = M // m
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    m_x_k = m * k
    m_x_K = m * K

    # FIXME vectorized kernel is currently erroneous
    vectorized = True

    dtype_in = np.dtype[dtype_map[dtype_in_str]]
    dtype_out = np.dtype[dtype_map[dtype_out_str]]

    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1_4col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            inA_ty = np.ndarray[(m * k,), dtype_in]
            inB_ty = np.ndarray[(k,), dtype_in]
            outC_ty = np.ndarray[(m,), dtype_out]
            A_ty = np.ndarray[(m, k), dtype_in]

            # AIE Core Function declarations
            func_type = "vectorized" if vectorized else "scalar"
            zero = external_func(f"zero_{func_type}_{dtype_out_str}", inputs=[outC_ty])
            matvec = external_func(
                f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
                inputs=[A_ty, inB_ty, outC_ty],
            )

            tiles = [
                [tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)
            ]
            # Tile declarations
            ShimTiles = tiles[0]
            MemTiles = tiles[1]
            cores = tiles[2]
            memA_fifos = []
            inA_fifos = []
            outC_fifos = []

            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cores):
                memA_fifos.append(
                    object_fifo(f"memA{i}", ShimTiles[i], MemTiles[i], 2, inA_ty)
                )
                inA_fifos.append(
                    object_fifo(
                        f"inA{i}",
                        MemTiles[i],
                        cores[i],
                        2,
                        A_ty,
                        (
                            [
                                (k  // 2, 2),
                                (m, k),
                                (2, 1),
                            ]
                            if vectorized
                            else []
                        ),  # transpose at 4-byte (2xbf16) granularity
                    )
                )
                object_fifo_link(memA_fifos[i], inA_fifos[i])

                # Output C
                outC_fifos.append(
                    object_fifo(f"outC{i}", cores[i], ShimTiles[i], 2, outC_ty)
                )

            # Input B
            inB_fifo = object_fifo(
                "inB", ShimTiles[1 % n_cores], cores[0:n_cores], 2, inB_ty
            )

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], f"mv_{m}x{k}.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC_fifos[i].acquire(
                            ObjectFifoPort.Produce,
                            1,
                        )
                        zero(elem_out)

                        for _ in range_(K_div_k):
                            elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = inB_fifo.acquire(ObjectFifoPort.Consume, 1)
                            matvec(elem_in_a, elem_in_b, elem_out)
                            inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                            inB_fifo.release(ObjectFifoPort.Consume, 1)

                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            @runtime_sequence(
                np.ndarray[(A_sz,), dtype_in],
                np.ndarray[(B_sz,), dtype_in],
                np.ndarray[(C_sz,), dtype_out],
            )
            def sequence(A, B, C):
                for pingpong in [0,1]:
                    bd_id_base = 8 * pingpong
                    B_size3 = M_div_m_div_n_cores // 2
                    npu_dma_memcpy_nd(
                        metadata=inB_fifo,
                        bd_id=bd_id_base,
                        mem=B,
                        sizes=[B_size3, 1, 1, K],
                        strides=[0, 0, 0, 1],
                    )                    
                    C_pingpong_offset = pingpong * M // 2 
                    A_pingpong_offset = pingpong * M *K //2
                    for i in range(n_cores):

                        C_offset = i * M//2//n_cores + C_pingpong_offset
                        C_sizes = M//n_cores// 2
                        npu_dma_memcpy_nd(
                            metadata=outC_fifos[i],
                            bd_id=bd_id_base + 1 ,
                            mem=C,
                            offsets=[0, 0, 0, C_offset],
                            sizes=[1, 1, 1, C_sizes],
                            strides=[0,0,0,1],
                        )
                        A_offset = i * M*K//2//n_cores + A_pingpong_offset
                        A_sizes = [M//m//n_cores//2,K_div_k,m,k]
                        A_strides = [ m_x_K,k,K,1]
                        npu_dma_memcpy_nd(
                            metadata=memA_fifos[i],
                            bd_id=bd_id_base +2,
                            mem=A,
                            offsets=[0, 0, 0, A_offset],
                            sizes=A_sizes,
                            strides=A_strides,
                        )
                    # dma_wait(*outC_fifos)
                dma_wait(*outC_fifos)
                dma_wait(*outC_fifos)                   

    print(ctx.module)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Vector Multiplication MLIR Design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-m", type=int, default=32)
    argparser.add_argument("-k", type=int, default=32)
    argparser.add_argument("--dtype_in", type=str, choices=["i16", "bf16"], default="i16")
    argparser.add_argument("--dtype_out", type=str, choices=["i32", "bf16", "f32"], default="i32")
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    args, _ = argparser.parse_known_args()  # <- ignore the rest args in makefile-common
    dev = args.dev
    M = args.M
    K = args.K
    m = args.m
    k = args.k
    dtype_in_str = args.dtype_in
    dtype_out_str = args.dtype_out
    n_aie_cols = args.n_aie_cols
    my_matmul(dev, M, K, m, k, n_aie_cols, dtype_in_str, dtype_out_str)