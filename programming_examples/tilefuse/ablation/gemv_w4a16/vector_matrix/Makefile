##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

subdir=vector_matrix
targetname=vector_matrix

# Currently does not accept reconfiguring size via these variables; must change
# in source at <targetname>.py as well as here
# GEVM: M=1 (always), K is reduction dim, N is output dim
M?=1
K?=4096
N?=4096
k?=128
n?=64

kernels=vm_${n}x${k}

KERNEL_DEFINES=-DDIM_N=${n} -DDIM_K=${k}
ifeq (${b_col_maj}, 1)
	KERNEL_DEFINES+=-DB_COL_MAJ
endif

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include ${SELF_DIR}../makefile-common

# Overwrite the kernels_dir defined in makefile-common for this example; on 
# AIE2p, we will also use the AIE2 kernel -- there is no specialized Strix 
# variant.
kernels_dir=${srcdir}/../../../../../aie_kernels/aie2

build/vm_${n}x${k}.o: ${kernels_dir}/vm_mix_int4_16x8.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}