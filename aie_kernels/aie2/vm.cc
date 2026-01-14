//===- mv.cc ----------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in, typename T_out, int N, int K>
void vecmat_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  // GEVM: a is vector (K,), b is matrix (K, N), c is vector (N,)
  // M=1 (always), N is output dimension, K is reduction dimension
  for (int col = 0; col < N; col++) {
    T_out runningSum = 0;
    for (int i = 0; i < K; i++) {
      runningSum += a[i] * b[i * N + col];
    }
    c[col] += runningSum;
  }
  event1();
}

template <typename T_in, typename T_out, typename T_acc, unsigned n, unsigned k,
          unsigned r, unsigned s>
void vecmat_vectorized(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(n % r == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 for vectorization
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value ||
                std::is_same<T_in, int16_t>::value);


  event0();

  T_out *__restrict c_ptr = c;
  
  AIE_LOOP_MIN_ITERATION_COUNT(n / r)
  for (int col = 0; col < n; col += r) {
    // Initialize accumulator with current values for this block of r columns
    aie::accum<T_acc, r> c_acc;
    c_acc.from_vector(aie::load_v<r>(c_ptr));
    
    T_in *__restrict b_ptr = b + col * 2; // Start at the correct column block in transposed b
    T_in *__restrict a_ptr = a;
    
    // Inner loop: iterate through k dimension in steps of 8
    for (int row = 0; row < k; row += s) {
      aie::vector<T_in, s> a_vec = aie::load_v<s>(a_ptr);
      
      const aie::vector<T_in, 2 * r> b_vec_0 = 
          aie::load_v<2 * r>(b_ptr);
      const aie::vector<T_in, 2 * r> b_vec_1 =
          aie::load_v<2 * r>(b_ptr + 2 * n);
      const aie::vector<T_in, 2 * r> b_vec_2 =
          aie::load_v<2 * r>(b_ptr + 4 * n);
      const aie::vector<T_in, 2 * r> b_vec_3 =
          aie::load_v<2 * r>(b_ptr + 6 * n);

      // Extract interleaved columns of b (now b is the matrix)
      const aie::vector<T_in, r> b_vec_0_0 = aie::filter_even(b_vec_0);
      const aie::vector<T_in, r> b_vec_0_1 = aie::filter_odd(b_vec_0);
      const aie::vector<T_in, r> b_vec_1_0 = aie::filter_even(b_vec_1);
      const aie::vector<T_in, r> b_vec_1_1 = aie::filter_odd(b_vec_1);
      const aie::vector<T_in, r> b_vec_2_0 = aie::filter_even(b_vec_2);
      const aie::vector<T_in, r> b_vec_2_1 = aie::filter_odd(b_vec_2);
      const aie::vector<T_in, r> b_vec_3_0 = aie::filter_even(b_vec_3);
      const aie::vector<T_in, r> b_vec_3_1 = aie::filter_odd(b_vec_3);

      // Accumulate: for each output element i (column col+i):
      // c_acc[i] = c_acc[i] + a_vec[0]*b_vec_0_0[i]
      //                     + a_vec[1]*b_vec_0_1[i]
      //                     + ...
      //                     + a_vec[7]*b_vec_3_1[i]
      // This computes the dot product of vector a with column (col+i) of matrix b
      c_acc = aie::accumulate<r>(
          c_acc, a_vec, 0, b_vec_0_0, b_vec_0_1, b_vec_1_0, b_vec_1_1,
          b_vec_2_0, b_vec_2_1, b_vec_3_0, b_vec_3_1);

      b_ptr += s * n; // Move to next 8 rows of b
      a_ptr += s;     // Move to next s (==8) elements of a
    }
    
    // After accumulating over all k, convert to output type and store
    aie::store_v(c_ptr, c_acc.template to_vector<T_out>());
    c_ptr += r; // Move to next r columns of output
  }

  event1();
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M and DIM_K at compile time using -DDIM_M 16 etc.
// For GEVM: DIM_M represents the output dimension tile size (n)
//           DIM_K represents the reduction dimension tile size (k)
// These dimensions must be divisible by the r, s dimensions used in
// the kernels.

#ifndef DIM_N
#define DIM_N 64  // Output dimension tile size (n)
#endif

#ifndef DIM_K
#define DIM_K 64  // Reduction dimension tile size (k)
#endif

#define combos(X)                                                              \
  X(bfloat16, bf16, bfloat16, bf16, accfloat)                                  \
  //X(int16, i16, int32, i32, acc32)

#define vecmat_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             ctype_acc)                                        \
  void vecmat_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    vecmat_scalar<ctype_in, ctype_out, DIM_N, DIM_K>(a_in, b_in, c_out);       \
  }

#define vecmat_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, ctype_acc)                     \
  void vecmat_vectorized_##mlir_type_in##_##mlir_type_out(                     \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    vecmat_vectorized<ctype_in, ctype_out, ctype_acc, DIM_N, DIM_K, 32, 8>(    \
        a_in, b_in, c_out);                                                    \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, ctype_acc)                       \
  void zero_vectorized_##mlir_type_out(ctype_out *c_out) {                     \
    zero_vectorized<ctype_out, DIM_N, 1>(c_out);                               \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           ctype_acc)                                          \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_N, 1>(c_out);                                   \
  }

combos(vecmat_scalar_c_func) combos(vecmat_vectorized_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"
