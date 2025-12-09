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

template <typename T_in, typename T_out, int M, int K>
void matvec_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < M; row++) {
    T_out runningSum = 0;
    for (int i = 0; i < K; i++) {
      runningSum += a[row * K + i] * b[i];
    }
    c[row] += runningSum;
  }
  event1();
}

template <typename T_in, typename T_out, typename T_acc, unsigned m, unsigned k,
          unsigned r, unsigned s>
void matvec_vectorized(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(m % r == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 because that is the number of
                         // column vectors (a_vec_0_0..a_vec_3_1) we create
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value ||
                std::is_same<T_in, int16_t>::value);


  event0();

  T_out *__restrict c_ptr = c;
  
  AIE_LOOP_MIN_ITERATION_COUNT(m / r)
  for (int row = 0; row < m; row += r) {
    // Initialize accumulator with zeros for this block of r rows
    aie::accum<T_acc, r> c_acc;
    c_acc.from_vector(aie::load_v<r>(c_ptr));
    
    T_in *__restrict a_ptr = a + row * 2; // Start at the correct row block in the transposed A
    T_in *__restrict b_ptr = b;
    
    // Inner loop: iterate through k dimension in steps of 8
    for (int col = 0; col < k; col += s) {
      aie::vector<T_in, s> b_vec = aie::load_v<s>(b_ptr);
      
      const aie::vector<T_in, 2 * r> a_vec_0 = 
          aie::load_v<2 * r>(a_ptr);
      const aie::vector<T_in, 2 * r> a_vec_1 =
          aie::load_v<2 * r>(a_ptr + 2 * m);
      const aie::vector<T_in, 2 * r> a_vec_2 =
          aie::load_v<2 * r>(a_ptr + 4 * m);
      const aie::vector<T_in, 2 * r> a_vec_3 =
          aie::load_v<2 * r>(a_ptr + 6 * m);

      // The even/odd calls below extract the interleaved columns of A.
      // We need to do this since A is only transposed (column-major) at
      // a granularity of 4 bytes, but bf16 are two bytes; therefore, we
      // end up with two interleaved columns at each 2*m interval.
      // After this, each of a_vec_0_0 contains rows row..row+r of some
      // column of A. The columns are col..col+8.
      const aie::vector<T_in, r> a_vec_0_0 = aie::filter_even(a_vec_0);
      const aie::vector<T_in, r> a_vec_0_1 = aie::filter_odd(a_vec_0);
      const aie::vector<T_in, r> a_vec_1_0 = aie::filter_even(a_vec_1);
      const aie::vector<T_in, r> a_vec_1_1 = aie::filter_odd(a_vec_1);
      const aie::vector<T_in, r> a_vec_2_0 = aie::filter_even(a_vec_2);
      const aie::vector<T_in, r> a_vec_2_1 = aie::filter_odd(a_vec_2);
      const aie::vector<T_in, r> a_vec_3_0 = aie::filter_even(a_vec_3);
      const aie::vector<T_in, r> a_vec_3_1 = aie::filter_odd(a_vec_3);

      // The accumulate call below produces the following output:
      // c_acc[i] = c_acc[i] + b_vec[0]*a_vec_0_0[i]
      //                     + b_vec[1]*a_vec_0_1[i]
      //                     + ...
      //                     + b_vec[7]*a_vec_3_1[i]
      // i.e., accumulating the dot product of vector b_vec with one row (row+i)
      // (recall that the different a_vecs are columns, thus we are
      // indexing into the same row i for each column).
      // The accumulator stays in high precision throughout the entire k loop.
      c_acc = aie::accumulate<r>(
          c_acc, b_vec, 0, a_vec_0_0, a_vec_0_1, a_vec_1_0, a_vec_1_1,
          a_vec_2_0, a_vec_2_1, a_vec_3_0, a_vec_3_1);

      a_ptr += s * m; // Move to next 8 columns of A
      b_ptr += s;     // Move to next s (==8) elements of b
    }
    
    // After accumulating over all k, convert to output type and store once
    aie::store_v(c_ptr, c_acc.template to_vector<T_out>());
    c_ptr += r; // Move to next r rows of output
  }

  event1();
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M and DIM_K at compile time using -DDIM_M 16 etc.
// These dimensions must be divisible by the r, s dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#define combos(X)                                                              \
  X(bfloat16, bf16, bfloat16, bf16, accfloat)                                  \
  //X(int16, i16, int32, i32, acc32)

#define matvec_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             ctype_acc)                                        \
  void matvec_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matvec_scalar<ctype_in, ctype_out, DIM_M, DIM_K>(a_in, b_in, c_out);       \
  }

#define matvec_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, ctype_acc)                     \
  void matvec_vectorized_##mlir_type_in##_##mlir_type_out(                     \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matvec_vectorized<ctype_in, ctype_out, ctype_acc, DIM_M, DIM_K, 16, 8>(    \
        a_in, b_in, c_out);                                                    \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, ctype_acc)                       \
  void zero_vectorized_##mlir_type_out(ctype_out *c_out) {                     \
    zero_vectorized<ctype_out, DIM_M, 1>(c_out);                               \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           ctype_acc)                                          \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_M, 1>(c_out);                                   \
  }

combos(matvec_scalar_c_func) combos(matvec_vectorized_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"
