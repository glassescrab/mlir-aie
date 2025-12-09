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

template <typename T_in_A, typename T_in_B, typename T_out, int N, int K>
void vecmat_scalar(T_in_A *a, T_in_B *b, T_out *c) {
  event0();
  // GEVM: a is vector (K,), b is matrix (K, N), c is vector (N,)
  // M=1 (always), N is output dimension, K is reduction dimension
  // Note: For mixed precision, b should already be dequantized to T_in_A type
  for (int col = 0; col < N; col++) {
    T_out runningSum = 0;
    for (int i = 0; i < K; i++) {
      runningSum += a[i] * b[i * N + col];
    }
    c[col] += runningSum;
  }
  event1();
}

template <typename T_in_A, typename T_in_B, typename T_out, typename T_acc, unsigned n, unsigned k,
          unsigned r, unsigned s>
void vecmat_vectorized(T_in_A *__restrict a, T_in_B *__restrict b,
                       T_out *__restrict c) {
  static_assert(n % r == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 for vectorization
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in_A, bfloat16>::value ||
                std::is_same<T_in_A, int16_t>::value);


  event0();

  const T_in_B *__restrict pB_quantized = b;
  const T_in_B *__restrict pBs_b = b + n * k;

  const unsigned colB = n / 8;      // n / 8 (number of column tiles in 8x8)
  const unsigned size_B = 8 * 8;    // 64 (elements per 8x8 tile)
  const unsigned half_r = r / 2;    // 8

// for 8x16 large tile


//   const unsigned colA = k / s;      // k / 8 (number of row tiles)
//   const unsigned colB = n / r;      // n / 16 (number of column tiles)
//   const unsigned size_B = s * r;    // 8 * 16 = 128 (elements per small tile)
//   const unsigned half_r = r / 2;    // 8
//   for (unsigned i = 0; i < colA; ++i) chess_prepare_for_pipelining chess_loop_range(4, ) {
//     for (unsigned j = 0; j < colB; ++j) chess_flatten_loop {
//       // Process left half of 8x16 tile (8x8 block)
//       {
//         aie::vector<T_in_B, 64> B_quantized_left;
//         B_quantized_left = aie::load_v<64>(pB_quantized + (i * colB + j) * size_B);
        
//         aie::vector<T_in_B, 16> Bs_row = aie::load_v<16>(pBs_b + i * 16);
//         aie::vector<T_in_A, 8> Bs_row_cast = aie::vector_cast<T_in_A>(Bs_row);
//         aie::vector<T_in_A, 64> Bs_left = aie::transpose(Bs_row_cast.template grow_replicate<64>(), 8, 8);
        
//         aie::vector<T_in_A, 64> B_dequantized_left = aie::to_float<T_in_A>(B_quantized_left);
//         auto B_scaled_left = aie::mul(B_dequantized_left, Bs_left);
//         aie::vector<T_in_A, 64> B_final_left = aie::to_vector<T_in_A>(B_scaled_left);
        
//         aie::store_v(b_buf + (i * colB + j) * size_B, B_final_left);
//       }
      
//       // Process right half of 8x16 tile (8x8 block)
//       {
//         aie::vector<T_in_B, 64> B_quantized_right;
//         B_quantized_right = aie::load_v<64>(pB_quantized + (i * colB + j) * size_B + 64);
        
//         aie::vector<T_in_B, 16> Bs_row = aie::load_v<16>(pBs_b + i * 16);
//         aie::vector<T_in_A, 8> Bs_row_cast = aie::vector_cast<T_in_A>(Bs_row);
//         aie::vector<T_in_A, 64> Bs_right = aie::transpose(Bs_row_cast.template grow_replicate<64>(), 8, 8);
        
//         aie::vector<T_in_A, 64> B_dequantized_right = aie::to_float<T_in_A>(B_quantized_right);
//         auto B_scaled_right = aie::mul(B_dequantized_right, Bs_right);
//         aie::vector<T_in_A, 64> B_final_right = aie::to_vector<T_in_A>(B_scaled_right);
        
//         aie::store_v(b_buf + (i * colB + j) * size_B + 64, B_final_right);
//       }
//     }
//   }

  T_out *__restrict c_ptr = c;

  AIE_LOOP_MIN_ITERATION_COUNT(n / r)
  for (int col = 0; col < n; col += r) {
    // Initialize accumulator with current values for this block of r columns
    aie::accum<T_acc, r> c_acc;
    c_acc.from_vector(aie::load_v<r>(c_ptr));
    
    T_in_A *__restrict a_ptr = a;
    
    // Iterate through k dimension in steps of s (8 rows at a time)
    for (int row = 0; row < k; row += s) chess_prepare_for_pipelining chess_loop_range(8, ) {
      const aie::vector<T_in_A, s> a_vec = aie::load_v<s>(a_ptr);
      
      // Calculate tile indices
      const int row_tile = row / 8;
      const int col_tile_left = col / 8;
      const int col_tile_right = col_tile_left + 1;
      
      // Load both quantized tiles
      const aie::vector<T_in_B, size_B> B_quantized_left = 
        aie::load_v<size_B>(pB_quantized + (row_tile * colB + col_tile_left) * size_B);
      const aie::vector<T_in_B, size_B> B_quantized_right = 
        aie::load_v<size_B>(pB_quantized + (row_tile * colB + col_tile_right) * size_B);
      
      // Load and prepare scale factors once for both tiles (they share the same row scales)
      const aie::vector<T_in_B, 16> Bs_row = aie::load_v<16>(pBs_b + row_tile * 16);
      const aie::vector<T_in_A, 8> Bs_row_cast = aie::vector_cast<T_in_A>(Bs_row);
      const aie::vector<T_in_A, size_B> Bs_replicated = aie::transpose(Bs_row_cast.template grow_replicate<size_B>(), 8, 8);
      
      // Dequantize both tiles in parallel
      const aie::vector<T_in_A, size_B> B_dequant_left = aie::to_float<T_in_A>(B_quantized_left);
      const aie::vector<T_in_A, size_B> B_dequant_right = aie::to_float<T_in_A>(B_quantized_right);
      
      const auto B_scaled_left = aie::mul(B_dequant_left, Bs_replicated);
      const auto B_scaled_right = aie::mul(B_dequant_right, Bs_replicated);
      
      // Build row vectors by extracting and concatenating 8-element chunks
      // This directly constructs the 16-element vectors needed for accumulate
      const aie::vector<T_in_A, r> b_vec_0 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(0),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(0));
      const aie::vector<T_in_A, r> b_vec_1 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(1),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(1));
      const aie::vector<T_in_A, r> b_vec_2 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(2),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(2));
      const aie::vector<T_in_A, r> b_vec_3 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(3),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(3));
      const aie::vector<T_in_A, r> b_vec_4 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(4),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(4));
      const aie::vector<T_in_A, r> b_vec_5 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(5),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(5));
      const aie::vector<T_in_A, r> b_vec_6 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(6),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(6));
      const aie::vector<T_in_A, r> b_vec_7 = aie::concat(
        B_scaled_left.template to_vector<T_in_A>().template extract<half_r>(7),
        B_scaled_right.template to_vector<T_in_A>().template extract<half_r>(7));
      
      // Accumulate: dot product of a_vec with each column of B
      c_acc = aie::accumulate<r>(
        c_acc, a_vec, 0, b_vec_0, b_vec_1, b_vec_2, b_vec_3, 
        b_vec_4, b_vec_5, b_vec_6, b_vec_7);

      a_ptr += s;
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
  X(bfloat16, bf16, int8, i8, bfloat16, bf16, accfloat, 16, 8)                    \
  //X(int16, i16, int16, i16, int32, i32, acc32, 8, 8)

#define vecmat_scalar_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,          \
                             mlir_type_in_B, ctype_out, mlir_type_out,         \
                             ctype_acc, r, s)                                 \
  void vecmat_scalar_##mlir_type_in_A##_##mlir_type_out(                      \
      ctype_in_A *a_in, ctype_in_B *b_in, ctype_out *c_out) {                 \
    vecmat_scalar<ctype_in_A, ctype_in_B, ctype_out, DIM_N, DIM_K>(a_in, b_in, c_out);    \
  }

#define vecmat_vectorized_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,      \
                                 mlir_type_in_B, ctype_out, mlir_type_out,     \
                                 ctype_acc, r, s)                             \
  void vecmat_vectorized_##mlir_type_in_A##_##mlir_type_out(                  \
      ctype_in_A *a_in, ctype_in_B *b_in, ctype_out *c_out) {                 \
    vecmat_vectorized<ctype_in_A, ctype_in_B, ctype_out, ctype_acc, DIM_N,    \
                      DIM_K, r, s>(a_in, b_in, c_out);                    \
  }

#define zero_vectorized_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,        \
                               mlir_type_in_B, ctype_out, mlir_type_out,       \
                               ctype_acc, r, s)                               \
  void zero_vectorized_##mlir_type_out(ctype_out *c_out) {                    \
    zero_vectorized<ctype_out, DIM_N, 1>(c_out);                              \
  }

#define zero_scalar_c_func(ctype_in_A, mlir_type_in_A, ctype_in_B,            \
                           mlir_type_in_B, ctype_out, mlir_type_out,           \
                           ctype_acc, r, s)                                   \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                        \
    zero_scalar<ctype_out, DIM_N, 1>(c_out);                                  \
  }

combos(vecmat_scalar_c_func) combos(vecmat_vectorized_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"
