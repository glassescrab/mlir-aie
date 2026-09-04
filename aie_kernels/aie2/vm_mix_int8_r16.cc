//===- vm_mix_int8_r16.cc ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// Copyright (C) 2025, University of Illinois Urbana-Champaign.
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

alignas(aie::vector_decl_align) static bfloat16 b_buf[64 * 64];

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




  // Pre-process B matrix: dequantize int8 values to bf16 and store in b_buf
  const T_in_B *__restrict pB_quantized = b;
  const T_in_B *__restrict pBs_b = b + n * k;

  const unsigned colA = k / 8;      // k / 8 (number of row tiles)
  const unsigned colB = n / 8;      // n / 8 (number of column tiles)
  const unsigned size_B = 8 * 8;    // 64 (elements per small tile)
  const unsigned half_r = r / 2;    // 8
  
  // Optimized preprocessing: use chess hints for better pipelining
  for (unsigned i = 0; i < colA; ++i) chess_prepare_for_pipelining chess_loop_range(8, ) {
    for (unsigned j = 0; j < colB; ++j) chess_flatten_loop {
      // Load per-output-channel scale factors, indexed by j (the column tile)
      // so the scale varies along the output-channel (n) axis. Hoisting this
      // out of the j loop was only valid under per-reduction-row scaling.
      const aie::vector<T_in_B, 16> Bs_row = aie::load_v<16>(pBs_b + j * 16);
      const aie::vector<T_in_A, 8> Bs_row_cast = aie::vector_cast<T_in_A>(Bs_row);
      const aie::vector<T_in_A, size_B> Bs_replicated =
        Bs_row_cast.template grow_replicate<size_B>();

      const aie::vector<T_in_B, size_B> B_quantized = 
        aie::load_v<size_B>(pB_quantized + (i * colB + j) * size_B);
      
      const aie::vector<T_in_A, size_B> B_dequantized = aie::to_float<T_in_A>(B_quantized);
      const auto B_scaled = aie::mul(B_dequantized, Bs_replicated);
      
      aie::store_v(b_buf + (i * colB + j) * size_B, B_scaled.template to_vector<T_in_A>());
    }
  }

  T_out *__restrict c_ptr = c;

  AIE_LOOP_MIN_ITERATION_COUNT(n / r)
  for (int col = 0; col < n; col += r) {
    // Initialize accumulator with current values for this block of r columns
    aie::accum<T_acc, r> c_acc;
    c_acc.from_vector(aie::load_v<r>(c_ptr));
    
    T_in_A *__restrict a_ptr = a;
    T_in_A *__restrict b_ptr = b_buf + col * s;

    // Inner loop: optimized with chess hints for better pipelining
    for (int row = 0; row < k; row += s) chess_prepare_for_pipelining chess_loop_range(8, ) {
      const aie::vector<T_in_A, s> a_vec = aie::load_v<s>(a_ptr);
      
      // Load from pre-dequantized buffer - two 8×8 tiles to form 8×16
      const aie::vector<T_in_A, r> b_vec_0 = aie::concat(aie::load_v<half_r>(b_ptr), aie::load_v<half_r>(b_ptr + s * half_r));
      const aie::vector<T_in_A, r> b_vec_1 = aie::concat(aie::load_v<half_r>(b_ptr + half_r), aie::load_v<half_r>(b_ptr + (s + 1) * half_r));
      const aie::vector<T_in_A, r> b_vec_2 = aie::concat(aie::load_v<half_r>(b_ptr + 2 * half_r), aie::load_v<half_r>(b_ptr + (s + 2) * half_r));
      const aie::vector<T_in_A, r> b_vec_3 = aie::concat(aie::load_v<half_r>(b_ptr + 3 * half_r), aie::load_v<half_r>(b_ptr + (s + 3) * half_r));
      const aie::vector<T_in_A, r> b_vec_4 = aie::concat(aie::load_v<half_r>(b_ptr + 4 * half_r), aie::load_v<half_r>(b_ptr + (s + 4) * half_r));
      const aie::vector<T_in_A, r> b_vec_5 = aie::concat(aie::load_v<half_r>(b_ptr + 5 * half_r), aie::load_v<half_r>(b_ptr + (s + 5) * half_r));
      const aie::vector<T_in_A, r> b_vec_6 = aie::concat(aie::load_v<half_r>(b_ptr + 6 * half_r), aie::load_v<half_r>(b_ptr + (s + 6) * half_r));
      const aie::vector<T_in_A, r> b_vec_7 = aie::concat(aie::load_v<half_r>(b_ptr + 7 * half_r), aie::load_v<half_r>(b_ptr + (s + 7) * half_r));

      c_acc = aie::accumulate<r>(
        c_acc, a_vec, 0, b_vec_0, b_vec_1, b_vec_2, b_vec_3, 
        b_vec_4, b_vec_5, b_vec_6, b_vec_7);

      b_ptr += n * s;
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
#define DIM_K 128  // Reduction dimension tile size (k)
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
