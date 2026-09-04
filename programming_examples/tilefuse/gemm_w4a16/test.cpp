//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// Copyright (C) 2025, University of Illinois Urbana-Champaign.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_IN_B
#define DTYPE_IN_B int8_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN_B;
using C_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

// #define TEST_STATIC



// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
// Relaxed tolerances for Random BF16/Int4 mix
constexpr long long verify_stochastic_threshold = 1024 * 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

float abs_tol = 0.5f;
float rel_tol = 0.05f;

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);

  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int b_col_maj = vm["b_col_maj"].as<int>();

  // Fix the seed to ensure reproducibility in CI.
  srand(1726250518); 
  
  n_warmup_iterations = 16;
  n_iterations = 1; 
  do_verify = 1;
  verbosity = 2;

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  int group_size = 128; // Fixed group size for int4 quantization

  bool do_verify_stochastic =
      (long long)M * N * K > verify_stochastic_threshold;

  if (verbosity >= 1) {
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;
  }

  int A_VOLUME = M * K;
  int B_VOLUME = N * K;
  int C_VOLUME = M * N;

  size_t A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
  // B matrix: hierarchical tiling with int4 weights + bf16 scales + int8 zeros (repeated)
  // Each 128x64 large tile contains: 128x32 bytes (int4 weights) + 256 bytes (bf16 scales) + 256 bytes (int8 zeros repeated) = 4608 bytes
  int num_large_tiles = (K / 128) * (N / 64);
  size_t B_SIZE = num_large_tiles * (128 * 32 + 128 * 2 + 128 * 2); // 4608 bytes per large tile
  size_t C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  int tmp_trace_size = (trace_size > 0) ? trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size * 4, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
#ifdef TEST_STATIC
    AVec[i] = static_cast<A_DATATYPE>(0.1f);
#else
    AVec[i] = static_cast<A_DATATYPE>((float)rand() / (float)RAND_MAX * 0.19f);
#endif
  }
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));

  // Map B buffer as char* since it contains mixed data types (int4 + bfloat16 + int8)
  char *bufB = bo_b.map<char *>();

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  std::vector<C_DATATYPE> CVec(C_VOLUME);
  memset(bufOut, 0, C_SIZE);

  // Initialize B as uint8 [K, N] row major (0-15)
  std::vector<uint8_t> BVec_uint8(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
#ifdef TEST_STATIC
    BVec_uint8[i] = 2;
#else
    BVec_uint8[i] = matmul_common::get_random<uint8_t>() % 16;
#endif
  }

  // Initialize Zero Points and Scales
  // Sizes: K/group_size, N
  // For each group (128 rows), we have N scales and N zeros.
  // One scale per column per group.
  int num_groups_k = K / group_size;
  int num_scales = num_groups_k * N;

  std::vector<std::bfloat16_t> BFactor_vec(num_scales); // [K/G, N]
  std::vector<uint8_t> Bzeropoint_vec(num_scales);      // [K/G, N]

  for (int i = 0; i < num_scales; i++) {
#ifdef TEST_STATIC
    BFactor_vec[i] = static_cast<std::bfloat16_t>(0.1f);
    Bzeropoint_vec[i] = 1;
#else
    BFactor_vec[i] =
        static_cast<std::bfloat16_t>((float)rand() / (float)RAND_MAX * 0.05f);
    Bzeropoint_vec[i] = rand() % 16;
#endif
  }

  // Reference Dequantization Logic (Pure C++)
  // Output: BVec_dequant (BF16) [K, N]
  std::vector<C_DATATYPE> BVec_dequant(B_VOLUME);

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      // 1. Get Weight (uint8)
      uint8_t w = BVec_uint8[k * N + n];

      // 2. Get Scale and Zero
      // Layout [K/G, N]
      int g = k / group_size;
      int scale_idx = g * N + n;

      std::bfloat16_t s = BFactor_vec[scale_idx];
      uint8_t z = Bzeropoint_vec[scale_idx];

      // 3. Int8 Subtraction: w - z
      // Result in Int8 range, then cast to float/bf16
      int8_t w_sub_z = static_cast<int8_t>(w) - static_cast<int8_t>(z);

      // 4. Scale
      BVec_dequant[k * N + n] = static_cast<C_DATATYPE>(static_cast<float>(w_sub_z) * static_cast<float>(s));
    }
  }

  // Pack the uint8 weights and scales/zeros into BVec buffer for the NPU
  std::vector<uint8_t> BVec_packed(B_SIZE);
  char *BVec_bytes = reinterpret_cast<char *>(BVec_packed.data());

  constexpr int LARGE_TILE_SIZE_ROW = 128;
  constexpr int LARGE_TILE_SIZE_COL = 64;
  constexpr int SMALL_TILE_SIZE = 8;

  int large_tile_index = 0;
  for (int col_mod = 0; col_mod < 8; col_mod++) {
    for (int large_tile_col = col_mod; large_tile_col < N / LARGE_TILE_SIZE_COL; large_tile_col += 8) {
      for (int large_tile_row = 0; large_tile_row < K / LARGE_TILE_SIZE_ROW; large_tile_row++) {
        // Calculate byte offset for this large tile
        // New Size: 128*32 (weights) + 64*2 (scales) + 64*2 (zeros) = 4096 + 128 + 128 = 4352 bytes
        size_t large_tile_offset = large_tile_index * (128 * 32 + 64 * 2 + 64 * 2);

        // Pack int4 weights (128x64 weights -> 128x32 bytes)
        for (int small_tile_row = 0; small_tile_row < LARGE_TILE_SIZE_ROW / SMALL_TILE_SIZE; small_tile_row++) {
          for (int small_tile_col = 0; small_tile_col < LARGE_TILE_SIZE_COL / SMALL_TILE_SIZE;
               small_tile_col++) {
            // ... tiling logic same as before ...
            int small_tile_index =
                small_tile_row * (LARGE_TILE_SIZE_COL / SMALL_TILE_SIZE) + small_tile_col;
            size_t small_tile_offset = small_tile_index * (SMALL_TILE_SIZE * SMALL_TILE_SIZE / 2);

            for (int i = 0; i < SMALL_TILE_SIZE; i++) {
              for (int j = 0; j < SMALL_TILE_SIZE; j += 2) {
                int matrix_row1 =
                    large_tile_row * LARGE_TILE_SIZE_ROW + small_tile_row * SMALL_TILE_SIZE + i;
                int matrix_col1 =
                    large_tile_col * LARGE_TILE_SIZE_COL + small_tile_col * SMALL_TILE_SIZE + j;
                int matrix_col2 = matrix_col1 + 1;

                if (matrix_row1 < K && matrix_col1 < N) {
                  uint8_t w1 = BVec_uint8[matrix_row1 * N + matrix_col1];
                  uint8_t w2 = BVec_uint8[matrix_row1 * N + matrix_col2];

                  uint8_t packed_byte = (w1 & 0x0F) | ((w2 & 0x0F) << 4);

                  size_t byte_offset =
                      large_tile_offset + small_tile_offset + (i * SMALL_TILE_SIZE + j) / 2;
                  BVec_bytes[byte_offset] = packed_byte;
                }
              }
            }
          }
        }

        // Pack Scales: 64 scales * 2 bytes = 128 bytes
        size_t scale_offset = large_tile_offset + 128 * 32;

        for (int i = 0; i < 64; i++) {
          // 1 Scale per Column?
          // We have 64 columns in this tile.
          // Tile Col Index: i (0..63)

          int g = large_tile_row;
          int col = large_tile_col * 64 + i;
          int scale_idx = g * N + col;

          // Write 1 BF16 (2 bytes)
          std::bfloat16_t *scale_ptr =
              reinterpret_cast<std::bfloat16_t *>(BVec_bytes + scale_offset + i * sizeof(std::bfloat16_t));
          *scale_ptr = BFactor_vec[scale_idx];
        }

        // Pack Zeros: 64 zeros * 1 byte... duplicated to align -> 128 bytes
        // "64 values then another 64 values" -> [z0..z63, z0..z63]
        size_t zero_point_offset = large_tile_offset + 128 * 32 + 64 * 2;

        for (int group = 0; group < LARGE_TILE_SIZE_COL / SMALL_TILE_SIZE; group++) {
          for (int repeat = 0; repeat < 2; repeat++) {
            for (int i = 0; i < 8; i++) {
              int g = large_tile_row;
              int col = large_tile_col * 64 + group * 8 + i;
              int scale_idx = g * N + col;

              uint8_t z = Bzeropoint_vec[scale_idx];

              // Offset: group * 8 + i
              BVec_bytes[zero_point_offset + group * 16 + repeat * 8 + i] = z;
            }
          }
        }
        // for (int repeat = 0; repeat < 2; repeat++) {
        //   for (int i = 0; i < 64; i++) {
        //     int g = large_tile_row;
        //     int col = large_tile_col * 64 + i;
        //     int scale_idx = g * N + col;

        //     uint8_t z = Bzeropoint_vec[scale_idx];


        large_tile_index++;
      }
    }
  }

  // Copy packed data to buffer
  memcpy(bufB, BVec_packed.data(), B_SIZE);

  char *bufTrace = bo_trace.map<char *>();
  if (trace_size > 0)
    memset(bufTrace, 0, trace_size);

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
    std::cout << "Verification tolerance " << abs_tol << " absolute, "
              << rel_tol << " relative.\n";
    std::cout << "B matrix layout: Each row contains " << K << " int8 weights + 1 bf16 scale factor\n";
    std::cout << "A = \n";
    matmul_common::print_matrix(AVec, K);
    std::cout << "B = \n";
    matmul_common::print_matrix(BVec_dequant, N);
  }

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;
  float macs = 2.0 * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel (iteration " << iter << ").\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out,
                      bo_tmp1, bo_trace);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }
    C_DATATYPE *bufC = bo_out.map<C_DATATYPE *>();
    
    if (do_verify) {
      memcpy(CVec.data(), bufOut, (CVec.size() * sizeof(C_DATATYPE)));
      
      if (verbosity >= 1) {
        if (do_verify_stochastic) {
          std::cout << "Verifying " << verify_stochastic_n_samples
                    << " random samples against reference matmul ..."
                    << std::endl;
        } else {
          std::cout << "Verifying against reference matmul ..." << std::endl;
        }
      }
      auto vstart = std::chrono::system_clock::now();
      if (do_verify_stochastic) {
        errors = matmul_common::verify_stochastic<A_DATATYPE, C_DATATYPE,
                                                  ACC_DATATYPE>(
            M, N, K, AVec, BVec_dequant, CVec, verify_stochastic_n_samples, verbosity,
            abs_tol, rel_tol, b_col_maj);
      } else {
        errors = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
            M, N, K, AVec, BVec_dequant, CVec, verbosity, abs_tol, rel_tol, b_col_maj);
      }
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << " s." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // Only write out trace of last iteration.
  if (trace_size > 0) {
    matmul_common::write_out_trace((char *)bufTrace, trace_size,
                                   vm["trace_file"].as<std::string>());
  }

  std::cout << std::endl
            << "Avg NPU matmul time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors;
    if (do_verify_stochastic) {
      std::cout << " (out of " << verify_stochastic_n_samples
                << " random samples)";
    }
    std::cout << "\n\n";

    std::cout << "\nFailed.\n\n";
    return 1;
  }
}