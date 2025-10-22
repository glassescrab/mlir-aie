//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
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

constexpr long long verify_stochastic_threshold = 1024 * 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
float abs_tol = matmul_common::get_abs_tol<C_DATATYPE>();
float rel_tol = matmul_common::get_rel_tol<C_DATATYPE>();

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
  srand(1726250518); // srand(time(NULL));

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  int m = 64;
  int k = 64;
  int n = 64;

  bool do_verify_stochastic =
      (long long)M * N * K > verify_stochastic_threshold;

  if (verbosity >= 1) {
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;
  }

  int A_VOLUME = M * K;
  int B_VOLUME = N * K;  // Will be updated later for quantized layout
  int C_VOLUME = M * N;

  size_t A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
  // B matrix: K int8 weights + 1 bfloat16 scale per row
  size_t B_SIZE = (B_VOLUME * sizeof(B_DATATYPE)) + K * N / n * sizeof(std::bfloat16_t);
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
    AVec[i] = matmul_common::get_random<A_DATATYPE>();
    // AVec[i] = (i / K) / 32;
    // if (i % K < 24) {
    //   AVec[i] = i / K / 8;
    // } else {
    //   AVec[i] = 0;
    // }

    // if (i < A_VOLUME / 2) {
    //   // Use 1 for the first half of the matrix
    //   if (i % 4 == 0)
    //     AVec[i] = 4; // Use 4 for even indices
    //   else
    //     AVec[i] = 8; // Use 8 for odd indices
    // }
    // else {
    //   // Use -4 for the second half of the matrix
    //   if (i % 4 == 0)
    //     AVec[i] = -4; // Use -4 for even indices
    //   else
    //     AVec[i] = -2; // Use -2 for odd indices
    // }
  }
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));
  
  // Map B buffer as char* since it contains mixed data types (int8 + bfloat16)
  char *bufB = bo_b.map<char *>();
  
  // size_t bytes_per_row = K * sizeof(B_DATATYPE) + sizeof(std::bfloat16_t);
  std::vector<B_DATATYPE> BVec(B_VOLUME + K * N / n * 2); // Mixed data type storage
  std::vector<A_DATATYPE> BVec_ref(B_VOLUME); // Keep original weights for reference
  std::vector<A_DATATYPE> BFactor_ref(K); // Keep original factors for reference

  for (int i = 0; i < B_VOLUME; i++) {
    // Initialize BVec with random integers between -7 and 8
    BVec_ref[i] = rand() % 16 - 8; // Random int8 values
    // if (i / K == 0) {
    //   BVec_ref[i] = 1;
    // } else {
    //   BVec_ref[i] = 0;
    // }
    // BVec_ref[0] = 1;
    // BVec_ref[i] = 1;
    // if (i < B_VOLUME / 2) {
    //   if (i % 2 == 0)
    //     BVec_ref[i] = 1; // Use 1 for even indices
    //   else
    //     BVec_ref[i] = 2; // Use 0.5 for odd indices
    // } else {
    //   if (i % 2 == 0)
    //     BVec_ref[i] = -2; // Use -2 for even indices
    //   else
    //     BVec_ref[i] = -1; // Use -1 for odd indices
    // }
  }

  for (int i = 0; i < K; i++) {
    // Initialize Factor_ref with 1/(random choice from 1, 2, 4, 8)
    BFactor_ref[i] = 1.0f / (1 << (rand() % 4));
    // Initialize Factor_ref with 1/(random int from 1 to 8)
    // BFactor_ref[i] = 1.0f / (rand() % 8 + 1);
    // if (i < 1024) {
    //   if (i % 2 == 0)
    //     BFactor_ref[i] = 1; // Use 1 for even indices
    //   else
    //     BFactor_ref[i] = 0.5; // Use 0.5 for odd indices
    // } else {
    //   if (i % 2 == 0)
    //     BFactor_ref[i] = 0.25; // Use 0.25 for even indices
    //   else
    //     BFactor_ref[i] = 0.5; // Use 0.5 for odd indices
    // }
  }

  constexpr int LARGE_TILE_SIZE = 64;
  constexpr int SMALL_TILE_SIZE = 8;

  // Use char* for mixed data type handling
  char* BVec_bytes = reinterpret_cast<char*>(BVec.data());

  // Change to column-major ordering of large tiles
  for (int large_tile_col = 0; large_tile_col < N / LARGE_TILE_SIZE; large_tile_col++) {
    for (int large_tile_row = 0; large_tile_row < K / LARGE_TILE_SIZE; large_tile_row++) {
      int large_tile_index = large_tile_col * (K / LARGE_TILE_SIZE) + large_tile_row; // Column-major indexing
      // Calculate byte offset for mixed data layout
      int large_tile_offset_bytes = large_tile_index * (LARGE_TILE_SIZE * LARGE_TILE_SIZE * sizeof(B_DATATYPE) + LARGE_TILE_SIZE * sizeof(std::bfloat16_t));
      
      // Store weight data in small tiles
      for (int small_tile_row = 0; small_tile_row < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_row++) {
        for (int small_tile_col = 0; small_tile_col < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_col++) {
          int small_tile_index = small_tile_row * (LARGE_TILE_SIZE / SMALL_TILE_SIZE) + small_tile_col;
          int small_tile_offset_bytes = small_tile_index * SMALL_TILE_SIZE * SMALL_TILE_SIZE * sizeof(B_DATATYPE);
          
          for (int i = 0; i < SMALL_TILE_SIZE; i++) {
            for (int j = 0; j < SMALL_TILE_SIZE; j++) {
              int weight_offset_bytes = large_tile_offset_bytes + small_tile_offset_bytes + (i * SMALL_TILE_SIZE + j) * sizeof(B_DATATYPE);
              int matrix_row = large_tile_row * LARGE_TILE_SIZE + small_tile_row * SMALL_TILE_SIZE + i;
              int matrix_col = large_tile_col * LARGE_TILE_SIZE + small_tile_col * SMALL_TILE_SIZE + j;

              if (matrix_row < K && matrix_col < N) {
                // Copy from reference matrix (K×N layout)
                int ref_index = matrix_row * N + matrix_col;
                B_DATATYPE* weight_ptr = reinterpret_cast<B_DATATYPE*>(BVec_bytes + weight_offset_bytes);
                // Divide the weight by the corresponding scaling factor before storing
                float scaled_weight = static_cast<float>(BVec_ref[ref_index]) / static_cast<float>(BFactor_ref[matrix_row]);
                *weight_ptr = static_cast<B_DATATYPE>(scaled_weight);
              }
            }
          }
        }
      }
      
      // Store scale factors at the end of each large tile
      for (int i = 0; i < LARGE_TILE_SIZE; i++) {
        int matrix_row = large_tile_row * LARGE_TILE_SIZE + i;
        if (matrix_row < K) {
          int scale_offset_bytes = large_tile_offset_bytes + LARGE_TILE_SIZE * LARGE_TILE_SIZE * sizeof(B_DATATYPE) + i * sizeof(std::bfloat16_t);
          std::bfloat16_t* scale_ptr = reinterpret_cast<std::bfloat16_t*>(BVec_bytes + scale_offset_bytes);
          *scale_ptr = static_cast<std::bfloat16_t>(BFactor_ref[matrix_row]);
        }
      }
    }
  }
  
  // for (int i = 0; i < B_VOLUME + K * N / 32 * 2; i++) {
  //   // AVec[i] = matmul_common::get_random<A_DATATYPE>();
  //   BVec[i] = i;
  // }
  // Copy the arranged data to the device buffer
  memcpy(bufB, BVec.data(), B_SIZE);

  // Write BVec_ref (original reference matrix) to CSV file
  std::ofstream bvec_ref_csv("BVec_ref.csv");
  bvec_ref_csv << "# BVec_ref Matrix (Original): " << K << " rows x " << N << " columns\n";
  for (int row = 0; row < K; row++) {
    for (int col = 0; col < N; col++) {
      bvec_ref_csv << static_cast<float>(BVec_ref[row * N + col]);
      if (col < N - 1) bvec_ref_csv << ",";
    }
    bvec_ref_csv << "\n";
  }
  bvec_ref_csv.close();

  // Write BFactor_ref (scale factors) to CSV file
  std::ofstream bfactor_ref_csv("BFactor_ref.csv");
  bfactor_ref_csv << "# BFactor_ref (Scale Factors): " << K << " values\n";
  for (int i = 0; i < K; i++) {
    bfactor_ref_csv << static_cast<float>(BFactor_ref[i]) << "\n";
  }
  bfactor_ref_csv.close();

  // Write entire BVec raw data to CSV file
  std::ofstream bvec_raw_csv("BVec_raw.csv");
  bvec_raw_csv << "# BVec Raw Data (Entire Vector): " << BVec.size() << " elements\n";
  bvec_raw_csv << "# Format: index,byte_value,hex_value,as_int8,interpretation\n";
  
  char* bvec_bytes = reinterpret_cast<char*>(BVec.data());
  for (size_t i = 0; i < BVec.size(); i++) {
    bvec_raw_csv << i << ","
                 << static_cast<int>(static_cast<unsigned char>(bvec_bytes[i])) << ","
                 << "0x" << std::hex << static_cast<int>(static_cast<unsigned char>(bvec_bytes[i])) << std::dec << ","
                 << static_cast<int>(static_cast<int8_t>(bvec_bytes[i])) << ",";
    
    // Add interpretation based on position
    if (i % (LARGE_TILE_SIZE * LARGE_TILE_SIZE + LARGE_TILE_SIZE * sizeof(std::bfloat16_t)) < LARGE_TILE_SIZE * LARGE_TILE_SIZE) {
      bvec_raw_csv << "weight";
    } else {
      bvec_raw_csv << "scale_factor_byte";
    }
    bvec_raw_csv << "\n";
  }
  bvec_raw_csv.close();

  // Write BVec with tile structure annotations
  std::ofstream bvec_structured_csv("BVec_structured.csv");
  bvec_structured_csv << "# BVec Structured Data (Tile Organization)\n";
  bvec_structured_csv << "# Large tiles: " << (K/LARGE_TILE_SIZE) * (N/LARGE_TILE_SIZE) << "\n";
  bvec_structured_csv << "# Each large tile: " << LARGE_TILE_SIZE*LARGE_TILE_SIZE << " weights + " << LARGE_TILE_SIZE << " scale factors\n";
  bvec_structured_csv << "# Format: tile_type,large_tile_idx,small_tile_idx,element_idx,value\n";
  
  // Change to column-major ordering of large tiles for CSV output
  for (int large_tile_col = 0; large_tile_col < N / LARGE_TILE_SIZE; large_tile_col++) {
    for (int large_tile_row = 0; large_tile_row < K / LARGE_TILE_SIZE; large_tile_row++) {
      int large_tile_index = large_tile_col * (K / LARGE_TILE_SIZE) + large_tile_row; // Column-major indexing
      int large_tile_offset_bytes = large_tile_index * (LARGE_TILE_SIZE * LARGE_TILE_SIZE * sizeof(B_DATATYPE) + LARGE_TILE_SIZE * sizeof(std::bfloat16_t));
      
      bvec_structured_csv << "# Large Tile " << large_tile_index << " (row=" << large_tile_row << ", col=" << large_tile_col << ")\n";
      
      // Print weights in small tiles
      for (int small_tile_row = 0; small_tile_row < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_row++) {
        for (int small_tile_col = 0; small_tile_col < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_col++) {
          int small_tile_index = small_tile_row * (LARGE_TILE_SIZE / SMALL_TILE_SIZE) + small_tile_col;
          int small_tile_offset_bytes = small_tile_index * SMALL_TILE_SIZE * SMALL_TILE_SIZE * sizeof(B_DATATYPE);
          
          for (int i = 0; i < SMALL_TILE_SIZE; i++) {
            for (int j = 0; j < SMALL_TILE_SIZE; j++) {
              int weight_offset_bytes = large_tile_offset_bytes + small_tile_offset_bytes + (i * SMALL_TILE_SIZE + j) * sizeof(B_DATATYPE);
              if (weight_offset_bytes < BVec.size()) {
                B_DATATYPE* weight_ptr = reinterpret_cast<B_DATATYPE*>(bvec_bytes + weight_offset_bytes);
                bvec_structured_csv << "weight," << large_tile_index << "," << small_tile_index 
                                   << "," << (i * SMALL_TILE_SIZE + j) << "," << static_cast<int>(*weight_ptr) << "\n";
              }
            }
          }
        }
      }
      
      // Print scale factors
      for (int i = 0; i < LARGE_TILE_SIZE; i++) {
        int scale_offset_bytes = large_tile_offset_bytes + LARGE_TILE_SIZE * LARGE_TILE_SIZE * sizeof(B_DATATYPE) + i * sizeof(std::bfloat16_t);
        if (scale_offset_bytes + sizeof(std::bfloat16_t) <= BVec.size()) {
          std::bfloat16_t* scale_ptr = reinterpret_cast<std::bfloat16_t*>(bvec_bytes + scale_offset_bytes);
          bvec_structured_csv << "scale," << large_tile_index << ",-1," << i << "," << static_cast<float>(*scale_ptr) << "\n";
        }
      }
    }
  }
  bvec_structured_csv.close();

  if (verbosity >= 1) {
    std::cout << "Data written to CSV files:\n";
    std::cout << "  - BVec_ref.csv (original reference matrix)\n";
    std::cout << "  - BFactor_ref.csv (original scale factors)\n";
    std::cout << "  - BVec_raw.csv (entire BVec as raw bytes)\n";
    std::cout << "  - BVec_structured.csv (BVec with tile annotations)\n";
  }

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  std::vector<C_DATATYPE> CVec(C_VOLUME);
  memset(bufOut, 0, C_SIZE);

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
    matmul_common::print_matrix(BVec_ref, N);
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
    
    // // Save actual output matrix C to CSV file
    // std::ofstream c_actual_csv("C_actual.csv");
    // c_actual_csv << "# Actual Output Matrix C: " << M << " rows x " << N << " columns\n";
    // c_actual_csv << "# Format: comma-separated values\n";
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     c_actual_csv << static_cast<float>(bufC[i * N + j]);
    //     if (j < N - 1) c_actual_csv << ",";
    //   }
    //   c_actual_csv << "\n";
    // }
    // c_actual_csv.close();

    // // Compute reference matrix C using CPU
    // std::vector<ACC_DATATYPE> CVec_ref(C_VOLUME, 0);
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     ACC_DATATYPE sum = 0;
    //     for (int k = 0; k < K; k++) {
    //       // Use the original reference values for computation
    //       ACC_DATATYPE a_val = static_cast<ACC_DATATYPE>(AVec[i * K + k]);
    //       ACC_DATATYPE b_val = static_cast<ACC_DATATYPE>(BVec_ref[k * N + j]);
    //       sum += a_val * b_val;
    //     }
    //     CVec_ref[i * N + j] = sum;
    //   }
    // }

    // // Save reference matrix C to CSV file
    // std::ofstream c_ref_csv("C_reference.csv");
    // c_ref_csv << "# Reference Matrix C: " << M << " rows x " << N << " columns\n";
    // c_ref_csv << "# Format: comma-separated values\n";
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     c_ref_csv << static_cast<float>(CVec_ref[i * N + j]);
    //     if (j < N - 1) c_ref_csv << ",";
    //   }
    //   c_ref_csv << "\n";
    // }
    // c_ref_csv.close();

    // // Save difference matrix (actual - reference) to CSV file
    // std::ofstream c_diff_csv("C_difference.csv");
    // c_diff_csv << "# Difference Matrix (Actual - Reference): " << M << " rows x " << N << " columns\n";
    // c_diff_csv << "# Format: comma-separated values\n";
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     float actual_val = static_cast<float>(bufC[i * N + j]);
    //     float ref_val = static_cast<float>(CVec_ref[i * N + j]);
    //     float diff = actual_val - ref_val;
    //     c_diff_csv << diff;
    //     if (j < N - 1) c_diff_csv << ",";
    //   }
    //   c_diff_csv << "\n";
    // }
    // c_diff_csv.close();

    // if (verbosity >= 1) {
    //   std::cout << "Output matrices saved to:\n";
    //   std::cout << "  - C_actual.csv (NPU output)\n";
    //   std::cout << "  - C_reference.csv (CPU reference)\n";
    //   std::cout << "  - C_difference.csv (actual - reference)\n";
    // }

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
            M, N, K, AVec, BVec_ref, CVec, verify_stochastic_n_samples, verbosity,
            abs_tol, rel_tol, b_col_maj);
      } else {
        errors = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
            M, N, K, AVec, BVec_ref, CVec, verbosity, abs_tol, rel_tol, b_col_maj);
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