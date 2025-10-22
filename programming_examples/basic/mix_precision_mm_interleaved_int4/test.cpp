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
  
  do_verify = true;

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  int m = 64;
  int k = 64;
  int n = 64;
  int group_size = 64; // Fixed group size for int4 quantization

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
  // Each 64x64 large tile contains: 64x32 bytes (int4 weights) + 128 bytes (bf16 scales) + 128 bytes (int8 zeros repeated) = 2304 bytes
  int num_large_tiles = (K / 64) * (N / 64);
  size_t B_SIZE = num_large_tiles * (64 * 32 + 64 * 2 + 64 * 2); // 2304 bytes per large tile
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
    // AVec[i] = matmul_common::get_random<A_DATATYPE>();
    AVec[i] = (i / K);
    // AVec[i] = (i / K) / 64;
    // AVec[i] = 1;
    // if (i < A_VOLUME / 2) {
    //   // Use 1 for the first half of the matrix
    //   AVec[i] = 4; // Use 1 for even indices
    // } else {
    //   // Use 2 for the second half of the matrix
    //   AVec[i] = 8; // Use 1 for odd indices
    // }
  }
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));
  
  // Map B buffer as char* since it contains mixed data types (int4 + bfloat16 + int8)
  char *bufB = bo_b.map<char *>();
  
  // size_t bytes_per_row = K * sizeof(B_DATATYPE) + sizeof(std::bfloat16_t);
  std::vector<int8_t> BVec(B_SIZE); // Allocate exact size for packed data
  std::vector<std::bfloat16_t> BVec_ref(B_VOLUME); // Keep original weights for reference
  std::vector<std::bfloat16_t> BFactor_ref(B_VOLUME / group_size); // Keep original factors for reference
  std::vector<int8_t> Bzeropoint_ref(B_VOLUME / group_size); // Keep original factors for reference

  for (int i = 0; i < B_VOLUME; i++) {
    // BVec_ref[i] = matmul_common::get_random<B_DATATYPE>();
    // BVec_ref[i] = (i % N) / 64;
    BVec_ref[i] = -1;
    // BVec_ref[i] = matmul_common::get_random<A_DATATYPE>();
    // if (i < 1000) {
    //   // Use 1 for the first half of the matrix
    //   BVec_ref[i] = 1; // Use 1 for even indices
    // } else {
    //   // Use 2 for the second half of the matrix
    //   BVec_ref[i] = -1; // Use -1 for odd indices
    // }
    // if (i < B_VOLUME / 2) {
    //   // Use 1 for the first half of the matrix
    //   BVec_ref[i] = 1; // Use 1 for even indices
    // } else {
    //   // Use 2 for the second half of the matrix
    //   BVec_ref[i] = -1; // Use -1 for odd indices
    // }
  }

  for (int i = 0; i < B_VOLUME / group_size; i++) {
    // BFactor_ref[i] = 1.0f; // Use 1 for all groups
    if (i < B_VOLUME / group_size / 2) {
      if (i % 2 == 0)
        BFactor_ref[i] = 1; // Use 1 for even indices
      else
        BFactor_ref[i] = 0.5; // Use 0.5 for odd indices
    } else {
      if (i % 2 == 0)
        BFactor_ref[i] = -1; // Use -1 for even indices
      else
        BFactor_ref[i] = -0.5; // Use -0.5 for odd indices
    }
  }

  for (int i = 0; i < B_VOLUME / group_size; i++) {
    // Bzeropoint_ref[i] = 0;
    if (i < B_VOLUME / group_size / 2) {
      if (i % 2 == 0)
        Bzeropoint_ref[i] = 1; // Use 1 for even indices
      else
        Bzeropoint_ref[i] = -1; // Use -1 for odd indices
    } else {
      if (i % 2 == 0)
        Bzeropoint_ref[i] = 2; // Use 2 for even indices
      else
        Bzeropoint_ref[i] = -2; // Use -2 for odd indices
    }
  }
  constexpr int LARGE_TILE_SIZE = 64;
  constexpr int SMALL_TILE_SIZE = 8;

  // Helper function to quantize weight to int4 using the correct formula
  auto quantize_to_int4 = [](float weight, float scale, int8_t zero_point) -> int8_t {
    // Quantization formula: quantized = (weight / scale) + zero_point
    float quantized = (weight / scale) + zero_point;
    // Clamp to signed int4 range [-8, 7]
    int quantized_int = std::max(-8, std::min(7, static_cast<int>(std::round(quantized))));
    // Return as signed int4 value
    return static_cast<int8_t>(quantized_int);
  };

  // Pack int4 weights into the BVec buffer with column-major large tile ordering
  char* BVec_bytes = reinterpret_cast<char*>(BVec.data());
  
  for (int large_tile_col = 0; large_tile_col < N / LARGE_TILE_SIZE; large_tile_col++) {
    for (int large_tile_row = 0; large_tile_row < K / LARGE_TILE_SIZE; large_tile_row++) {
      int large_tile_index = large_tile_col * (K / LARGE_TILE_SIZE) + large_tile_row; // Column-major indexing
      
      // Calculate byte offset for this large tile (2304 bytes per tile)
      size_t large_tile_offset = large_tile_index * (64 * 32 + 64 * 2 + 64 * 2);
      
      // Pack int4 weights (64x64 weights -> 64x32 bytes, 2 weights per byte)
      for (int small_tile_row = 0; small_tile_row < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_row++) {
        for (int small_tile_col = 0; small_tile_col < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_col++) {
          int small_tile_index = small_tile_row * (LARGE_TILE_SIZE / SMALL_TILE_SIZE) + small_tile_col;
          size_t small_tile_offset = small_tile_index * (SMALL_TILE_SIZE * SMALL_TILE_SIZE / 2); // 32 bytes per 8x8 small tile
          
          for (int i = 0; i < SMALL_TILE_SIZE; i++) {
            for (int j = 0; j < SMALL_TILE_SIZE; j += 2) { // Process 2 weights per byte
              int matrix_row1 = large_tile_row * LARGE_TILE_SIZE + small_tile_row * SMALL_TILE_SIZE + i;
              int matrix_col1 = large_tile_col * LARGE_TILE_SIZE + small_tile_col * SMALL_TILE_SIZE + j;
              int matrix_row2 = matrix_row1;
              int matrix_col2 = matrix_col1 + 1;

              if (matrix_row1 < K && matrix_col1 < N && matrix_col2 < N) {
                // Calculate linear indices in the original matrix (row-major)
                int ref_index1 = matrix_row1 * N + matrix_col1;
                int ref_index2 = matrix_row2 * N + matrix_col2;
                
                // Determine which groups these elements belong to
                int group_index1 = ref_index1 / group_size;
                int group_index2 = ref_index2 / group_size;
                
                // Quantize two weights using their respective group parameters
                int8_t weight1_q = quantize_to_int4(BVec_ref[ref_index1], BFactor_ref[group_index1], Bzeropoint_ref[group_index1]);
                int8_t weight2_q = quantize_to_int4(BVec_ref[ref_index2], BFactor_ref[group_index2], Bzeropoint_ref[group_index2]);
                
                // Pack two signed 4-bit weights into one byte (weight1 in lower 4 bits, weight2 in upper 4 bits)
                uint8_t packed_byte = (static_cast<uint8_t>(weight1_q) & 0x0F) | ((static_cast<uint8_t>(weight2_q) & 0x0F) << 4);
                
                size_t byte_offset = large_tile_offset + small_tile_offset + (i * SMALL_TILE_SIZE + j) / 2;
                BVec_bytes[byte_offset] = packed_byte;
              }
            }
          }
        }
      }
      
      // Store scale factors for this large tile (64 bf16 values = 128 bytes)
      // We need to determine which scale factors are relevant for this 64x64 tile
      size_t scale_offset = large_tile_offset + 64 * 32;
      for (int i = 0; i < LARGE_TILE_SIZE; i++) {
        int matrix_row = large_tile_row * LARGE_TILE_SIZE + i;
        if (matrix_row < K) {
          // For this row, find the scale factor for the first element in this tile
          int start_col = large_tile_col * LARGE_TILE_SIZE;
          int ref_index_start = matrix_row * N + start_col;
          int group_index = ref_index_start / group_size;
          
          std::bfloat16_t* scale_ptr = reinterpret_cast<std::bfloat16_t*>(BVec_bytes + scale_offset + i * sizeof(std::bfloat16_t));
          *scale_ptr = static_cast<std::bfloat16_t>(BFactor_ref[group_index]);
        }
      }
      
      // Store zero-points for this large tile (64 int8 values, grouped by 8 and repeated)
      // 8 groups of 8 zero-points, each group repeated twice = 8 * 8 * 2 = 128 bytes
      size_t zero_point_offset = large_tile_offset + 64 * 32 + 64 * 2;
      
      // Process in groups of 8 zero-points
      for (int group = 0; group < 8; group++) {
        // First, store the original 8 zero-points
        for (int i = 0; i < 8; i++) {
          int row_index = group * 8 + i;
          if (row_index < LARGE_TILE_SIZE) {
            int matrix_row = large_tile_row * LARGE_TILE_SIZE + row_index;
            if (matrix_row < K) {
              int start_col = large_tile_col * LARGE_TILE_SIZE;
              int ref_index_start = matrix_row * N + start_col;
              int group_index = ref_index_start / group_size;
              
              int8_t* zero_point_ptr = reinterpret_cast<int8_t*>(BVec_bytes + zero_point_offset + group * 16 + i);
              *zero_point_ptr = Bzeropoint_ref[group_index];
            }
          }
        }
        
        // Then, repeat the same 8 zero-points
        for (int i = 0; i < 8; i++) {
          int row_index = group * 8 + i;
          if (row_index < LARGE_TILE_SIZE) {
            int matrix_row = large_tile_row * LARGE_TILE_SIZE + row_index;
            if (matrix_row < K) {
              int start_col = large_tile_col * LARGE_TILE_SIZE;
              int ref_index_start = matrix_row * N + start_col;
              int group_index = ref_index_start / group_size;
              
              int8_t* zero_point_ptr = reinterpret_cast<int8_t*>(BVec_bytes + zero_point_offset + group * 16 + 8 + i);
              *zero_point_ptr = Bzeropoint_ref[group_index];
            }
          }
        }
      }
    }
  }
  
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
  bfactor_ref_csv << "# BFactor_ref (Scale Factors): " << (B_VOLUME / group_size) << " values\n";
  for (int i = 0; i < B_VOLUME / group_size; i++) {
    bfactor_ref_csv << static_cast<float>(BFactor_ref[i]) << "\n";
  }
  bfactor_ref_csv.close();

  // Write Bzeropoint_ref (zero points) to CSV file
  std::ofstream bzeropoint_ref_csv("Bzeropoint_ref.csv");
  bzeropoint_ref_csv << "# Bzeropoint_ref (Zero Points): " << (B_VOLUME / group_size) << " values\n";
  for (int i = 0; i < B_VOLUME / group_size; i++) {
    bzeropoint_ref_csv << static_cast<int>(Bzeropoint_ref[i]) << "\n";
  }
  bzeropoint_ref_csv.close();

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
    size_t large_tile_size_bytes = 64 * 32 + 64 * 2 + 64 * 2; // 2304 bytes per tile
    size_t offset_in_tile = i % large_tile_size_bytes;
    
    if (offset_in_tile < 64 * 32) {
      bvec_raw_csv << "int4_weights";
    } else if (offset_in_tile < 64 * 32 + 64 * 2) {
      bvec_raw_csv << "scale_factor_bf16";
    } else {
      bvec_raw_csv << "zero_point_int8_repeated";
    }
    bvec_raw_csv << "\n";
  }
  bvec_raw_csv.close();

  // Write BVec with tile structure annotations
  std::ofstream bvec_structured_csv("BVec_structured.csv");
  bvec_structured_csv << "# BVec Structured Data (Int4 Tile Organization)\n";
  bvec_structured_csv << "# Large tiles: " << (K/LARGE_TILE_SIZE) * (N/LARGE_TILE_SIZE) << "\n";
  bvec_structured_csv << "# Each large tile: " << LARGE_TILE_SIZE*LARGE_TILE_SIZE/2 << " bytes (int4 weights) + " 
                      << LARGE_TILE_SIZE*2 << " bytes (bf16 scales) + " << LARGE_TILE_SIZE*2 << " bytes (int8 zeros repeated)\n";
  bvec_structured_csv << "# Format: tile_type,large_tile_idx,small_tile_idx,element_idx,value1,value2\n";
  
  for (int large_tile_col = 0; large_tile_col < N / LARGE_TILE_SIZE; large_tile_col++) {
    for (int large_tile_row = 0; large_tile_row < K / LARGE_TILE_SIZE; large_tile_row++) {
      int large_tile_index = large_tile_col * (K / LARGE_TILE_SIZE) + large_tile_row;
      size_t large_tile_offset = large_tile_index * (64 * 32 + 64 * 2 + 64 * 2);
      
      bvec_structured_csv << "# Large Tile " << large_tile_index << " (row=" << large_tile_row << ", col=" << large_tile_col << ")\n";
      
      // Print int4 weights in small tiles
      for (int small_tile_row = 0; small_tile_row < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_row++) {
        for (int small_tile_col = 0; small_tile_col < LARGE_TILE_SIZE / SMALL_TILE_SIZE; small_tile_col++) {
          int small_tile_index = small_tile_row * (LARGE_TILE_SIZE / SMALL_TILE_SIZE) + small_tile_col;
          size_t small_tile_offset = small_tile_index * (SMALL_TILE_SIZE * SMALL_TILE_SIZE / 2);
          
          for (int i = 0; i < SMALL_TILE_SIZE; i++) {
            for (int j = 0; j < SMALL_TILE_SIZE; j += 2) {
              size_t byte_offset = large_tile_offset + small_tile_offset + (i * SMALL_TILE_SIZE + j) / 2;
              if (byte_offset < BVec.size()) {
                uint8_t packed_byte = static_cast<uint8_t>(BVec_bytes[byte_offset]);
                uint8_t weight1 = packed_byte & 0x0F;
                uint8_t weight2 = (packed_byte >> 4) & 0x0F;
                bvec_structured_csv << "int4_weights," << large_tile_index << "," << small_tile_index 
                                   << "," << (i * SMALL_TILE_SIZE + j) << "," << static_cast<int>(weight1) 
                                   << "," << static_cast<int>(weight2) << "\n";
              }
            }
          }
        }
      }
      
      // Print scale factors
      size_t scale_offset = large_tile_offset + 64 * 32;
      for (int i = 0; i < LARGE_TILE_SIZE; i++) {
        if (scale_offset + i * sizeof(std::bfloat16_t) + sizeof(std::bfloat16_t) <= BVec.size()) {
          std::bfloat16_t* scale_ptr = reinterpret_cast<std::bfloat16_t*>(BVec_bytes + scale_offset + i * sizeof(std::bfloat16_t));
          bvec_structured_csv << "scale_bf16," << large_tile_index << ",-1," << i << "," << static_cast<float>(*scale_ptr) << ",0\n";
        }
      }
      
      // Print zero-points (grouped by 8, each group repeated twice)
      size_t zero_point_offset = large_tile_offset + 64 * 32 + 64 * 2;
      for (int group = 0; group < 8; group++) {
        // Print first occurrence of the 8 zero-points in this group
        for (int i = 0; i < 8; i++) {
          size_t offset = zero_point_offset + group * 16 + i;
          if (offset < BVec.size()) {
            int8_t* zero_point_ptr = reinterpret_cast<int8_t*>(BVec_bytes + offset);
            bvec_structured_csv << "zero_point_group" << group << "_first," << large_tile_index << ",-1," << i << "," << static_cast<int>(*zero_point_ptr) << "," << group << "\n";
          }
        }
        // Print second occurrence (repetition) of the same 8 zero-points
        for (int i = 0; i < 8; i++) {
          size_t offset = zero_point_offset + group * 16 + 8 + i;
          if (offset < BVec.size()) {
            int8_t* zero_point_ptr = reinterpret_cast<int8_t*>(BVec_bytes + offset);
            bvec_structured_csv << "zero_point_group" << group << "_repeat," << large_tile_index << ",-1," << i << "," << static_cast<int>(*zero_point_ptr) << "," << group << "\n";
          }
        }
      }
    }
  }
  bvec_structured_csv.close();

  if (verbosity >= 1) {
    std::cout << "Data written to CSV files:\n";
    std::cout << "  - BVec_ref.csv (original reference matrix)\n";
    std::cout << "  - BFactor_ref.csv (original scale factors)\n";
    std::cout << "  - Bzeropoint_ref.csv (original zero points)\n";
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
    
    // Save output matrix C to CSV file
    std::ofstream output_c_csv("output_C.csv");
    output_c_csv << "# Output Matrix C from NPU: " << M << " rows x " << N << " columns\n";
    output_c_csv << "# Format: comma-separated values\n";
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        output_c_csv << static_cast<float>(bufC[i * N + j]);
        if (j < N - 1) output_c_csv << ",";
      }
      output_c_csv << "\n";
    }
    output_c_csv.close();

    if (do_verify) {
      memcpy(CVec.data(), bufOut, (CVec.size() * sizeof(C_DATATYPE)));
      
      // // Compute reference C matrix
      // std::vector<C_DATATYPE> CRef(C_VOLUME);
      // for (int i = 0; i < M; i++) {
      //   for (int j = 0; j < N; j++) {
      //     ACC_DATATYPE sum = 0;
      //     for (int k = 0; k < K; k++) {
      //       sum += static_cast<ACC_DATATYPE>(AVec[i * K + k]) * 
      //              static_cast<ACC_DATATYPE>(BVec_ref[k * N + j]);
      //     }
      //     CRef[i * N + j] = static_cast<C_DATATYPE>(sum);
      //   }
      // }
      
      // // Save reference matrix C to CSV file
      // std::ofstream reference_c_csv("reference_C.csv");
      // reference_c_csv << "# Reference Matrix C (CPU computed): " << M << " rows x " << N << " columns\n";
      // reference_c_csv << "# Format: comma-separated values\n";
      // for (int i = 0; i < M; i++) {
      //   for (int j = 0; j < N; j++) {
      //     reference_c_csv << static_cast<float>(CRef[i * N + j]);
      //     if (j < N - 1) reference_c_csv << ",";
      //   }
      //   reference_c_csv << "\n";
      // }
      // reference_c_csv.close();
      
      // // Save difference matrix to CSV file
      // std::ofstream diff_c_csv("diff_C.csv");
      // diff_c_csv << "# Difference Matrix (Output - Reference): " << M << " rows x " << N << " columns\n";
      // diff_c_csv << "# Format: comma-separated values\n";
      // for (int i = 0; i < M; i++) {
      //   for (int j = 0; j < N; j++) {
      //     float diff = static_cast<float>(CVec[i * N + j]) - static_cast<float>(CRef[i * N + j]);
      //     diff_c_csv << diff;
      //     if (j < N - 1) diff_c_csv << ",";
      //   }
      //   diff_c_csv << "\n";
      // }
      // diff_c_csv.close();
      
      // if (verbosity >= 1) {
      //   std::cout << "Output matrices saved to:\n";
      //   std::cout << "  - output_C.csv (NPU output)\n";
      //   std::cout << "  - reference_C.csv (CPU reference)\n";
      //   std::cout << "  - diff_C.csv (difference)\n";
      // }
      
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