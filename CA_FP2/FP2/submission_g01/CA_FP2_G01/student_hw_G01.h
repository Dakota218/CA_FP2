#pragma once

#include <array>

namespace student_hw {

// ============================================================
// TA grading baseline hardware configuration
// ------------------------------------------------------------
// This file defines the official grading baseline hardware
// parameters.
//
// The TA-side evaluation script compares student submissions
// against this baseline to compute normalized cost and profit.
//
// This baseline may be different from the actual hardware
// configuration.
// ============================================================
struct Config {
  static constexpr int ARRAY_MACS_PER_CYCLE = 8;
  static constexpr int DRAM_LANES = 8;
  static constexpr int SRAM_DMA_WIDTH = 64;
  static constexpr int NOC_LINK_WIDTH = 64;
  static constexpr int VECTOR_LANES = 128;

  // Fixed interface constant.
  static constexpr int NUM_BUFS = 12;

  // ----------------------------------------------------------
  // Local SRAM bank capacities (unit: FP32 elements)
  // ----------------------------------------------------------
  static constexpr int BUF_ARRAY_IN_SIZE = 4096;       // Bank 0
  static constexpr int BUF_ARRAY_OUT_SIZE = 4096;      // Bank 1
  static constexpr int BUF_VEC_IN_SIZE = 4096;         // Bank 2
  static constexpr int BUF_VEC_W_SIZE = 4096;          // Bank 3
  static constexpr int BUF_VEC_OUT_SIZE = 4096;        // Bank 4
  static constexpr int BUF_OTHER_IN_SIZE = 4096;        // Bank 5
  static constexpr int BUF_OTHER_OUT_SIZE = 4096;       // Bank 6
  static constexpr int BUF_K_SIZE = 2048;              // Bank 7
  static constexpr int BUF_V_SIZE = 2048;              // Bank 8
  static constexpr int BUF_Q_SV_SIZE = 2048;           // Bank 9
  static constexpr int BUF_RMS_SIZE = 2048;            // Bank 10
  static constexpr int BUF_CORE_CONTROLLER_SIZE = 512; // Bank 11

  inline static constexpr std::array<int, NUM_BUFS> BUF_SIZES = {
      BUF_ARRAY_IN_SIZE,  BUF_ARRAY_OUT_SIZE, BUF_VEC_IN_SIZE,
      BUF_VEC_W_SIZE,     BUF_VEC_OUT_SIZE,   BUF_OTHER_IN_SIZE,
      BUF_OTHER_OUT_SIZE, BUF_K_SIZE,         BUF_V_SIZE,
      BUF_Q_SV_SIZE,      BUF_RMS_SIZE,       BUF_CORE_CONTROLLER_SIZE};
};

} // namespace student_hw