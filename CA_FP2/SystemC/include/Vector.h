#ifndef VECTOR_H
#define VECTOR_H
#include "hw_config_selector.h"
#include <cmath>
#include <limits>
#include <string>
#include <systemc.h>
#include <vector>

template <typename T, int VSIZE, int MODE = 0>
class Vector : public sc_core::sc_module {
public:
  SC_HAS_PROCESS(Vector);

  // === Ports ===
  sc_core::sc_in<bool> clk;

  sc_core::sc_vector<sc_core::sc_in<T>>
      input_buffer; // size = ActiveHWConfig::BUF_VEC_IN_SIZE
  sc_core::sc_vector<sc_core::sc_in<T>>
      weight_buffer; // size = ActiveHWConfig::BUF_VEC_W_SIZE
  sc_core::sc_vector<sc_core::sc_out<T>>
      output_buffer; // size = ActiveHWConfig::BUF_VEC_OUT_SIZE

  sc_core::sc_in<int> DIM; // 有效元素數 n（計算區間為 [0, n)）
  sc_core::sc_out<bool> Vector_busy; // 對外 busy（唯一 writer）
  sc_core::sc_out<bool> Vector_done; // 對外 done 脈衝一拍（唯一 writer）

  // 舊版保留的控制腳位
  sc_core::sc_in<int> Vector_step;   // 目前未使用，但保留介面不動
  sc_core::sc_in<bool> vector_start; // 上升緣觸發
  sc_core::sc_in<int>
      vector_mode; // 0:rms, 1:rope, 2:softmax, 3:silu, 4:add, 5:mul

  Vector(sc_core::sc_module_name name)
      : sc_core::sc_module(name), clk("clk"),
        input_buffer("input_buffer", ActiveHWConfig::BUF_VEC_IN_SIZE),
        weight_buffer("weight_buffer", ActiveHWConfig::BUF_VEC_W_SIZE),
        output_buffer("output_buffer", ActiveHWConfig::BUF_VEC_OUT_SIZE),
        DIM("DIM"), Vector_busy("Vector_busy"), Vector_done("Vector_done"),
        Vector_step("Vector_step"), vector_start("vector_start"),
        vector_mode("vector_mode") {
    SC_THREAD(main_fsm);
    sensitive << clk.pos();
    dont_initialize();
  }

private:
  // === 延遲係數（每個 vector chunk 的週期數） ===
  // inline static int rms_latency = 951;      // 固定延遲
  // inline static int rope_latency = 22;      // per 128 elements
  // inline static int rope_overhead = 3;      // 固定延遲
  // inline static int silu_latency = 20;      // per 128 elements
  // inline static int silu_overhead = 23;     // 固定開銷
  // inline static int softmax_latency = 69;   // per 128
  // elements（另有固定開銷） inline static int softmax_overhead = 244; //
  // 固定開銷 inline static int add_latency = 11; // per 128
  // elements（element-wise add） inline static int mul_latency =
  //     11; // per 128 elements（element-wise mul，預設 11）
  // inline static int elementwise_overhead = 14; // 固定開銷
  inline static int rms_latency = 77; // RMSNorm0+1+2+3
  inline static int rms_overhead = 88;

  inline static int rope_latency = 18;
  inline static int rope_overhead = 3;

  inline static int softmax_latency = 56; // Softmax0+1+2+3 aggregated
  inline static int softmax_overhead = 133;

  inline static int silu_latency = 20;
  inline static int silu_overhead = 23;

  inline static int add_latency = 11;
  inline static int add_overhead = 14;

  inline static int mul_latency = 11;
  inline static int mul_overhead = 14;

  // === 內部狀態 ===
  enum Mode {
    IDLE = -1,
    RMS = 0,
    ROPE = 1,
    SOFTMAX = 2,
    SILU = 3,
    ADD = 4,
    MUL = 5
  };

  // 你原本會在 RMS 過程存下的值，保留語意
  T rms_value{};
  static constexpr int kVecInSize = ActiveHWConfig::BUF_VEC_IN_SIZE;
  static constexpr int kVecWSize = ActiveHWConfig::BUF_VEC_W_SIZE;
  static constexpr int kVecOutSize = ActiveHWConfig::BUF_VEC_OUT_SIZE;
  static constexpr int kVecLanes = ActiveHWConfig::VECTOR_LANES;

  static int min2(int a, int b) { return (a < b) ? a : b; }

  static int min3(int a, int b, int c) { return min2(min2(a, b), c); }

  static int unary_limit() {
    // Softmax / SiLU: only input + output matter
    return min2(kVecInSize, kVecOutSize);
  }

  static int binary_limit() {
    // RMS / ADD / MUL: input + weight + output all matter
    return min3(kVecInSize, kVecWSize, kVecOutSize);
  }

  static int rope_limit() {
    // ROPE weight layout: [cos0, sin0, cos1, sin1, ...]
    // 若 useN = 2*pairs     -> 需要 2*pairs 個 weight
    // 若 useN = 2*pairs + 1 -> 最後一個 passthrough，仍只需 2*pairs 個 weight
    // 因此 weight 能支援的最大 useN = 2*(kVecWSize/2) + 1
    const int max_by_weight = 2 * (kVecWSize / 2) + 1;
    return min3(kVecInSize, kVecOutSize, max_by_weight);
  }

  int clamp_dim(const char *op_name, int req, int limit) {
    if (req <= 0) {
      std::string msg = std::string(op_name) + " dimension <= 0, clamped to 0";
      SC_REPORT_ERROR(this->name(), msg.c_str());
      return 0;
    }

    if (req > limit) {
      std::string msg = std::string(op_name) +
                        " dimension > working range, clamped to " +
                        std::to_string(limit);
      SC_REPORT_ERROR(this->name(), msg.c_str());
      return limit;
    }

    return req;
  }

  static int chunks_by_vector_lanes(int n) {
    if (n <= 0)
      return 0;

    if (kVecLanes <= 0) {
      SC_REPORT_ERROR("Vector", "VECTOR_LANES must be > 0");
      return 0;
    }

    return (n + kVecLanes - 1) / kVecLanes;
  }
  static int calc_iter_latency(int n, int per_iter_latency, int overhead) {
    const int niters = chunks_by_vector_lanes(n);
    if (niters <= 0)
      return 0;
    return (niters - 1) * per_iter_latency + overhead;
  }

  // === 單執行緒主控 ===
  void main_fsm() {
    // 1) 初始狀態
    Vector_busy.write(false);
    Vector_done.write(false);

    bool prev_start = false; // 用來做 vector_start 上升緣偵測
    Mode cur = IDLE;

    // 2) 讓 output 在 reset 後有已知值（可選：不想清零可移除）
    for (int i = 0; i < kVecOutSize; ++i)
      output_buffer[i].write(T{});

    // 3) 同步主迴圈
    while (true) {
      wait(); // 每拍進來一次

      // 預設 done 為 0（done 是一拍脈衝）
      Vector_done.write(false);

      // 若閒置，偵測 start 上升緣並鎖存模式
      if (cur == IDLE) {
        const bool start_now = vector_start.read();
        const bool start_rise = (start_now && !prev_start);

        if (start_rise) {
          const int m = vector_mode.read();
          switch (m) {
          case 0:
            cur = RMS;
            break;
          case 1:
            cur = ROPE;
            break;
          case 2:
            cur = SOFTMAX;
            break;
          case 3:
            cur = SILU;
            break;
          case 4:
            cur = ADD;
            break; // element-wise 加法
          case 5:
            cur = MUL;
            break; // ★ 新增：element-wise 乘法
          default:
            SC_REPORT_ERROR(this->name(),
                            "Invalid vector_mode; return to IDLE");
            cur = IDLE;
            break;
          }
          if (cur != IDLE) {
            Vector_busy.write(true); // 進入工作狀態
          }
        }
      }

      // 依目前模式執行對應工作（包含延遲 wait 與輸出）
      switch (cur) {
      case RMS: {
        int n = clamp_dim("RMS", DIM.read(), binary_limit());

        if (n == 0) {
          Vector_busy.write(false);
          Vector_done.write(true);
          wait();
          Vector_done.write(false);
          cur = IDLE;
          break;
        }

        float sum_sq = 0.0f;
        for (int i = 0; i < n; ++i) {
          const float x = static_cast<float>(input_buffer[i].read());
          sum_sq += x * x;
        }

        float rms = std::sqrt((sum_sq / static_cast<float>(n)) + 1e-5f);
        if (rms == 0.0f) {
          rms = 1.0f;
          SC_REPORT_ERROR(this->name(),
                          "RMS is zero, set to 1.0 to avoid division by zero");
        }
        rms_value = static_cast<T>(rms);

        const int total_rms_wait =
            calc_iter_latency(n, rms_latency, rms_overhead);
        for (int k = 0; k < total_rms_wait; ++k)
          wait();

        for (int i = 0; i < n; ++i) {
          const float in_val = static_cast<float>(input_buffer[i].read());
          const float w_val = static_cast<float>(weight_buffer[i].read());
          const float norm = (in_val * w_val) / rms;
          output_buffer[i].write(static_cast<T>(norm));
        }

        Vector_busy.write(false);
        Vector_done.write(true);
        wait();
        Vector_done.write(false);
        cur = IDLE;
        break;
      }
      case ROPE: {
        int useN = clamp_dim("ROPE", DIM.read(), rope_limit());

        const int total_rope_wait =
            calc_iter_latency(useN, rope_latency, rope_overhead);
        for (int k = 0; k < total_rope_wait; ++k)
          wait();

        // const int pairs = useN / 2;
        // for (int k = 0; k < pairs; ++k) {
        //   const int i0 = 2 * k, i1 = i0 + 1;
        //   const float a = static_cast<float>(input_buffer[i0].read());
        //   const float b = static_cast<float>(input_buffer[i1].read());
        //   const float c = static_cast<float>(weight_buffer[i0].read()); //
        //   cos const float d = static_cast<float>(weight_buffer[i1].read());
        //   // sin const float y0 = (a * c) - (b * d); const float y1 = (a * d)
        //   + (b * c); output_buffer[i0].write(static_cast<T>(y0));
        //   output_buffer[i1].write(static_cast<T>(y1));
        // }
        const int pairs = useN / 2; // 例如 useN=64 => pairs=32

        for (int k = 0; k < pairs; ++k) {
          const int i0 = k;         // 0..31
          const int i1 = k + pairs; // 32..63  (前半配後半)  <-- HF官方

          const float a = static_cast<float>(input_buffer[i0].read());
          const float b = static_cast<float>(input_buffer[i1].read());

          // weight layout: [cos0, sin0, cos1, sin1, ...]
          const float c = static_cast<float>(
              weight_buffer[2 * k + 0].read()); // cos(theta_k)
          const float d = static_cast<float>(
              weight_buffer[2 * k + 1].read()); // sin(theta_k)

          const float y0 = (a * c) - (b * d);
          const float y1 = (a * d) + (b * c);

          output_buffer[i0].write(static_cast<T>(y0));
          output_buffer[i1].write(static_cast<T>(y1));
        }
        if (useN % 2 == 1) {
          SC_REPORT_ERROR(this->name(),
                          "ROPE dimension is odd, last element unchanged");
          const int last = useN - 1;
          output_buffer[last].write(static_cast<T>(input_buffer[last].read()));
        }

        Vector_busy.write(false);
        Vector_done.write(true);
        wait();
        Vector_done.write(false);
        cur = IDLE;
        break;
      }

      case SOFTMAX: {
        int n = clamp_dim("SOFTMAX", DIM.read(), unary_limit());

        if (n == 0) {
          Vector_busy.write(false);
          Vector_done.write(true);
          wait();
          Vector_done.write(false);
          cur = IDLE;
          break;
        }

        float x_max = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i) {
          const float x = static_cast<float>(input_buffer[i].read());
          if (x > x_max)
            x_max = x;
        }
        std::vector<float> exps(n);
        float denom = 0.0;
        for (int i = 0; i < n; ++i) {
          const float x = static_cast<float>(input_buffer[i].read());
          const float e = std::exp(x - x_max);
          exps[i] = e;
          denom += e;
        }

        const int total_softmax_wait =
            calc_iter_latency(n, softmax_latency, softmax_overhead);
        for (int k = 0; k < total_softmax_wait; ++k)
          wait();

        for (int i = 0; i < n; ++i) {
          const float p = exps[i] / denom;
          output_buffer[i].write(static_cast<T>(p));
        }

        Vector_busy.write(false);
        Vector_done.write(true);
        wait();
        Vector_done.write(false);
        cur = IDLE;
        break;
      }

      case SILU: {
        int n = clamp_dim("SILU", DIM.read(), unary_limit());

        if (n == 0) {
          Vector_busy.write(false);
          Vector_done.write(true);
          wait();
          Vector_done.write(false);
          cur = IDLE;
          break;
        }

        const int total_silu_wait =
            calc_iter_latency(n, silu_latency, silu_overhead);
        for (int k = 0; k < total_silu_wait; ++k)
          wait();

        for (int i = 0; i < n; ++i) {
          const float x = static_cast<float>(input_buffer[i].read());
          float sigmoid;
          if (x >= 0.0) {
            const float z = std::exp(-x);
            sigmoid = 1.0 / (1.0 + z);
          } else {
            const float z = std::exp(x);
            sigmoid = z / (1.0 + z);
          }
          const float y = x * sigmoid;
          output_buffer[i].write(static_cast<T>(y));
        }

        Vector_busy.write(false);
        Vector_done.write(true);
        wait();
        Vector_done.write(false);
        cur = IDLE;
        break;
      }

      case ADD: {
        int n = clamp_dim("ADD", DIM.read(), binary_limit());

        if (n == 0) {
          Vector_busy.write(false);
          Vector_done.write(true);
          wait();
          Vector_done.write(false);
          cur = IDLE;
          break;
        }

        const int total_add_wait =
            calc_iter_latency(n, add_latency, add_overhead);
        for (int k = 0; k < total_add_wait; ++k)
          wait();

        for (int i = 0; i < n; ++i) {
          const float a = static_cast<float>(input_buffer[i].read());
          const float b = static_cast<float>(weight_buffer[i].read());
          output_buffer[i].write(static_cast<T>(a + b));
        }

        Vector_busy.write(false);
        Vector_done.write(true);
        wait();
        Vector_done.write(false);
        cur = IDLE;
        break;
      }

      case MUL: { // ★ 新增：element-wise 乘法
        int n = clamp_dim("MUL", DIM.read(), binary_limit());

        if (n == 0) {
          Vector_busy.write(false);
          Vector_done.write(true);
          wait();
          Vector_done.write(false);
          cur = IDLE;
          break;
        }

        // 可選延遲：每 128 elements mul_latency 週期
        const int total_mul_wait =
            calc_iter_latency(n, mul_latency, mul_overhead);
        for (int k = 0; k < total_mul_wait; ++k)
          wait();

        // 前 n 個做逐項相乘
        for (int i = 0; i < n; ++i) {
          const float a = static_cast<float>(input_buffer[i].read());
          const float b = static_cast<float>(weight_buffer[i].read());
          output_buffer[i].write(static_cast<T>(a * b));
        }
        // [n..VSIZE-1] 不動

        Vector_busy.write(false);
        Vector_done.write(true);
        wait();
        Vector_done.write(false);
        cur = IDLE;
        break;
      }

      case IDLE:
        break;
      default:
        // 空轉即可
        break;
      }

      // 更新上緣偵測暫存
      prev_start = vector_start.read();
    } // while(true)
  }   // main_fsm
};

#endif // VECTOR_H
