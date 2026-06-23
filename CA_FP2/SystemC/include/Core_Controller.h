#ifndef CORE_CONTROLLER_H
#define CORE_CONTROLLER_H

#include "hw_config_selector.h"
#include "mem_types.h" // 為了 ArrayKey
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <sysc/utils/sc_report.h>
#include <systemc.h>
#include <vector>

/**
 * Core_Controller
 * ===================================================================================
 * 【角色 / 系統責任】
 *   - 從 CSV (TcoreX.csv) 逐行讀取指令
 *   - 解析成結構（ARRAY / VECTOR / DRAM / DMA / NoC）
 *   - 依序執行：一條指令「發完 + 等自己該等的事件」才讀下一條
 *
 * 【Hazard 管理（buf-level scoreboard）】
 *   - 以 buf index (0..NUM_BUFS-1) 建立 buf_busy_[b]：
 *       true  = 有某個 module 正在使用這個 buffer
 *       false = buffer 空閒，可被新的 module 使用
 *   - Array / Vector / DMA / Router SEND 在啟動前：
 *       1) 先查詢自己會用到的 buffer，呼叫 wait_bufs_idle(...) 等到全 idle
 *       2) 啟動後立刻 mark_bufs_busy(...) 標記為 busy
 *       3) 等到對應 done（或完成點）再 mark_bufs_idle(...) 釋放
 *
 * 【各單元行為摘要】
 *   - Array / Vector：
 *       * 非阻塞啟動（nowait）
 *       * 啟動條件：
 *           - 自己使用的 buffer（Array: 0/1，Vector: 2/3/4）皆為 idle
 *           - 自己 module 的 busy / reserved 為 false
 *       * 不再硬性等待 DMA / Router idle，只要 buffer 不衝突就可與 DMA / Router
 * 並行
 *
 *   - DMA / DRAM：
 *       * 所有 DRAM READ/WRITE 都包成一筆 DMA 交易（MEM = 31）
 *       * 單一 DMA 通道：同一時間最多一筆交易（透過 wait_dma_idle() 確保）
 *       * 啟動前：
 *           - 用 buf_busy_ 確認此次 src/dst 對應的 buffer 沒人用
 *           - 用 dma_busy 確認沒有其他 DMA 在跑
 *       * 啟動後：
 *           - 等 dma_done 拉高才釋放這次使用到的 buffer
 *
 *   - Router：
 *       * Router 模組本身有 SEND / RECV 兩條 FSM，可 full-duplex
 *       * Core_Controller 端的策略目前仍偏保守：
 *           - issue_router_and_wait()
 *            Router 目前允許與 A/V/DMA 平行，只由 buf_busy_ 與 router
 * reserved/busy 保護
 *           - SEND 另外利用 buf_busy_ 保護 src_buf 的 buffer hazard
 *           - RECV 僅靠 Router 自身的 busy/reserved，不碰 buffer scoreboard
 *
 * 【reserved 旗標】
 *   - vector_reserved_ / array_reserved_
 *   - router_send_reserved_ / router_recv_reserved_
 *   用途：
 *     - 在 start pulse 拉出去之後、外部 busy 尚未拉起的那幾拍，
 *       先用 reserved 擋 re-entrance，同時搭配 track_xxx_done() 在 done
 * 時清掉。
 *
 * ===================================================================================
 */
template <int MODE = 0> class Core_Controller : public sc_core::sc_module {
public:
  SC_HAS_PROCESS(Core_Controller);

  // ===== Buffer 索引常數（可調整）=====
  static constexpr int BUF_ARRAY_IN = 0;  // 2048
  static constexpr int BUF_ARRAY_OUT = 1; // 2048
  static constexpr int BUF_VEC_IN = 2;    // 2048
  static constexpr int BUF_VEC_W = 3;     // 2048
  static constexpr int BUF_VEC_OUT = 4;   // 2048
  static constexpr int BUF_OTHER_IN = 5;  // 512
  static constexpr int BUF_OTHER_OUT = 6; // 512

  static constexpr int BUF_K = 7;                // 2048
  static constexpr int BUF_V = 8;                // 2048
  static constexpr int BUF_Q_SV = 9;             // 2048
  static constexpr int BUF_RMS = 10;             // 2048
  static constexpr int BUF_CORE_CONTROLLER = 11; // 512

  static constexpr int NUM_BUFS = 12;
  static_assert(ActiveHWConfig::NUM_BUFS == 12,
                "HW config must keep NUM_BUFS = 12.");
  static constexpr int CAP_ARRAY_IN = ActiveHWConfig::BUF_SIZES[BUF_ARRAY_IN];
  static constexpr int CAP_ARRAY_OUT = ActiveHWConfig::BUF_SIZES[BUF_ARRAY_OUT];
  static constexpr int CAP_VEC_IN = ActiveHWConfig::BUF_SIZES[BUF_VEC_IN];
  static constexpr int CAP_VEC_W = ActiveHWConfig::BUF_SIZES[BUF_VEC_W];
  static constexpr int CAP_VEC_OUT = ActiveHWConfig::BUF_SIZES[BUF_VEC_OUT];
  static constexpr int CAP_OTHER_IN = ActiveHWConfig::BUF_SIZES[BUF_OTHER_IN];
  static constexpr int CAP_OTHER_OUT = ActiveHWConfig::BUF_SIZES[BUF_OTHER_OUT];
  static constexpr int CAP_K = ActiveHWConfig::BUF_SIZES[BUF_K];
  static constexpr int CAP_V = ActiveHWConfig::BUF_SIZES[BUF_V];
  static constexpr int CAP_Q_SV = ActiveHWConfig::BUF_SIZES[BUF_Q_SV];
  static constexpr int CAP_RMS = ActiveHWConfig::BUF_SIZES[BUF_RMS];
  static constexpr int CAP_CORE_CONTROLLER =
      ActiveHWConfig::BUF_SIZES[BUF_CORE_CONTROLLER];
  static constexpr int NUM_DMA_CH = 2; // ★ 每顆 Tcore 有 2 組 DMA channel
  bool buf_busy_[NUM_BUFS]{}; // true = 有某個 module 在用這個 buf

  // =========================================================================
  // Ports：系統時脈
  // =========================================================================
  sc_core::sc_in<bool> clk;

  // ====== 全部完成旗標 ======
  sc_core::sc_out<bool> all_done; // 高=全部 CSV 指令皆已執行完成

  // =========================================================================
  // Vector 控制腳位（對齊 Vector.h）
  // =========================================================================
  sc_core::sc_out<bool> vector_start;
  sc_core::sc_out<int> vector_mode;
  sc_core::sc_out<int> vector_DIM;
  sc_core::sc_in<bool> Vector_busy;
  sc_core::sc_in<bool> Vector_done;

  // =========================================================================
  // Array 控制腳位（ISA 語意版：對齊 IS/OS/QK/PV）
  // =========================================================================
  sc_core::sc_out<bool> Array_start;

  // 0=IS, 1=OS, 2=QK, 3=PV
  sc_core::sc_out<sc_dt::sc_uint<2>> Array_op;

  // ★ NEW：把 weight_name/layer/slice_idx/head 包在一起送去 Array
  sc_core::sc_out<ArrayKey> Array_key;

  // row/col 仍獨立（IS/OS 需要）
  sc_core::sc_out<int> Array_row;
  sc_core::sc_out<int> Array_col;

  sc_core::sc_in<bool> Array_busy;
  sc_core::sc_in<bool> Array_done;

  // =========================================================================
  // DMA 控制腳位（多通道版，0..NUM_DMA_CH-1）
  // =========================================================================
  // =======================
  // SEL_MODEL 協議（Core -> DMA）
  // =======================
  // 當 src_sel == tcore_sel::SEL_MODEL 時，代表 DMA 要從
  // ../DRAM_json/Llama_manifest.json 依 (layer, weight_name, slice_idx)
  // 查到：bin_file / bin_offset_bytes / bin_size_bytes， 然後到
  // ../DRAM_json/<bin_file> 讀取 FP32 (4 bytes/elem) 資料搬到 dst buffer。
  //   - dma_kv_layer[ch]    : layer id
  //   - dma_model_key[ch]   : ArrayKey{ weight_name[64], layer, slice_idx }
  //   - dma_src_offset[ch]  : 可放 hash(weight_name) 做 debug
  //   一致性檢查（非必要）
  //   - dma_length[ch]      : 讀取 elements 數（N），DMA 會檢查 N*4 <=
  //   bin_size_bytes
  sc_core::sc_out<bool> dma_start[NUM_DMA_CH];
  sc_core::sc_in<bool> dma_busy[NUM_DMA_CH];
  sc_core::sc_in<bool> dma_done[NUM_DMA_CH];
  sc_core::sc_out<sc_dt::sc_uint<5>> dma_src_sel[NUM_DMA_CH];
  sc_core::sc_out<sc_dt::sc_uint<5>> dma_dst_sel[NUM_DMA_CH];
  sc_core::sc_out<int> dma_src_offset[NUM_DMA_CH];
  sc_core::sc_out<int> dma_dst_offset[NUM_DMA_CH];
  sc_core::sc_out<int> dma_length[NUM_DMA_CH];
  // ★ 新增：給 K/V ROM 使用的 kv_head 與 prev/cur flag（每個 DMA channel 一組）
  sc_core::sc_out<int> dma_kv_head[NUM_DMA_CH]; // 0..7
  sc_core::sc_out<bool>
      dma_kv_is_prev[NUM_DMA_CH]; // 1 = token_pos-1, 0 = token_pos
  // ★ 新增：給 DMA(KVFILE) 用的 token_pos（每個 DMA channel 一組）

  sc_core::sc_out<int> dma_kv_layer[NUM_DMA_CH]; // ★ NEW
  // ★ NEW：給 DMA 用的 MODEL weight_name（固定 64B）
  // 只在 SEL_MODEL 交易時有效；其他交易會被清成空字串
  sc_core::sc_out<ArrayKey> dma_model_key[NUM_DMA_CH];

  // =========================================================================
  // Router 控制腳位（對齊新版 Router）
  // =========================================================================
  sc_core::sc_out<bool> Router_start;
  sc_core::sc_out<bool> Router_is_send;

  // ===== Multicast extension (Controller -> Router) =====
  // 讓 Router/NoC 可以「一次 SEND、多點送達」（真正 multicast）
  static constexpr int MAX_MCAST_DESTS = 64;

  sc_core::sc_out<bool> Router_is_mcast; // 1=multicast, 0=unicast/recv
  sc_core::sc_out<sc_dt::sc_uint<7>>
      Router_mcast_num; // fanout count (<= MAX_MCAST_DESTS)

  sc_core::sc_out<sc_dt::sc_uint<5>> Router_mcast_dst_x[MAX_MCAST_DESTS];
  sc_core::sc_out<sc_dt::sc_uint<5>> Router_mcast_dst_y[MAX_MCAST_DESTS];
  sc_core::sc_out<sc_dt::sc_uint<4>> Router_mcast_dst_buf[MAX_MCAST_DESTS];
  sc_core::sc_out<sc_dt::sc_uint<16>> Router_mcast_dst_offset[MAX_MCAST_DESTS];

  sc_core::sc_out<sc_dt::sc_uint<4>> Router_src_buf;
  sc_core::sc_out<sc_dt::sc_uint<4>> Router_dst_buf;
  sc_core::sc_out<sc_dt::sc_uint<16>> Router_src_offset;
  sc_core::sc_out<sc_dt::sc_uint<16>> Router_dst_offset;
  sc_core::sc_out<sc_dt::sc_uint<16>> Router_length;
  sc_core::sc_out<sc_dt::sc_uint<5>> Router_dst_x;
  sc_core::sc_out<sc_dt::sc_uint<5>> Router_dst_y;
  sc_core::sc_out<sc_dt::sc_uint<32>> Router_hash;

  // ★ 新版：四個 busy/done（send / recv 各一組）
  sc_core::sc_in<bool> Router_send_busy;
  sc_core::sc_in<bool> Router_send_done;
  sc_core::sc_in<bool> Router_recv_busy;
  sc_core::sc_in<bool> Router_recv_done;

  // ★ Router RECV 目的 buffer 回報（由 Router 根據 NoC cmd 告訴 Controller）
  sc_core::sc_in<sc_dt::sc_uint<4>> Router_recv_dst_buf;
  sc_core::sc_in<bool> Router_recv_dst_valid;
  sc_core::sc_out<bool> Router_recv_buf_grant; // Controller → Router
  sc_signal<bool> router_recv_grant_mirror_;

  sc_core::sc_in<int> token_pos;
  Core_Controller(sc_core::sc_module_name name,
                  const std::string &csv_path = "", int tcore_id = 0)
      : sc_module(name), token_pos("token_pos"), tcore_id_(tcore_id),
        csv_path_(resolve_csv_path_by_mode_(csv_path, tcore_id)) {
    SC_THREAD(run);

    SC_THREAD(track_vector_done);
    SC_THREAD(track_array_done);
    SC_THREAD(track_router_send_done);
    SC_THREAD(track_router_recv_done);
    SC_THREAD(track_router_recv_buf);
    SC_THREAD(track_dma_done); // ★ 新增：追蹤 dma_done
    {
      std::ostringstream oss;
      oss << "Core_Controller: tcore_id=" << tcore_id_
          << ", csv_path=" << csv_path_;
      SC_REPORT_INFO(this->name(), oss.str().c_str());
    }
  }

  // =========================================================================
  // buf_busy_ helper（initializer_list 版本：給固定 {0,1} 這種用）
  // =========================================================================
  bool any_buf_busy(std::initializer_list<int> bufs) const {
    for (int b : bufs) {
      if (b < 0 || b >= NUM_BUFS) {
        std::ostringstream oss;
        oss << b << " out of range (0.." << (NUM_BUFS - 1) << ")";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
        continue;
      }
      if (buf_busy_[b])
        return true;
    }
    return false;
  }

  void wait_bufs_idle(std::initializer_list<int> bufs) {
    while (any_buf_busy(bufs)) {
      wait_one_cycle();
    }
  }

  void mark_bufs_busy(std::initializer_list<int> bufs) {
    for (int b : bufs) {
      if (b < 0 || b >= NUM_BUFS) {
        std::ostringstream oss;
        oss << b << " out of range (0.." << (NUM_BUFS - 1) << ")";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
        continue;
      }
      if (buf_busy_[b]) {
        std::ostringstream oss;
        oss << "mark_bufs_busy(): buf " << b << " already busy!";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
      }
      buf_busy_[b] = true;
    }
  }

  void mark_bufs_idle(std::initializer_list<int> bufs) {
    for (int b : bufs) {
      if (b < 0 || b >= NUM_BUFS) {
        std::ostringstream oss;
        oss << b << " out of range (0.." << (NUM_BUFS - 1) << ")";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
        continue;
      }
      buf_busy_[b] = false;
    }
  }

  // =========================================================================
  // buf_busy_ helper（std::vector<int> 版本：給 DMA 動態 used_bufs 用）
  // =========================================================================
  bool any_buf_busy(const std::vector<int> &bufs) const {
    for (int b : bufs) {
      if (b < 0 || b >= NUM_BUFS) {
        std::ostringstream oss;
        oss << b << " out of range (0.." << (NUM_BUFS - 1) << ")";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
        continue;
      }
      if (buf_busy_[b])
        return true;
    }
    return false;
  }

  void wait_bufs_idle(const std::vector<int> &bufs) {
    while (any_buf_busy(bufs)) {
      wait_one_cycle();
    }
  }

  void mark_bufs_busy(const std::vector<int> &bufs) {
    for (int b : bufs) {
      if (b < 0 || b >= NUM_BUFS) {
        std::ostringstream oss;
        oss << b << " out of range (0.." << (NUM_BUFS - 1) << ")";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
        continue;
      }
      if (buf_busy_[b]) {
        std::ostringstream oss;
        oss << "mark_bufs_busy(): buf " << b << " already busy!";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
      }
      buf_busy_[b] = true;
    }
  }

  void mark_bufs_idle(const std::vector<int> &bufs) {
    for (int b : bufs) {
      if (b < 0 || b >= NUM_BUFS) {
        std::ostringstream oss;
        oss << b << " out of range (0.." << (NUM_BUFS - 1) << ")";
        SC_REPORT_ERROR(this->name(), oss.str().c_str());
        continue;
      }
      buf_busy_[b] = false;
    }
  }
  int token_pos_1b() {
    // ✅ delta-settle：避免同 timestamp 下 token_pos 剛被 write，但你讀到舊值
    wait(sc_core::SC_ZERO_TIME);

    int tp = token_pos.read();
    if (tp < 1) {
      SC_REPORT_ERROR(name(), "token_pos must be 1-based (>=1)");
      tp = 1;
    }
    return tp;
  }

  // =========================================================================
  // 檢查：所有 submodule + scoreboard 是否都 idle
  //
  // 條件：
  //   - Array / Vector / DMA / Router SEND / Router RECV 都不 busy
  //   - reserved_ 旗標都清掉
  //   - DRAM 不在使用中（dram_busy_ == false）
  //   - 所有 buf_busy_[0..NUM_BUFS-1] 都是 false
  // =========================================================================
  bool everything_idle() const {
    // Array / Vector：busy + reserved + done 都要是 0
    if (Array_busy.read() || array_reserved_ || Array_done.read())
      return false;
    if (Vector_busy.read() || vector_reserved_ || Vector_done.read())
      return false;

    // DMA：busy + reserved + done 都要是 0
    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      if (dma_busy[ch].read() || dma_reserved_[ch] || dma_done[ch].read())
        return false;
    }

    // Router SEND / RECV：busy + reserved + done 都要是 0
    if (Router_send_busy.read() || router_send_reserved_ ||
        Router_send_done.read())
      return false;
    if (Router_recv_busy.read() || router_recv_reserved_ ||
        Router_recv_done.read())
      return false;

    // DRAM lock
    if (dram_busy())
      return false;

    // buf scoreboard
    for (int b = 0; b < NUM_BUFS; ++b) {
      if (buf_busy_[b])
        return false;
    }
    return true;
  }

private:
  static constexpr int COMPILER_ADDRS_PER_TCORE = 8192;
  int compiler_base() const { return tcore_id_ * COMPILER_ADDRS_PER_TCORE; }

  int compiler_limit() const {
    return compiler_base() + COMPILER_ADDRS_PER_TCORE; // exclusive
  }
  enum class CompilerOffsetMode {
    LOCAL_PER_TCORE = 0,   // CSV 寫 0..8191
    GLOBAL_PARTITIONED = 1 // CSV 直接寫該 Tcore 對應的全域區段
  };

  // ===== 在這裡切換模式 =====
  static constexpr CompilerOffsetMode COMPILER_OFFSET_MODE =
      CompilerOffsetMode::GLOBAL_PARTITIONED;

  // 可選：如果你知道 global address 的總大小，就填正數；不知道就留 -1
  static constexpr int COMPILER_GLOBAL_ADDR_LIMIT = -1;
  int tcore_id_{0};
  static inline std::string join_path_(const std::string &dir,
                                       const std::string &file) {
    if (!file.empty() && file[0] == '/')
      return file;
    if (dir.empty())
      return file;
    if (dir.back() == '/')
      return dir + file;
    return dir + "/" + file;
  }

  static inline std::string getenv_or_default_(const char *name,
                                               const std::string &fallback) {
    const char *v = std::getenv(name);
    if (v && v[0] != '\0')
      return std::string(v);
    return fallback;
  }

  static inline std::string asmap_default_run_dir_() {
    return "/home/wilson/SystemC/auto_sched_map/runs/llama_ffn_plan000_v0";
  }

  static inline std::string asmap_instr_dir_() {
    return getenv_or_default_("ASMAP_INSTR_DIR",
                              join_path_(asmap_default_run_dir_(), "instr"));
  }

  static inline std::string mode6_core_csv_path_(int tcore_id) {
    return join_path_(asmap_instr_dir_(),
                      "core_" + std::to_string(tcore_id) + ".csv");
  }
  static inline std::string basename_(const std::string &path) {
    if (path.empty())
      return "";

    std::size_t end = path.find_last_not_of('/');
    if (end == std::string::npos)
      return "";

    std::size_t pos = path.find_last_of('/', end);
    if (pos == std::string::npos)
      return path.substr(0, end + 1);

    return path.substr(pos + 1, end - pos);
  }

  static inline std::string mode7_group_suffix_() {
    const std::string path = asmap_instr_dir_();
    for (std::size_t i = path.size(); i >= 4; --i) {
      const std::size_t p = i - 4;
      if (path[p] == '_' && path[p + 1] == 'G' && path[p + 2] >= '0' &&
          path[p + 2] <= '9' && path[p + 3] >= '0' && path[p + 3] <= '9') {
        return path.substr(p + 1, 3);
      }
    }
    return "G01";
  }

  static inline std::string mode7_core_csv_path_(int tcore_id) {
    const std::string group = mode7_group_suffix_();
    return join_path_(asmap_instr_dir_(), "core_" + std::to_string(tcore_id) +
                                              "_" + group + ".csv");
  }
  static inline std::string
  resolve_csv_path_by_mode_(const std::string &csv_path, int tcore_id) {
    if constexpr (MODE == 6) {
      // MODE 6: auto_sched_map LLaMA FFN mode.
      return mode6_core_csv_path_(tcore_id);
    } else if constexpr (MODE == 7) {
      // MODE 7: FP2 generated codegen output.
      // Example:
      //   core_0.csv ~ core_63.csv
      return mode7_core_csv_path_(tcore_id);
    } else {
      // Other modes keep the original behavior.
      if (!csv_path.empty())
        return csv_path;
      return "../ISA/Tcore0.csv";
    }
  }
  int resolve_compiler_offset_checked(int raw_off, int len) const {
    if (tcore_id_ < 0) {
      SC_REPORT_FATAL(name(), "tcore_id_ must be >= 0");
      return 0;
    }

    if (raw_off < 0) {
      SC_REPORT_FATAL(name(), "COMPILER dram_offset must be >= 0");
      return 0;
    }

    if (len <= 0) {
      SC_REPORT_FATAL(name(), "COMPILER len must be >= 1");
      return 0;
    }

    const int base = compiler_base();
    const int limit = compiler_limit(); // exclusive

    if (COMPILER_OFFSET_MODE == CompilerOffsetMode::LOCAL_PER_TCORE) {
      // raw_off 是 local offset: 0..8191
      if (raw_off >= COMPILER_ADDRS_PER_TCORE) {
        std::ostringstream oss;
        oss << "COMPILER local offset out of range: tcore=" << tcore_id_
            << " local_off=" << raw_off << " valid=[0,"
            << (COMPILER_ADDRS_PER_TCORE - 1) << "]";
        SC_REPORT_FATAL(name(), oss.str().c_str());
        return 0;
      }

      if (len > COMPILER_ADDRS_PER_TCORE - raw_off) {
        std::ostringstream oss;
        oss << "COMPILER local access crosses per-Tcore boundary: tcore="
            << tcore_id_ << " local_off=" << raw_off << " len=" << len
            << " range_size=" << COMPILER_ADDRS_PER_TCORE;
        SC_REPORT_FATAL(name(), oss.str().c_str());
        return 0;
      }

      return base + raw_off;
    }

    // GLOBAL_PARTITIONED
    // raw_off 本身就是已分區的 global offset
    if (raw_off < base || raw_off >= limit) {
      std::ostringstream oss;
      oss << "COMPILER global offset out of assigned Tcore range: tcore="
          << tcore_id_ << " global_off=" << raw_off << " valid=[" << base << ","
          << (limit - 1) << "]";
      SC_REPORT_FATAL(name(), oss.str().c_str());
      return 0;
    }

    if (len > limit - raw_off) {
      std::ostringstream oss;
      oss << "COMPILER global access crosses assigned Tcore boundary: tcore="
          << tcore_id_ << " global_off=" << raw_off << " len=" << len
          << " valid=[" << base << "," << (limit - 1) << "]";
      SC_REPORT_FATAL(name(), oss.str().c_str());
      return 0;
    }

    return raw_off;
  }
  static inline std::string to_upper(std::string s) {
    for (char &c : s)
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
  }
  static constexpr int PACK_HEAD_BITS = 8; // head 0..255 足夠
  static inline int pack_layer_head(int layer, int head) {
    return (layer << PACK_HEAD_BITS) | (head & ((1 << PACK_HEAD_BITS) - 1));
  }
  static inline int unpack_layer(int packed) {
    return packed >> PACK_HEAD_BITS;
  }
  static inline int unpack_head(int packed) {
    return packed & ((1 << PACK_HEAD_BITS) - 1);
  }

  static constexpr int KV_LEN_DEFAULT = 64;
  static constexpr int KV_PREV_FLAG =
      0x80; // ★用 head 的 bit7 當 prev 標記（不影響 layer unpack）

  static inline int kv_local_idx(int kv_head) { return (kv_head & 1); }

  static inline int kv_buf_off(int layer, int kv_head) {
    return 128 * layer + 64 * kv_local_idx(kv_head);
  }

  // packed(layer, head) 但把「prev/cur」藏在 head 的 bit7
  int pack_layer_head_kv(int layer, int kv_head, bool is_prev) {
    int head_enc = kv_head | (is_prev ? KV_PREV_FLAG : 0);
    return pack_layer_head(layer, head_enc);
  }
  static int buf_capacity(int buf_id) {
    if (buf_id < 0 || buf_id >= NUM_BUFS)
      return -1;
    return ActiveHWConfig::BUF_SIZES[static_cast<std::size_t>(buf_id)];
  }

  static bool buf_range_valid(int buf_id, int offset, int len) {
    if (buf_id < 0 || buf_id >= NUM_BUFS)
      return false;
    if (offset < 0 || len < 0)
      return false;

    const int cap = buf_capacity(buf_id);
    if (cap < 0)
      return false;

    return offset <= cap && len <= (cap - offset);
  }
  // ★ Array scoreboard 用：記住「這一次 Array 指令」鎖住了哪些 buf
  //   - 一般：{BUF_ARRAY_IN, BUF_ARRAY_OUT}
  //   - odd-QK：再加 BUF_K
  //   - odd-PV：再加 BUF_V
  std::vector<int> current_array_bufs_;

  // ★ DRAM busy（要放在第一次使用 DramOwner 之前）
  enum class DramOwner : int { NONE = 0, ARRAY = 1, DMA = 2 };
  bool try_acquire_dram(DramOwner who, int ch = -1) {
    if (dram_owner_ != DramOwner::NONE)
      return false;
    dram_owner_ = who;
    dram_owner_ch_ = ch;
    return true;
  }
  // =========================================================================
  // 在多通道 DMA 架構下，分配一條「目前 idle」的 DMA channel
  //   - 條件：dma_busy[ch] == 0 且 dma_reserved_[ch] == false
  //   - 若暫時沒有空閒 channel，會一拍一拍等，直到找到為止
  // =========================================================================
  int alloc_dma_channel() {
    int spin = 0;
    while (true) {
      for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
        if (!dma_busy[ch].read() && !dma_reserved_[ch] &&
            !dma_done[ch].read()) {
          return ch;
        }
      }
      if (++spin % 1000 == 0) {
        std::ostringstream oss;
        oss << sc_core::sc_time_stamp() << " alloc_dma_channel waiting... ";
        for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
          oss << "[ch" << ch << " busy=" << dma_busy[ch].read()
              << " reserved=" << dma_reserved_[ch]
              << " done=" << dma_done[ch].read() << "] ";
        }
        SC_REPORT_WARNING(this->name(), oss.str().c_str());
      }
      wait_one_cycle();
    }
  }

  // =========================================================================
  // done 追蹤執行緒：Vector_done → 清除軟 busy（reserved）
  // =========================================================================
  void track_vector_done() {
    bool prev = false;
    while (true) {
      wait_one_cycle();
      bool cur = Vector_done.read();
      if (cur && !prev) {
        if (!vector_reserved_) {
          SC_REPORT_WARNING(
              this->name(),
              "track_vector_done(): Vector_done but not reserved");
        }
        vector_reserved_ = false;
        mark_bufs_idle({BUF_VEC_IN, BUF_VEC_W, BUF_VEC_OUT});
      }
      prev = cur;
    }
  }
  // =========================================================================
  // done 追蹤執行緒：Array_done → 清除軟 busy（reserved）+ 釋放 DRAM
  // =========================================================================
  void track_array_done() {
    bool prev = false;
    while (true) {
      wait_one_cycle();
      bool cur = Array_done.read();
      if (cur && !prev) {
        if (!array_reserved_) {
          SC_REPORT_WARNING(this->name(),
                            "track_array_done(): Array_done but not reserved");
        }
        array_reserved_ = false;

        if (!current_array_bufs_.empty()) {
          mark_bufs_idle(current_array_bufs_);
          current_array_bufs_.clear();
        } else {
          mark_bufs_idle({BUF_ARRAY_IN, BUF_ARRAY_OUT});
        }

        release_dram(DramOwner::ARRAY);
      }
      prev = cur;
    }
  }
  // =========================================================================
  // done 追蹤執行緒：Router_send_done → 清除 SEND 的軟 busy
  // =========================================================================
  void track_router_send_done() {
    bool prev = false;
    while (true) {
      wait_one_cycle();
      bool cur = Router_send_done.read();
      if (cur && !prev) {
        router_send_reserved_ = false;
        if (current_router_send_buf_ >= 0) {
          mark_bufs_idle({current_router_send_buf_});
          current_router_send_buf_ = -1;
        }
      }
      prev = cur;
    }
  }
  // =========================================================================
  // done 追蹤執行緒：Router_recv_done → 清除 RECV 的軟 busy
  // =========================================================================
  void track_router_recv_done() {
    bool prev = false;
    while (true) {
      wait_one_cycle();
      bool cur = Router_recv_done.read();
      if (cur && !prev) {
        router_recv_reserved_ = false;
      }
      prev = cur;
    }
  }
  // =========================================================================
  // done 追蹤執行緒：DMA_done → 清除 DMA 的軟 busy + 釋放 buf / DRAM
  // =========================================================================
  void track_dma_done() {
    bool prev_done[NUM_DMA_CH]{}; // 記錄各 channel 上一拍的 done

    while (true) {
      wait_one_cycle();
      for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
        bool cur_done = dma_done[ch].read();

        // 只在 done 的「上升沿」處理一次
        if (cur_done && !prev_done[ch]) {
          // 1) 清掉這條 DMA channel 的軟 busy
          dma_reserved_[ch] = false;

          // 2) 若這筆 DMA 有動到 DRAM，釋放 dram_busy_
          if (current_dma_uses_dram_[ch]) {
            // 用 owner-based DRAM lock 機制釋放
            release_dram(DramOwner::DMA, ch);
            current_dma_uses_dram_[ch] = false;
          }

          // 3) 釋放這筆 DMA 佔用到的 buffer
          if (!current_dma_bufs_[ch].empty()) {
            // （選擇性）debug：檢查這些 buf 當下應該要是 busy
            for (int b : current_dma_bufs_[ch]) {
              if (b < 0 || b >= NUM_BUFS)
                continue;
              if (!buf_busy_[b]) {
                std::ostringstream oss;
                oss << "track_dma_done(): buf " << b
                    << " is not busy when DMA done (ch=" << ch << ")";
                SC_REPORT_ERROR(this->name(), oss.str().c_str());
              }
            }

            mark_bufs_idle(current_dma_bufs_[ch]);
            current_dma_bufs_[ch].clear();
          }
        }

        prev_done[ch] = cur_done;
      }
    }
  }

  // =========================================================================
  // done 追蹤執行緒：Router_recv_buf → 利用 NoC cmd 得知是哪一個 buf
  //
  // 行為：
  //   - 等待 Router_recv_busy 變成 1（代表開始一筆 RECV）
  //   - 等待 Router_recv_dst_valid=1（Router 已讀到 cmd，告訴我們 dst_buf）
  //   - 針對該 dst_buf 呼叫 mark_bufs_busy(...)
  //   - 等到 Router_recv_done=1，再釋放這個 buf
  //
  // 注意：
  //   - ISA 裡的 `NoC,IN` 不帶 buf 資訊，真正的 buf 只有 NoC cmd 知道
  //   - 這個 thread 就是負責把 NoC cmd 中的 dst_buf 映射回 buf_busy_ scoreboard
  // =========================================================================
  void track_router_recv_buf() {
    // 由這個 thread 單獨負責 drive Router_recv_buf_grant
    Router_recv_buf_grant.write(false);
    router_recv_grant_mirror_.write(false);
    current_router_recv_buf_ = -1;
    while (true) {
      // 1) 等 RECV 開始
      do {
        wait_one_cycle();
      } while (!Router_recv_busy.read());

      // 2) 等 Router 告訴我們 dst_buf 是誰
      while (!Router_recv_dst_valid.read()) {
        wait_one_cycle();
      }

      int dstb = static_cast<int>(Router_recv_dst_buf.read().to_uint());
      if (dstb < 0 || dstb >= NUM_BUFS) {
        {
          std::ostringstream oss;
          oss << "track_router_recv_buf(): dstb out of range (0.."
              << (NUM_BUFS - 1) << ")";
          SC_REPORT_ERROR(this->name(), oss.str().c_str());
        }
        // 不鎖任何 buf（scoreboard 不動），但為了避免 Router 卡死，
        // 仍然給一個 grant，讓 Router 可以把 data 全部 read 掉。
        current_router_recv_buf_ = -1;
        Router_recv_buf_grant.write(true);
        router_recv_grant_mirror_.write(true);
      } else {
        // ★★★ 3) 等這顆 buf 在 scoreboard 上真的變成 idle ★★★
        lock_bufs({dstb});
        current_router_recv_buf_ = dstb;

        std::ostringstream oss;
        oss << sc_core::sc_time_stamp()
            << " [controller] Router RECV lock buf=" << dstb;
        SC_REPORT_INFO(this->name(), oss.str().c_str());

        // 告訴 Router：「可以開始 do_recv(cmd) 了」
        Router_recv_buf_grant.write(true);
        router_recv_grant_mirror_.write(true);
      }

      // 4) 等這筆 RECV 完成
      while (!Router_recv_done.read()) {
        wait_one_cycle();
      }

      // 收尾
      Router_recv_buf_grant.write(false);
      router_recv_grant_mirror_.write(false);

      if (current_router_recv_buf_ >= 0) {
        mark_bufs_idle({current_router_recv_buf_});

        std::ostringstream oss;
        oss << sc_core::sc_time_stamp()
            << " [controller] Router RECV unlock buf="
            << current_router_recv_buf_;
        SC_REPORT_INFO(this->name(), oss.str().c_str());
      }

      current_router_recv_buf_ = -1;
    }
  }

  // =========================================================================
  // 內部「軟 busy」旗標
  // =========================================================================
  bool vector_reserved_{false};
  bool array_reserved_{false};
  bool router_send_reserved_{false};
  bool router_recv_reserved_{false};
  // 目前這筆 Router SEND 佔用的 buffer 編號（-1 代表目前沒有）
  int current_router_send_buf_{-1};
  // 目前這筆 Router RECV 佔用的 buffer 編號（-1 代表目前沒有）
  int current_router_recv_buf_{-1};

  DramOwner dram_owner_{DramOwner::NONE};
  int dram_owner_ch_{-1}; // 只有 DMA owner 時才用

  bool dram_busy() const { return dram_owner_ != DramOwner::NONE; }

  void acquire_dram(DramOwner who, int ch = -1) {
    while (dram_owner_ != DramOwner::NONE) {
      wait_one_cycle();
    }
    dram_owner_ = who;
    dram_owner_ch_ = ch;
  }

  void release_dram(DramOwner who, int ch = -1) {
    if (dram_owner_ != who) {
      SC_REPORT_ERROR(this->name(), "release_dram(): owner mismatch");
      return;
    }
    if (who == DramOwner::DMA && dram_owner_ch_ != ch) {
      SC_REPORT_ERROR(this->name(), "release_dram(): DMA ch mismatch");
      return;
    }
    dram_owner_ = DramOwner::NONE;
    dram_owner_ch_ = -1;
  }

  // ★ DMA scoreboard 用：
  //   - dma_reserved_：start 已經發出去，但 dma_busy 尚未拉起的短暫區間
  //   - current_dma_bufs_：這筆 DMA 佔用到的 local buffer 編號
  //   - current_dma_uses_dram_：這筆 DMA 是否動到 MEM（DRAM）
  bool dma_reserved_[NUM_DMA_CH]{};               // per-channel 軟 busy
  std::vector<int> current_dma_bufs_[NUM_DMA_CH]; // per-channel 佔用的 buf
  bool current_dma_uses_dram_[NUM_DMA_CH]{};      // per-channel 是否碰 DRAM
  // 1) 嘗試一次把一組 bufs 全部鎖住（全都 idle 才會成功）
  bool try_lock_bufs(const std::vector<int> &bufs) {
    for (int b : bufs) {
      if (b < 0 || b >= NUM_BUFS)
        return false;
      if (buf_busy_[b])
        return false;
    }
    for (int b : bufs)
      buf_busy_[b] = true;
    return true;
  }
  static inline bool valid_buf_idx(int b) { return (b >= 0 && b < NUM_BUFS); }
  // 2) 一直等到鎖成功
  void lock_bufs(const std::vector<int> &bufs) {
    for (int b : bufs) {
      if (!valid_buf_idx(b)) {
        std::ostringstream oss;
        oss << "lock_bufs(): invalid buf index " << b << " (range 0.."
            << (NUM_BUFS - 1) << ")";
        SC_REPORT_FATAL(this->name(), oss.str().c_str());
      }
    }
    while (!try_lock_bufs(bufs))
      wait_one_cycle();
  }

  void lock_bufs(std::initializer_list<int> bufs) {
    std::vector<int> v(bufs.begin(), bufs.end());
    lock_bufs(v);
  }
  // =========================================================================
  // 等待 A/V 全 idle（包含 reserved）
  //   - 這是 Router / DMA 若要「不與 A/V 平行」可用的 helper
  // =========================================================================
  void wait_idle_AV_with_reservation() {
    while (Array_busy.read() || array_reserved_ || Vector_busy.read() ||
           vector_reserved_) {
      wait_one_cycle();
    }
  }

  // =========================================================================
  // 等待 Router SEND path idle（只看 SEND 那一邊）
  //   - Router SEND 本身使用，允許 RECV 忙（full-duplex）
  // =========================================================================
  void wait_router_send_idle_with_reservation() {
    while (Router_send_busy.read() || router_send_reserved_ ||
           Router_send_done.read()) {
      wait_one_cycle();
    }
  }

  // =========================================================================
  // 等待 Router RECV path idle（只看 RECV 那一邊）
  //   - Router RECV 本身使用，允許 SEND 忙（full-duplex）
  // =========================================================================
  void wait_router_recv_idle_with_reservation() {
    while (Router_recv_busy.read() || router_recv_reserved_ ||
           Router_recv_done.read()) {
      wait_one_cycle();
    }
  }

  // =========================================================================
  // 小工具：清空 multicast ports（避免上一筆 mcast 殘留影響後續 unicast/recv）
  // =========================================================================
  void clear_router_mcast_outputs() {
    Router_is_mcast.write(false);
    Router_mcast_num.write(0);
    for (int i = 0; i < MAX_MCAST_DESTS; ++i) {
      Router_mcast_dst_x[i].write(0);
      Router_mcast_dst_y[i].write(0);
      Router_mcast_dst_buf[i].write(0);
      Router_mcast_dst_offset[i].write(0);
    }
  }

  // =========================================================================
  // 指令種類列舉 & 統一容器（解碼產物）
  // =========================================================================
  enum class ArrayOp { IS, OS, QK, PV };
  static inline sc_dt::sc_uint<2> array_op_u2(ArrayOp op) {
    switch (op) {
    case ArrayOp::IS:
      return 0;
    case ArrayOp::OS:
      return 1;
    case ArrayOp::QK:
      return 2;
    case ArrayOp::PV:
      return 3;
    default:
      return 0;
    }
  }

  static inline const char *array_op_name(ArrayOp op) {
    switch (op) {
    case ArrayOp::IS:
      return "IS";
    case ArrayOp::OS:
      return "OS";
    case ArrayOp::QK:
      return "QK";
    case ArrayOp::PV:
      return "PV";
    default:
      return "UNK";
    }
  }

  enum class Kind { ARRAY, VECTOR, DRAM, DMA, NOC, NOC_MCAST, KV };

  struct ArrayInstr {
    ArrayOp op{ArrayOp::IS};

    std::string weight_name; // IS/OS 用；QK/PV 可為空字串

    int layer{0};
    int slice_idx{0};
    int row{0};
    int col{0};

    int kv_head{0};

    std::string to_string() const {
      std::ostringstream oss;
      oss << "ARRAY " << array_op_name(op);

      if (op == ArrayOp::IS || op == ArrayOp::OS) {
        oss << " [w=" << weight_name << "]"
            << " [" << layer << "]"
            << " [" << slice_idx << "]"
            << " [" << row << "]"
            << " [" << col << "]";
      } else {
        oss << " [" << layer << "]"
            << " [" << kv_head << "]";
      }

      return oss.str();
    }
  };

  struct VectorInstr {
    int op{0}; // 0..5
    int dim{0};
  };

  struct DRAMInstr {
    bool is_read{true}; // true=READ, false=WRITE

    // typed DRAM
    bool is_rope{false};     // RoPE ROM -> BUF
    bool is_model{false};    // model.json by weight_name
    std::string weight_name; // e.g. "model.layers.0...."

    bool is_rms_att{false}; // RMS_ATT weights
    bool is_rms_ffn{false}; // RMS_FFN weights

    bool is_k{false}; // KV ROM -> BUF (or BUF -> KV)
    bool is_v{false};
    bool is_token_input{false};  // I/O ROM -> BUF
    bool is_token_output{false}; // BUF -> I/O ROM
    bool is_Compiler{false};     // Compiler ROM

    int buf_index{0};

    // legacy DRAM
    int row{0};
    int rows{0};

    // common
    int buf_offset{0};
    int len_elems{0};
    // TOKEN_I / TOKEN_O optional file offset.
    // Old format:
    //   DRAM,READ,TOKEN_I,buf,buf_offset,len
    //   DRAM,WRITE,TOKEN_O,buf,buf_offset,len
    //
    // New format:
    //   DRAM,READ,TOKEN_I,buf,buf_offset,token_io_offset,len
    //   DRAM,WRITE,TOKEN_O,buf,buf_offset,token_io_offset,len
    //
    // token_io_offset is element offset in TOKEN_I / TOKEN_O file.
    bool has_token_io_offset{false};
    int token_io_offset{0};

    int layer_id{0};
    int kv_head{0};

    int slice_idx{0}; // ★ NEW: for MODEL manifest lookup
  };
  struct DMAInstr {
    int src_buf{0};
    int src_offset{0};
    int dst_buf{0};
    int dst_offset{0};
    int len{0};
  };

  struct NocInstr {
    bool is_receive{false}; // true = NoC,IN
    int src_buf{0};
    int src_offset{0};
    int dst_x{0};
    int dst_y{0};
    int dst_buf{0};
    int dst_offset{0};
    int len{0};
    std::uint32_t hash{0}; // NEW
  };

  struct NocMcastInstr {
    int src_buf{0};
    int src_offset{0};
    int num{0}; // multicast fanout count

    struct Dest {
      int dst_x{0};
      int dst_y{0};
      int dst_buf{0};
      int dst_offset{0};
    };

    std::vector<Dest> dests; // size == num
    int len{0};

    // MODE 1~5 固定為 0；MODE 6/其他可由 CSV 指定
    std::uint32_t hash{0};
  };
  struct KVInstr {
    bool is_write{true}; // 先只做 WRITE
    bool is_k{false};
    bool is_v{false};
    int layer_id{0};
    int kv_head{0};
    int len_elems{0}; // 64
  };

  struct Decoded {
    Kind kind;
    ArrayInstr a;
    VectorInstr v;
    DRAMInstr m;
    DMAInstr d;
    NocInstr n;
    NocMcastInstr nm;
    KVInstr kv; // ★新增
  };

  std::string csv_path_;

  // =========================================================================
  // 字串工具
  // =========================================================================
  static inline std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
      return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
  }
  static inline std::string
  strip_inline_comment_hash_(const std::string &line) {
    const std::size_t p = line.find('#');
    if (p == std::string::npos)
      return line;
    return line.substr(0, p);
  }
  static inline std::vector<std::string> split_csv(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      out.push_back(trim(tok));
    }
    return out;
  }
  void issue_kv_write(const KVInstr &kv) {
    const int tp = token_pos_1b(); // 你說 token_pos 從 1 開始
    const bool odd = (tp & 1);

    const int len = kv.len_elems;
    const int off = kv_buf_off(kv.layer_id, kv.kv_head);

    const int target_buf = kv.is_k ? BUF_K : BUF_V;
    if (!buf_range_valid(target_buf, off, len)) {
      std::ostringstream oss;
      oss << "KV buf overflow: buf=" << target_buf << " off=" << off
          << " len=" << len << " cap=" << buf_capacity(target_buf)
          << " (layer=" << kv.layer_id << " head=" << kv.kv_head << ")";
      SC_REPORT_FATAL(name(), oss.str().c_str());
    }
    if (len != KV_LEN_DEFAULT) {
      SC_REPORT_WARNING(name(), "KV: len != 64 (current design assumes 64)");
    }

    // ===========
    // K path
    // ===========
    if (kv.is_k) {
      if (odd) {
        // odd: BUF_K[off] = BUF_VEC_OUT[0]
        kick_dma_nowait(BUF_VEC_OUT, 0, BUF_K, off, len);

        std::ostringstream oss;
        oss << sc_core::sc_time_stamp()
            << " [controller] KV(K) odd: stage BUF_K off=" << off
            << " <= VEC_OUT[0], len=" << len << " (layer=" << kv.layer_id
            << ", kv_head=" << kv.kv_head << ", token_pos=" << tp << ")";
        SC_REPORT_INFO(this->name(), oss.str().c_str());
      } else {
        // even: 兩次 DRAM WRITE

        // 傳遞 token_pos snapshot 到 dst_offset，DMA 端可用來避免 tp 抖動風險
        kick_dma_nowait(BUF_K, off, tcore_sel::SEL_KV_K, /*dst_offset=*/tp, len,
                        kv.kv_head, /*kv_is_prev=*/true,
                        /*kv_layer=*/kv.layer_id, "", 0);

        kick_dma_nowait(BUF_VEC_OUT, 0, tcore_sel::SEL_KV_K, /*dst_offset=*/tp,
                        len, kv.kv_head, /*kv_is_prev=*/false,
                        /*kv_layer=*/kv.layer_id, "", 0);
        std::ostringstream oss;
        oss << sc_core::sc_time_stamp() << " [controller] KV(K) even: commit2 "
            << " prev<=BUF_K off=" << off << ", cur<=VEC_OUT[0]"
            << " len=" << len << " (layer=" << kv.layer_id
            << ", kv_head=" << kv.kv_head << ", token_pos=" << tp << ")";
        SC_REPORT_INFO(this->name(), oss.str().c_str());
      }
      return;
    }

    // ===========
    // V path
    // ===========
    if (kv.is_v) {

      if (odd) {
        // odd: BUF_V[off] = BUF_ARRAY_OUT[0]
        kick_dma_nowait(BUF_ARRAY_OUT, 0, BUF_V, off, len);

        std::ostringstream oss;
        oss << sc_core::sc_time_stamp()
            << " [controller] KV(V) odd: stage BUF_V off=" << off
            << " <= ARRAY_OUT[" << 0 << "], len=" << len
            << " (layer=" << kv.layer_id << ", kv_head=" << kv.kv_head
            << ", token_pos=" << tp << ")";
        SC_REPORT_INFO(this->name(), oss.str().c_str());
      } else {
        // even: 兩次 DRAM WRITE

        // 傳遞 token_pos snapshot 到 dst_offset，DMA 端可用來避免 tp 抖動風險
        kick_dma_nowait(BUF_V, off, tcore_sel::SEL_KV_V, /*dst_offset=*/tp, len,
                        kv.kv_head, true, /*kv_layer=*/kv.layer_id, "", 0);

        kick_dma_nowait(BUF_ARRAY_OUT, 0, tcore_sel::SEL_KV_V,
                        /*dst_offset=*/tp, len, kv.kv_head, false,
                        /*kv_layer=*/kv.layer_id, "", 0);
        std::ostringstream oss;
        oss << sc_core::sc_time_stamp() << " [controller] KV(V) even: commit2 "
            << " prev<=BUF_V off=" << off << ", cur<=ARRAY_OUT[" << 0 << "]"
            << " len=" << len << " (layer=" << kv.layer_id
            << ", kv_head=" << kv.kv_head << ", token_pos=" << tp << ")";
        SC_REPORT_INFO(this->name(), oss.str().c_str());
      }
      return;
    }
  }

  // =========================================================================
  // 解析一行 CSV → Decoded
  // =========================================================================
  bool decode_line(const std::string &line, Decoded &out) {
    std::string work = line;

    // MODE 7: FP2 submission CSV supports inline comments.
    // Everything after '#' is ignored before CSV splitting.
    if constexpr (MODE == 7) {
      work = strip_inline_comment_hash_(work);
    }

    work = trim(work);
    if (work.empty())
      return false;

    auto t = split_csv(work);
    if (t.empty())
      return false;
    if (t[0].empty() || t[0][0] == '#')
      return false;

    // ---- ARRAY ----
    // New formats (space or comma separated, both OK):
    //   ARRAY, IS, weight_name, layer, slice_idx, row, col
    //   ARRAY, OS, weight_name, layer, slice_idx, row, col
    //   ARRAY, QK, layer, kv_head
    //   ARRAY, PV, layer, kv_head

    if (to_upper(t[0]) == "ARRAY") {
      if (t.size() < 2)
        return false;

      const std::string op_u = to_upper(t[1]);

      ArrayInstr a;

      if (op_u == "IS" || op_u == "OS") {
        if (t.size() != 7)
          return false;

        a.op = (op_u == "IS") ? ArrayOp::IS : ArrayOp::OS;

        // ★ weight_name 直接收字串
        a.weight_name = strip_quotes(t[2]);
        if (a.weight_name.empty()) {
          SC_REPORT_ERROR(name(), "ARRAY(IS/OS): weight_name is empty");
          return false;
        }

        a.layer = std::stoi(t[3]);
        a.slice_idx = std::stoi(t[4]);
        a.row = std::stoi(t[5]);
        a.col = std::stoi(t[6]);

        out.kind = Kind::ARRAY;
        out.a = a;
        return true;
      }

      if (op_u == "QK" || op_u == "PV") {
        if (t.size() != 4)
          return false;

        a.op = (op_u == "QK") ? ArrayOp::QK : ArrayOp::PV;
        a.layer = std::stoi(t[2]);
        a.kv_head = std::stoi(t[3]);

        out.kind = Kind::ARRAY;
        out.a = a;
        return true;
      }

      SC_REPORT_ERROR(name(), "ARRAY: only IS/OS/QK/PV are supported");
      return false;
    }

    // ---- VECTOR ----
    if (to_upper(t[0]) == "VECTOR") {
      if (t.size() != 3)
        return false;

      VectorInstr v;
      v.op = std::stoi(t[1]); // 0..5
      v.dim = std::stoi(t[2]);

      if (v.op < 0 || v.op > 5) {
        SC_REPORT_ERROR(name(), "VECTOR: OP out of range (0..5)");
        return false; // ★ 必加：不要讓錯的 op 繼續往下跑
      }

      out.kind = Kind::VECTOR;
      out.v = v;
      return true;
    }
    // ---- KV ----
    // Format: KV, WRITE, k_cache/v_cache, layer, kv_head, len
    if (to_upper(t[0]) == "KV") {
      if (t.size() != 6)
        return false;

      KVInstr kv;

      const std::string rw_u = to_upper(t[1]);
      const std::string typ_u = to_upper(t[2]);

      if (rw_u != "WRITE" && rw_u != "W") {
        SC_REPORT_ERROR(name(), "KV: only WRITE is supported for now");
        return false;
      }

      kv.is_write = true;
      kv.is_k = (typ_u == "K_CACHE" || typ_u == "K");
      kv.is_v = (typ_u == "V_CACHE" || typ_u == "V");
      if (!kv.is_k && !kv.is_v) {
        SC_REPORT_ERROR(name(), "KV: type must be K or V");
        return false;
      }

      kv.layer_id = std::stoi(t[3]);
      kv.kv_head = std::stoi(t[4]);
      kv.len_elems = std::stoi(t[5]);

      out.kind = Kind::KV;
      out.kv = kv;
      return true;
    }

    // ---- DRAM ----
    if (to_upper(t[0]) == "DRAM") {
      if (t.size() < 6)
        return false;

      DRAMInstr m;

      const std::string rw_u = to_upper(t[1]);
      if (rw_u == "READ" || rw_u == "R") {
        m.is_read = true;
      } else if (rw_u == "WRITE" || rw_u == "W") {
        m.is_read = false;
      } else {
        SC_REPORT_ERROR(name(), "DRAM: unknown RW token (use READ/WRITE)");
        return false;
      }

      const std::string typ = to_upper(t[2]);

      // =========================================================
      // 1) DRAM, READ, ROPE, dst_buf, dst_offset, len
      // =========================================================
      if (typ == "ROPE") {
        if (!m.is_read) {
          SC_REPORT_ERROR(name(), "DRAM ROPE: only READ is supported");
          return false;
        }
        if (t.size() != 6)
          return false;

        m.is_rope = true;
        m.buf_index = std::stoi(t[3]);  // dst_buf
        m.buf_offset = std::stoi(t[4]); // dst_offset
        m.len_elems = std::stoi(t[5]);  // elems

        if (m.len_elems <= 0) {
          SC_REPORT_ERROR(name(), "DRAM ROPE: len must be >= 1");
          return false;
        }

        if (!buf_range_valid(m.buf_index, m.buf_offset, m.len_elems)) {
          std::ostringstream oss;
          oss << "DRAM ROPE: local buffer range out of bounds"
              << " buf=" << m.buf_index << " offset=" << m.buf_offset
              << " len=" << m.len_elems << " cap=" << buf_capacity(m.buf_index);
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        out.kind = Kind::DRAM;
        out.m = m;
        return true;
      }

      // =========================================================
      // 2) DRAM, READ, model, "weight", dst_buf, dst_offset, layer, len
      // =========================================================
      if (typ == "MODEL") {
        if (!m.is_read) {
          SC_REPORT_ERROR(name(), "DRAM MODEL: only READ is supported");
          return false;
        }
        if (t.size() != 8 && t.size() != 9)
          return false;

        m.is_model = true;
        m.weight_name = strip_quotes(t[3]);
        if (m.weight_name.empty()) {
          SC_REPORT_ERROR(name(), "DRAM MODEL: weight_name is empty");
          return false;
        }

        m.buf_index = std::stoi(t[4]);  // dst_buf
        m.buf_offset = std::stoi(t[5]); // dst_offset
        m.layer_id = std::stoi(t[6]);   // layer
        if (t.size() == 9) {
          m.slice_idx = std::stoi(t[7]);
          m.len_elems = std::stoi(t[8]);
        } else {
          m.slice_idx = 0;
          m.len_elems = std::stoi(t[7]);
        }

        if (m.len_elems <= 0) {
          SC_REPORT_ERROR(name(), "DRAM MODEL: len must be >= 1");
          return false;
        }

        if (!buf_range_valid(m.buf_index, m.buf_offset, m.len_elems)) {
          std::ostringstream oss;
          oss << "DRAM MODEL: local buffer range out of bounds"
              << " buf=" << m.buf_index << " offset=" << m.buf_offset
              << " len=" << m.len_elems << " cap=" << buf_capacity(m.buf_index);
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        out.kind = Kind::DRAM;
        out.m = m;
        return true;
      }
      // =========================================================
      // 3) TOKEN I/O：支援 token_I / token_O（也相容 TOKEN_IN / TOKEN_OUT）
      //    Format: DRAM, READ/WRITE, token_I/token_O, buf, buf_offset, len
      //    - token_I  => SEL_TOKEN_IN
      //    - token_O  => SEL_TOKEN_OUT
      // =========================================================
      if (typ == "TOKEN_I" || typ == "TOKEN_IN" || typ == "TOKEN_INPUT" ||
          typ == "TOKEN_O" || typ == "TOKEN_OUT" || typ == "TOKEN_OUTPUT") {

        // Old:
        //   DRAM,READ,TOKEN_I,buf,buf_offset,len
        //   DRAM,WRITE,TOKEN_O,buf,buf_offset,len
        //
        // New:
        //   DRAM,READ,TOKEN_I,buf,buf_offset,token_io_offset,len
        //   DRAM,WRITE,TOKEN_O,buf,buf_offset,token_io_offset,len
        if (t.size() != 6 && t.size() != 7)
          return false;

        m.buf_index = std::stoi(t[3]); // local buf (READ=dst, WRITE=src)
        m.buf_offset = std::stoi(t[4]);

        if (t.size() == 6) {
          // Legacy format.
          m.has_token_io_offset = false;
          m.token_io_offset = 0;
          m.len_elems = std::stoi(t[5]);
        } else {
          // New format.
          m.has_token_io_offset = true;
          m.token_io_offset = std::stoi(t[5]);
          m.len_elems = std::stoi(t[6]);

          if (m.token_io_offset < 0) {
            SC_REPORT_ERROR(
                name(), "DRAM TOKEN_I/TOKEN_O: token_io_offset must be >= 0");
            return false;
          }
        }

        if (m.len_elems <= 0) {
          SC_REPORT_ERROR(name(), "DRAM TOKEN_I/TOKEN_O: len must be >= 1");
          return false;
        }

        if (!buf_range_valid(m.buf_index, m.buf_offset, m.len_elems)) {
          std::ostringstream oss;
          oss << "DRAM TOKEN_I/TOKEN_O: local buffer range out of bounds"
              << " buf=" << m.buf_index << " offset=" << m.buf_offset
              << " len=" << m.len_elems << " cap=" << buf_capacity(m.buf_index);
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        const bool is_in =
            (typ == "TOKEN_I" || typ == "TOKEN_IN" || typ == "TOKEN_INPUT");
        m.is_token_input = is_in;
        m.is_token_output = !is_in;

        out.kind = Kind::DRAM;
        out.m = m;
        return true;
      }

      // =========================================================
      // 4) DRAM, READ/WRITE, COMPILER, buf, buf_offset, dram_offset, len
      // =========================================================
      if (typ == "COMPILER") {
        if (t.size() != 7)
          return false;

        m.is_Compiler = true;
        m.buf_index = std::stoi(t[3]);
        m.buf_offset = std::stoi(t[4]);
        m.row = std::stoi(t[5]); // DRAM_offset
        m.len_elems = std::stoi(t[6]);

        if (m.buf_index < 0 || m.buf_index >= NUM_BUFS) {
          {
            std::ostringstream oss;
            oss << "DRAM COMPILER: buf_idx out of range (0.." << (NUM_BUFS - 1)
                << ")";
            SC_REPORT_ERROR(name(), oss.str().c_str());
          }
          return false;
        }
        if (m.buf_offset < 0 || m.row < 0) {
          SC_REPORT_ERROR(name(),
                          "DRAM COMPILER: buf_offset/DRAM_offset must be >=0");
          return false;
        }
        if (m.len_elems <= 0) { // len 1-based => 至少 1
          SC_REPORT_ERROR(name(), "DRAM COMPILER: len must be >= 1");
          return false;
        }
        const int base = tcore_id_ * COMPILER_ADDRS_PER_TCORE;
        const int limit = base + COMPILER_ADDRS_PER_TCORE;

        if (COMPILER_OFFSET_MODE == CompilerOffsetMode::LOCAL_PER_TCORE) {
          if (m.row >= COMPILER_ADDRS_PER_TCORE) {
            std::ostringstream oss;
            oss << "DRAM COMPILER(local): dram_offset out of range"
                << " off=" << m.row << " valid=[0,"
                << (COMPILER_ADDRS_PER_TCORE - 1) << "]";
            SC_REPORT_ERROR(name(), oss.str().c_str());
            return false;
          }

          if (m.len_elems > COMPILER_ADDRS_PER_TCORE - m.row) {
            std::ostringstream oss;
            oss << "DRAM COMPILER(local): access crosses per-Tcore boundary"
                << " off=" << m.row << " len=" << m.len_elems
                << " range_size=" << COMPILER_ADDRS_PER_TCORE;
            SC_REPORT_ERROR(name(), oss.str().c_str());
            return false;
          }
        } else { // GLOBAL_PARTITIONED
          if (m.row < base || m.row >= limit) {
            std::ostringstream oss;
            oss << "DRAM COMPILER(global): dram_offset out of assigned Tcore "
                   "range"
                << " tcore=" << tcore_id_ << " off=" << m.row << " valid=["
                << base << "," << (limit - 1) << "]";
            SC_REPORT_ERROR(name(), oss.str().c_str());
            return false;
          }

          if (m.len_elems > limit - m.row) {
            std::ostringstream oss;
            oss << "DRAM COMPILER(global): access crosses assigned Tcore "
                   "boundary"
                << " tcore=" << tcore_id_ << " off=" << m.row
                << " len=" << m.len_elems << " valid=[" << base << ","
                << (limit - 1) << "]";
            SC_REPORT_ERROR(name(), oss.str().c_str());
            return false;
          }
        }
        if (!buf_range_valid(m.buf_index, m.buf_offset, m.len_elems)) {
          std::ostringstream oss;
          oss << "DRAM COMPILER: local buffer range out of bounds"
              << " buf=" << m.buf_index << " offset=" << m.buf_offset
              << " len=" << m.len_elems << " cap=" << buf_capacity(m.buf_index);
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }
        out.kind = Kind::DRAM;
        out.m = m;
        return true;
      }

      // =========================================================
      // 5) DRAM, READ/WRITE, K/V, buf, buf_offset, layer, kv_head, len
      // =========================================================
      // if (typ == "K_CACHE" || typ == "V_CACHE" || typ == "K" || typ == "V") {
      //   if (t.size() != 8)
      //     return false;
      //
      //   m.is_k = (typ == "K" || typ == "K_CACHE");
      //   m.is_v = (typ == "V" || typ == "V_CACHE");
      //   m.buf_index =
      //       std::stoi(t[3]); // local buf (READ=dst_buf, WRITE=src_buf)
      //   m.buf_offset = std::stoi(t[4]); // local buf offset
      //   m.layer_id = std::stoi(t[5]);   // layer
      //   m.kv_head = std::stoi(t[6]);    // kv_head
      //   m.len_elems = std::stoi(t[7]);  // elems
      //
      //   if (m.len_elems <= 0) {
      //     SC_REPORT_ERROR(name(), "DRAM K/V: len must be >= 1");
      //     return false;
      //   }
      //
      //   if (!buf_range_valid(m.buf_index, m.buf_offset, m.len_elems)) {
      //     std::ostringstream oss;
      //     oss << "DRAM K/V: local buffer range out of bounds"
      //         << " buf=" << m.buf_index
      //         << " offset=" << m.buf_offset
      //         << " len=" << m.len_elems
      //         << " cap=" << buf_capacity(m.buf_index);
      //     SC_REPORT_ERROR(name(), oss.str().c_str());
      //     return false;
      //   }
      //
      //   out.kind = Kind::DRAM;
      //   out.m = m;
      //   return true;
      // }
      SC_REPORT_ERROR(
          name(), "DRAM: legacy format removed; use typed DRAM formats only");
      return false;
    }

    // ---- DMA ----
    if (to_upper(t[0]) == "DMA") {
      if (t.size() != 6)
        return false;

      DMAInstr d;
      d.src_buf = std::stoi(t[1]);
      d.src_offset = std::stoi(t[2]);
      d.dst_buf = std::stoi(t[3]);
      d.dst_offset = std::stoi(t[4]);
      d.len = std::stoi(t[5]);

      if (!valid_buf_idx(d.src_buf) || !valid_buf_idx(d.dst_buf)) {
        std::ostringstream oss;
        oss << "DMA: src_buf/dst_buf out of range (0.." << (NUM_BUFS - 1)
            << ")";
        SC_REPORT_ERROR(name(), oss.str().c_str());
        return false;
      }

      if (d.src_buf == d.dst_buf) {
        SC_REPORT_ERROR(name(), "DMA: src_buf and dst_buf cannot be the same");
        return false;
      }

      if (d.len <= 0) {
        SC_REPORT_ERROR(name(), "DMA: len must be >= 1");
        return false;
      }

      if (!buf_range_valid(d.src_buf, d.src_offset, d.len)) {
        std::ostringstream oss;
        oss << "DMA: source range out of bounds"
            << " src_buf=" << d.src_buf << " src_offset=" << d.src_offset
            << " len=" << d.len << " cap=" << buf_capacity(d.src_buf);
        SC_REPORT_ERROR(name(), oss.str().c_str());
        return false;
      }

      if (!buf_range_valid(d.dst_buf, d.dst_offset, d.len)) {
        std::ostringstream oss;
        oss << "DMA: destination range out of bounds"
            << " dst_buf=" << d.dst_buf << " dst_offset=" << d.dst_offset
            << " len=" << d.len << " cap=" << buf_capacity(d.dst_buf);
        SC_REPORT_ERROR(name(), oss.str().c_str());
        return false;
      }

      out.kind = Kind::DMA;
      out.d = d;
      return true;
    }
    // =========================================================
    // NoC RECEIVE
    //
    // MODE 1~5:
    //   - 不使用 hash
    //   - 只接受 NoC,IN
    //
    // MODE 其他:
    //   - 保留舊功能
    //   - 接受 NoC,IN 或 NoC,IN,hash
    // =========================================================
    if (to_upper(t[0]) == "NOC" && (t.size() == 2 || t.size() == 3) &&
        to_upper(t[1]) == "IN") {
      NocInstr n;
      n.is_receive = true;

      if constexpr (MODE >= 1 && MODE <= 5) {
        if (t.size() != 2) {
          SC_REPORT_ERROR(name(),
                          "NoC,IN: hash field is not supported in MODE 1~5");
          return false;
        }
        n.hash = 0u;
      } else {
        n.hash = (t.size() == 3)
                     ? static_cast<std::uint32_t>(std::stoull(t[2], nullptr, 0))
                     : 0u;
      }

      out.kind = Kind::NOC;
      out.n = n;
      return true;
    }
    // =========================================================
    // NoC SEND / MCAST
    //
    // MODE 1~5:
    //   - 沒有 hash
    //   - unicast 只能是 8 欄：
    //       NoC,src_buf,src_offset,dst_x,dst_y,dst_buf,dst_offset,len
    //   - multicast 是：
    //       NoC,src_buf,src_offset,num,dst_x,dst_y,dst_buf,dst_offset,...,len
    //     fanout=1 時剛好 9 欄，必須被解成 multicast。
    //
    // MODE 其他:
    //   - 保留 hash
    //   - 9 欄優先解成 unicast+hash
    //   - multicast 只接受 fanout>=2，避免和 unicast+hash 衝突。
    // =========================================================
    if (to_upper(t[0]) == "NOC") {
      constexpr bool NOC_HASH_ENABLED = !(MODE >= 1 && MODE <= 5);

      auto parse_mcast = [&]() -> bool {
        if (t.size() < 9)
          return false;

        NocMcastInstr nm;
        nm.src_buf = std::stoi(t[1]);
        nm.src_offset = std::stoi(t[2]);
        nm.num = std::stoi(t[3]);

        if (nm.num <= 0)
          return false;

        const std::size_t expect_no_hash =
            static_cast<std::size_t>(5 + 4 * nm.num);
        const std::size_t expect_with_hash = expect_no_hash + 1;

        bool has_mcast_hash = false;

        if constexpr (NOC_HASH_ENABLED) {
          if (t.size() == expect_with_hash) {
            // 新格式：multicast + hash
            has_mcast_hash = true;
          } else if (t.size() == expect_no_hash) {
            // 舊格式：multicast without hash
            // 注意：hash-enabled mode 下，fanout=1 且沒有 hash 會和
            // unicast+hash 撞格式， 所以不允許。
            if (nm.num == 1) {
              return false;
            }
            has_mcast_hash = false;
          } else {
            return false;
          }
        } else {
          // MODE 1~5：完全沒有 hash，只接受舊格式
          if (t.size() != expect_no_hash) {
            return false;
          }
          has_mcast_hash = false;
        }

        nm.dests.clear();
        nm.dests.reserve(static_cast<std::size_t>(nm.num));

        for (int i = 0; i < nm.num; ++i) {
          const std::size_t base = static_cast<std::size_t>(4 + 4 * i);

          typename Core_Controller<MODE>::NocMcastInstr::Dest d;
          d.dst_x = std::stoi(t[base + 0]);
          d.dst_y = std::stoi(t[base + 1]);
          d.dst_buf = std::stoi(t[base + 2]);
          d.dst_offset = std::stoi(t[base + 3]);

          nm.dests.push_back(d);
        }

        nm.len = std::stoi(t[4 + 4 * nm.num]);
        if constexpr (NOC_HASH_ENABLED) {
          nm.hash = has_mcast_hash ? static_cast<std::uint32_t>(std::stoull(
                                         t[5 + 4 * nm.num], nullptr, 0))
                                   : 0u;
        } else {
          nm.hash = 0u;
        }
        if (nm.len <= 0) {
          SC_REPORT_ERROR(name(), "NoC multicast: len must be >= 1");
          return false;
        }

        if (!valid_buf_idx(nm.src_buf)) {
          std::ostringstream oss;
          oss << "NoC multicast: src_buf out of range (0.." << (NUM_BUFS - 1)
              << ")";
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        if (!buf_range_valid(nm.src_buf, nm.src_offset, nm.len)) {
          std::ostringstream oss;
          oss << "NoC multicast: source range out of bounds"
              << " src_buf=" << nm.src_buf << " src_offset=" << nm.src_offset
              << " len=" << nm.len << " cap=" << buf_capacity(nm.src_buf);
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        for (int i = 0; i < nm.num; ++i) {
          if (!valid_buf_idx(nm.dests[i].dst_buf)) {
            std::ostringstream oss;
            oss << "NoC multicast: dst_buf out of range at fanout[" << i
                << "] (0.." << (NUM_BUFS - 1) << ")";
            SC_REPORT_ERROR(name(), oss.str().c_str());
            return false;
          }

          if (!buf_range_valid(nm.dests[i].dst_buf, nm.dests[i].dst_offset,
                               nm.len)) {
            std::ostringstream oss;
            oss << "NoC multicast: destination range out of bounds"
                << " fanout[" << i << "]"
                << " dst_buf=" << nm.dests[i].dst_buf
                << " dst_offset=" << nm.dests[i].dst_offset << " len=" << nm.len
                << " cap=" << buf_capacity(nm.dests[i].dst_buf);
            SC_REPORT_ERROR(name(), oss.str().c_str());
            return false;
          }
        }

        out.kind = Kind::NOC_MCAST;
        out.nm = nm;
        return true;
      };

      auto parse_unicast = [&]() -> bool {
        if constexpr (NOC_HASH_ENABLED) {
          if (!(t.size() == 8 || t.size() == 9))
            return false;
        } else {
          // MODE 1~5 沒有 hash，因此 unicast 只允許 8 欄
          if (t.size() != 8)
            return false;
        }

        NocInstr n;
        n.is_receive = false;
        n.src_buf = std::stoi(t[1]);
        n.src_offset = std::stoi(t[2]);
        n.dst_x = std::stoi(t[3]);
        n.dst_y = std::stoi(t[4]);
        n.dst_buf = std::stoi(t[5]);
        n.dst_offset = std::stoi(t[6]);
        n.len = std::stoi(t[7]);

        if constexpr (NOC_HASH_ENABLED) {
          n.hash =
              (t.size() == 9)
                  ? static_cast<std::uint32_t>(std::stoull(t[8], nullptr, 0))
                  : 0u;
        } else {
          n.hash = 0u;
        }

        if (n.len <= 0) {
          std::ostringstream oss;
          oss << "NoC: len must be >= 1"
              << " line=\"" << line << "\""
              << " parsed_as=UNICAST"
              << " src_buf=" << n.src_buf << " src_offset=" << n.src_offset
              << " dst_x=" << n.dst_x << " dst_y=" << n.dst_y
              << " dst_buf=" << n.dst_buf << " dst_offset=" << n.dst_offset
              << " len=" << n.len << " hash=0x" << std::hex << n.hash
              << std::dec;
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        if (!valid_buf_idx(n.src_buf) || !valid_buf_idx(n.dst_buf)) {
          std::ostringstream oss;
          oss << "NoC: src_buf/dst_buf out of range (0.." << (NUM_BUFS - 1)
              << ")";
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        if (!buf_range_valid(n.src_buf, n.src_offset, n.len)) {
          std::ostringstream oss;
          oss << "NoC: source range out of bounds"
              << " src_buf=" << n.src_buf << " src_offset=" << n.src_offset
              << " len=" << n.len << " cap=" << buf_capacity(n.src_buf);
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        if (!buf_range_valid(n.dst_buf, n.dst_offset, n.len)) {
          std::ostringstream oss;
          oss << "NoC: destination range out of bounds"
              << " dst_buf=" << n.dst_buf << " dst_offset=" << n.dst_offset
              << " len=" << n.len << " cap=" << buf_capacity(n.dst_buf);
          SC_REPORT_ERROR(name(), oss.str().c_str());
          return false;
        }

        out.kind = Kind::NOC;
        out.n = n;
        return true;
      };

      if constexpr (MODE >= 1 && MODE <= 5) {
        // MODE 1~5 沒有 hash：
        // 9 欄 fanout=1 multicast 必須先解，不然會被誤判。
        if (parse_mcast())
          return true;

        if (parse_unicast())
          return true;

        SC_REPORT_ERROR(
            name(),
            "NoC: invalid format in MODE 1~5 "
            "(unicast must have 8 fields; multicast uses fanout format)");
        return false;
      } else {
        // hash-enabled mode：
        // 9 欄優先是 unicast+hash；fanout=1 multicast 不在這裡猜。
        if (parse_unicast())
          return true;

        if (parse_mcast())
          return true;

        SC_REPORT_ERROR(name(), "NoC: invalid format");
        return false;
      }
    }
    return false;
  }

  static inline std::string strip_quotes(std::string s) {
    s = trim(s);
    if (s.size() >= 2) {
      const char a = s.front();
      const char b = s.back();
      if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
        s = s.substr(1, s.size() - 2);
        s = trim(s);
      }
    }
    return s;
  }

  // 32-bit FNV-1a hash：把 weight_name 轉成可塞進 int 的 token
  static inline std::uint32_t fnv1a_32(const std::string &s) {
    std::uint32_t h = 2166136261u;
    for (unsigned char c : s) {
      h ^= static_cast<std::uint32_t>(c);
      h *= 16777619u;
    }
    return h;
  }
  // =========================================================================
  // 等待一個時脈正緣 / 多個 cycle
  // =========================================================================
  void wait_one_cycle() { wait(clk.posedge_event()); }

  void wait_n_cycles(int n) {
    for (int i = 0; i < n; ++i) {
      wait_one_cycle();
    }
  }
  // =========================================================================
  // Vector：非阻塞啟動（nowait）
  // =========================================================================
  // ===== 小工具：Vector mode / buf name 轉字串（可放在 class private/public
  // 都行）=====
  static inline const char *vec_mode_str_(int mode) {
    switch (mode) {
    // 如果你有其他 mode，自己在這裡補：
    case 0:
      return "RMSNORM";
    case 1:
      return "RoPE";
    case 2:
      return "SOFTMAX";
    case 3:
      return "SiLU";
    case 4:
      return "ADD";
    case 5:
      return "MUL";
    default:
      return "UNKNOWN";
    }
  }

  static inline const char *buf_name_(int b) {
    switch (b) {
    case BUF_VEC_IN:
      return "BUF_VEC_IN";
    case BUF_VEC_W:
      return "BUF_VEC_W";
    case BUF_VEC_OUT:
      return "BUF_VEC_OUT";
    // 若你想也印其他 buf，可再補
    default:
      return "BUF_?";
    }
  }
  // ===== 你要的：帶 log 的 issue_vector_nowait =====
  void issue_vector_nowait(int mode, int dim) {
    int waited = 0;
    while (Vector_busy.read() || vector_reserved_ || Vector_done.read()) {
      waited++;
      wait_one_cycle();
    }

    // local variables for logging / port driving
    const int tp = token_pos_1b(); // 1-based token_pos
    int dim_eff = dim;
    bool dim_from_tp = false;
    bool dim_forced_const = false;

    if constexpr (MODE == 0 || MODE == 6) {
      // Original LLaMA modes use token_pos as the softmax dimension.
      if (mode == 2) {
        dim_eff = tp;
        dim_from_tp = true;
      }
    } else if constexpr (MODE >= 1 && MODE <= 4) {
      // Legacy MNIST modes always have 10 output classes.
      if (mode == 2) {
        dim_eff = 10;
        dim_forced_const = true;
      }
    }

    const int mode_v = mode;
    const int dim_v = dim_eff;

    // buffer lock（保持原本行為）
    lock_bufs({BUF_VEC_IN, BUF_VEC_W, BUF_VEC_OUT});

    // drive ports
    vector_mode.write(mode_v);
    vector_DIM.write(dim_v);

    // log
    {
      std::ostringstream oss;
      oss << sc_core::sc_time_stamp() << " [controller] VECTOR"
          << " mode=" << vec_mode_str_(mode_v) << "(" << mode_v << ")"
          << " dim=" << dim << " dim_eff=" << dim_v;

      if (dim_from_tp)
        oss << " (from token_pos)";
      if (dim_forced_const)
        oss << " (forced=10 by MODE=" << MODE << ")";

      oss << " token_pos=" << tp;

      if (waited > 0)
        oss << " waited=" << waited << "cy";

      oss << " lock={" << buf_name_(BUF_VEC_IN) << "," << buf_name_(BUF_VEC_W)
          << "," << buf_name_(BUF_VEC_OUT) << "}";

      SC_REPORT_INFO(this->name(), oss.str().c_str());
    }

    // start pulse
    vector_reserved_ = true;
    vector_start.write(true);
    wait_one_cycle();
    vector_start.write(false);
    wait(sc_core::SC_ZERO_TIME); // delta-settle
  }

  void issue_array_nowait_isa(const ArrayInstr &a) {
    // 1) Array module 必須 idle（避免你提到的「done thread 清鎖」時序陰影）
    while (Array_busy.read() || array_reserved_ || Array_done.read()) {
      wait_one_cycle();
    }

    if ((a.op == ArrayOp::IS || a.op == ArrayOp::OS) &&
        (a.row <= 0 || a.row > CAP_ARRAY_IN || a.col <= 0 ||
         a.col > CAP_ARRAY_OUT)) {
      std::ostringstream oss;
      oss << "ARRAY(IS/OS): shape exceeds config capacity "
          << "row=" << a.row << " cap_in=" << CAP_ARRAY_IN << " col=" << a.col
          << " cap_out=" << CAP_ARRAY_OUT;
      SC_REPORT_ERROR(name(), oss.str().c_str());
      return;
    }
    // 2) buffer hazard
    std::vector<int> bufs = {BUF_ARRAY_IN, BUF_ARRAY_OUT};
    const int tp = token_pos_1b();
    const bool odd = (tp & 1);
    if (odd && a.op == ArrayOp::QK)
      bufs.push_back(BUF_K);
    if (odd && a.op == ArrayOp::PV)
      bufs.push_back(BUF_V);

    while (true) {
      // (i) 先鎖 BUF（不成功就等）
      if (!try_lock_bufs(bufs)) {
        wait_one_cycle();
        continue;
      }

      // (ii) 再嘗試鎖 DRAM（不成功就把 BUF 放掉再等）
      if (!try_acquire_dram(DramOwner::ARRAY)) {
        mark_bufs_idle(bufs);
        wait_one_cycle();
        continue;
      }

      // 成功：BUF + DRAM 都拿到了
      current_array_bufs_ = bufs;
      break;
    }

    // 4) Drive ISA ports
    Array_op.write(array_op_u2(a.op));

    ArrayKey key{};
    key.layer = a.layer;

    // ★★★ NEW: 用 local 變數保存要寫出去的 row/col，避免 sc_out.read() 讀到舊值
    int row_v = 0;
    int col_v = 0;

    if (a.op == ArrayOp::IS || a.op == ArrayOp::OS) {
      key.set_weight_name(a.weight_name);
      key.slice_idx = a.slice_idx;
      key.head = 0;

      row_v = a.row;
      col_v = a.col;
    } else {
      key.set_weight_name("");
      key.slice_idx = 0;
      key.head = static_cast<sc_dt::sc_uint<8>>(a.kv_head);

      if (a.op == ArrayOp::QK) {
        row_v = KV_LEN_DEFAULT; // 64
        col_v = tp;
      } else { // PV
        row_v = tp;
        col_v = KV_LEN_DEFAULT; // 64
      }
    }

    Array_row.write(row_v);
    Array_col.write(col_v);
    Array_key.write(key);

    // log（★用 row_v/col_v，不用 Array_row.read()/Array_col.read()）
    {
      std::ostringstream oss;
      oss << sc_core::sc_time_stamp() << " [controller] " << a.to_string()
          << " token_pos=" << tp << (odd ? " (odd)" : " (even)")
          << " row=" << row_v << " col=" << col_v;
      if (odd && a.op == ArrayOp::QK)
        oss << " +lock BUF_K";
      if (odd && a.op == ArrayOp::PV)
        oss << " +lock BUF_V";
      SC_REPORT_INFO(this->name(), oss.str().c_str());
    }

    // 5) start pulse
    array_reserved_ = true;
    Array_start.write(true);
    wait_one_cycle();
    Array_start.write(false);
    wait(sc_core::SC_ZERO_TIME); // ✅ delta-settle
  }

  // =========================================================================
  // DRAM：只給 DRAM READ/WRITE 用的封裝
  // =========================================================================
  void kick_dram_nowait(const DRAMInstr &m) {
    // 防呆：避免同時多個 bool=true
    const int typed_cnt = (int)m.is_rope + (int)m.is_model + (int)m.is_rms_att +
                          (int)m.is_rms_ffn + (int)m.is_k + (int)m.is_v +
                          (int)m.is_token_input + (int)m.is_token_output +
                          (int)m.is_Compiler;

    if (typed_cnt > 1) {
      SC_REPORT_ERROR(name(),
                      "DRAMInstr: multiple type flags are true (ambiguous)");
      return;
    }

    // ==========================
    // 1) ROPE read：src_offset 用 token_pos
    // DRAM, READ, ROPE, dst_buf, dst_offset, len
    // ==========================
    if (m.is_rope) {
      if (!m.is_read) {
        SC_REPORT_ERROR(name(), "RoPE only supports READ");
        return;
      }
      const int pos = token_pos_1b(); // ★ token snapshot
      kick_dma_nowait(tcore_sel::SEL_ROPE, pos, m.buf_index, m.buf_offset,
                      m.len_elems,
                      /*kv_head=*/0, /*kv_is_prev=*/false, /*kv_layer=*/-1,
                      /*model_weight_name=*/"", /*model_slice_idx=*/0);
      return;
    }

    // ==========================
    // 2) MODEL weight read
    // DRAM, READ, model, "weight", dst_buf, dst_offset, layer, len
    // ==========================
    if (m.is_model) {
      if (!m.is_read) {
        SC_REPORT_ERROR(name(), "MODEL only supports READ");
        return;
      }

      const std::uint32_t wid = fnv1a_32(m.weight_name);

      // ★ 使用 SEL_MODEL：src_offset 帶 hash(weight_name)，layer 走
      // dma_kv_layer[ch] ★ 注意：hash 可能讓 int 變負值，但 bit pattern
      // 仍保留；DMA端用 (uint32_t)cast 回來即可
      kick_dma_nowait(
          tcore_sel::SEL_MODEL, static_cast<int>(wid), m.buf_index,
          m.buf_offset, m.len_elems,
          /*kv_head=*/0,
          /*kv_is_prev=*/false,
          /*kv_layer=*/m.layer_id, // layer 給 kv_layer port + mk.layer
          /*model_weight_name=*/m.weight_name,
          /*model_slice_idx=*/m.slice_idx);

      std::ostringstream oss;
      oss << sc_core::sc_time_stamp() << " [controller] MODEL READ"
          << " weight=\"" << m.weight_name << "\""
          << " hash=0x" << std::hex << wid << std::dec
          << " layer=" << m.layer_id << " -> buf=" << m.buf_index
          << " buf_offset=" << m.buf_offset << " slice=" << m.slice_idx
          << " len=" << m.len_elems;
      SC_REPORT_INFO(this->name(), oss.str().c_str());

      return;
    }
    // ==========================
    // 3) TOKEN I/O：token_pos 當 index
    // DRAM, READ/WRITE, token_I/token_O, buf, buf_offset, len
    //   - token_I  => SEL_TOKEN_IN
    //   - token_O  => SEL_TOKEN_OUT
    // ==========================
    if (m.is_token_input || m.is_token_output) {
      const int tok = token_pos_1b();
      const int sel =
          m.is_token_input ? tcore_sel::SEL_TOKEN_IN : tcore_sel::SEL_TOKEN_OUT;

      // Enforce direction:
      //   TOKEN_I : DRAM -> SRAM only
      //   TOKEN_O : SRAM -> DRAM only
      if (m.is_token_input && !m.is_read) {
        SC_REPORT_ERROR(name(), "TOKEN_I only supports READ (DRAM -> SRAM)");
        return;
      }
      if (m.is_token_output && m.is_read) {
        SC_REPORT_ERROR(name(), "TOKEN_O only supports WRITE (SRAM -> DRAM)");
        return;
      }

      // External TOKEN_I/O file element offset.
      //
      // New format:
      //   use explicit m.token_io_offset.
      //
      // Old format:
      //   MODE 0 keeps old token_pos-based behavior.
      //   MODE 1/2/3/4/5/6 keeps old flat-file behavior, offset = 0.
      int token_file_offset = 0;

      if (m.has_token_io_offset) {
        token_file_offset = m.token_io_offset;
      } else {
        if constexpr (MODE == 0) {
          token_file_offset = tok; // legacy token slot, 1-based
        } else {
          token_file_offset = 0; // legacy flat file behavior
        }
      }

      if (m.is_read) {
        // external TOKEN_I file -> local buffer
        // src_offset carries TOKEN_I file offset.
        kick_dma_nowait(sel, token_file_offset, m.buf_index, m.buf_offset,
                        m.len_elems);
      } else {
        // local buffer -> external TOKEN_O file
        // dst_offset carries TOKEN_O file offset.
        kick_dma_nowait(m.buf_index, m.buf_offset, sel, token_file_offset,
                        m.len_elems);
      }
      return;
    }

    // ==========================
    // 4) COMPILER：用 m.row 當 dram_offset（decode 時塞進去）
    // DRAM, R/W, COMPILER, buf, buf_offset, dram_offset, len
    // ==========================
    if (m.is_Compiler) {
      const int raw_off = m.row;
      const int off = resolve_compiler_offset_checked(raw_off, m.len_elems);

      if (m.is_read) {
        kick_dma_nowait(tcore_sel::SEL_COMPILER, off, m.buf_index, m.buf_offset,
                        m.len_elems);
      } else {
        kick_dma_nowait(m.buf_index, m.buf_offset, tcore_sel::SEL_COMPILER, off,
                        m.len_elems);
      }
      return;
    }

    // ==========================
    // 5) KV K/V：用 pack(layer,head) 當 idx，token_pos 由 DRAM 模組自己用
    // DRAM, READ/WRITE, K/V, buf, buf_offset, layer, kv_head, len
    // ==========================
    if (m.is_k || m.is_v) {
      const int sel = m.is_k ? tcore_sel::SEL_KV_K : tcore_sel::SEL_KV_V;

      if (m.is_read) {
        SC_REPORT_ERROR(name(), "DRAM READ K/V not supported yet (KVFILE->BUF "
                                "not implemented in DMA_Controller)");
        return;
      } else {
        kick_dma_nowait(m.buf_index, m.buf_offset, sel, /*dst_offset=*/0,
                        m.len_elems, m.kv_head, /*kv_is_prev=*/false,
                        /*kv_layer=*/m.layer_id);
      }
      return;
    } else {
      SC_REPORT_FATAL(
          name(),
          "DRAM: unknown instruction type (missing or multiple type flags)");
      return;
    }
  }

  // =========================================================================
  // DMA：啟動並不會等待完成（non-blocking）
  // =========================================================================
  void kick_dma_nowait(int src_sel, int src_offset, int dst_sel, int dst_offset,
                       int length, int kv_head = 0, bool kv_is_prev = false,
                       int kv_layer = -1,
                       const std::string &model_weight_name = "",
                       int model_slice_idx = 0) {

    bool use_dram = tcore_sel::is_external_sel(src_sel) ||
                    tcore_sel::is_external_sel(dst_sel);

    std::vector<int> used_bufs;
    if (tcore_sel::is_local_buf_sel(src_sel))
      used_bufs.push_back(src_sel);
    if (tcore_sel::is_local_buf_sel(dst_sel))
      used_bufs.push_back(dst_sel);

    // ★ 放這裡：把重複的 buf 去掉（避免 {b,b} 造成 mark_bufs_busy 自己撞到）
    if (used_bufs.size() >= 2) {
      std::sort(used_bufs.begin(), used_bufs.end());
      used_bufs.erase(std::unique(used_bufs.begin(), used_bufs.end()),
                      used_bufs.end());
    }

    if (use_dram) {
      while (Array_busy.read() || array_reserved_) {
        wait_one_cycle();
      }
    }
    // wait_bufs_idle(used_bufs);
    // mark_bufs_busy(used_bufs);
    lock_bufs(used_bufs); // 確保後續不會有其他指令搶走這些 buf
    int ch = alloc_dma_channel();

    current_dma_bufs_[ch] = used_bufs;
    current_dma_uses_dram_[ch] = use_dram;
    const bool is_kv =
        (src_sel == tcore_sel::SEL_KV_K) || (src_sel == tcore_sel::SEL_KV_V) ||
        (dst_sel == tcore_sel::SEL_KV_K) || (dst_sel == tcore_sel::SEL_KV_V);

    int kv_layer_eff = (kv_layer >= 0) ? kv_layer : 0;

    if (!is_kv) {
      kv_head = 0;
      kv_is_prev = false;
      // ★ 不要把 kv_layer_eff 強制洗成 0，MODEL 會需要它
    } else {
      // kv_layer 必須提供；若沒提供，先用 legacy offset 推回去（過渡期保命）
      if (kv_layer < 0) {
        if (src_sel == tcore_sel::SEL_KV_K || src_sel == tcore_sel::SEL_KV_V) {
          kv_layer_eff = src_offset;
        } else {
          kv_layer_eff = dst_offset;
        }
        SC_REPORT_WARNING(
            this->name(),
            "kick_dma_nowait(): kv_layer not provided; using legacy "
            "offset mapping as kv_layer");
      }

      if (kv_head < 0 || kv_head > 7) {
        SC_REPORT_ERROR(
            this->name(),
            "kick_dma_nowait(): kv_head out of expected range (0..7)");
      }
      if (kv_layer_eff < 0) {
        SC_REPORT_ERROR(this->name(),
                        "kick_dma_nowait(): kv_layer_eff < 0 (invalid)");
        kv_layer_eff = 0;
      }
    }

    if (use_dram) {
      // 用 owner-based DRAM lock：確保一次只有一筆 DRAM 交易（且與 Array 互斥）
      acquire_dram(DramOwner::DMA, ch);
    }

    dma_src_sel[ch].write(static_cast<sc_dt::sc_uint<5>>(src_sel));
    dma_dst_sel[ch].write(static_cast<sc_dt::sc_uint<5>>(dst_sel));
    dma_src_offset[ch].write(src_offset);
    dma_dst_offset[ch].write(dst_offset);
    dma_length[ch].write(length);
    dma_kv_head[ch].write(kv_head);
    dma_kv_is_prev[ch].write(kv_is_prev);

    dma_kv_layer[ch].write(kv_layer_eff); // ★ NEW：獨立 kv_layer port
    const bool is_model =
        (src_sel == tcore_sel::SEL_MODEL) || (dst_sel == tcore_sel::SEL_MODEL);

    // kick_dma_nowait(...) 裡面，建立 mk 的地方
    ArrayKey mk{};
    mk.head = 0;

    if (is_model) {
      mk.layer = static_cast<sc_dt::sc_uint<8>>(
          (kv_layer_eff >= 0) ? kv_layer_eff : 0);
      mk.slice_idx = model_slice_idx; // ★★★補上這行
      mk.set_weight_name(model_weight_name);
    } else {
      mk.layer = 0; // 建議清乾淨避免殘留誤判
      mk.slice_idx = 0;
      mk.set_weight_name("");
    }

    dma_model_key[ch].write(mk);
    std::ostringstream oss;
    if (src_sel == tcore_sel::SEL_ROPE) {
      oss << sc_core::sc_time_stamp() << " [controller] ROPE READ"
          << " (ch=" << ch << ")"
          << " pos=" << src_offset << " -> buf=" << dst_sel
          << " buf_offset=" << dst_offset << " elems=" << length;
    } else if (src_sel == tcore_sel::SEL_KV_K ||
               src_sel == tcore_sel::SEL_KV_V) {
      oss << sc_core::sc_time_stamp() << " [controller] KV READ"
          << " (" << (src_sel == tcore_sel::SEL_KV_K ? "K" : "V") << ")"
          << " (ch=" << ch << ")"
          << " layer=" << kv_layer_eff << " kv_head=" << kv_head
          << " slot=" << (kv_is_prev ? "prev(tp-1)" : "cur(tp)")
          << " -> buf=" << dst_sel << " buf_offset=" << dst_offset
          << " len=" << length;
    } else if (dst_sel == tcore_sel::SEL_KV_K ||
               dst_sel == tcore_sel::SEL_KV_V) {
      oss << sc_core::sc_time_stamp() << " [controller] KV WRITE"
          << " (" << (dst_sel == tcore_sel::SEL_KV_K ? "K" : "V") << ")"
          << " (ch=" << ch << ")"
          << " buf=" << src_sel << " buf_offset=" << src_offset
          << " -> layer=" << kv_layer_eff << " kv_head=" << kv_head
          << " slot=" << (kv_is_prev ? "prev(tp-1)" : "cur(tp)")
          << " len=" << length;
    } else {
      oss << sc_core::sc_time_stamp() << " [controller] DMA start"
          << " (ch=" << ch << ")"
          << " src_sel=" << src_sel << " dst_sel=" << dst_sel
          << " src_offset=" << src_offset << " dst_offset=" << dst_offset
          << " length=" << length;
    }
    SC_REPORT_INFO(this->name(), oss.str().c_str());

    dma_reserved_[ch] = true;
    dma_start[ch].write(true);
    wait_one_cycle();
    dma_start[ch].write(false);

    // ✅ delta-settle：讓 TB/其他 thread 在同 timestamp 觸發的更新先落地
    wait(sc_core::SC_ZERO_TIME);

    // ★ buf_busy_ / dram_busy_ 的釋放交給 track_dma_done()
  }

  // =========================================================================
  // Router：Multicast SEND（真正語意：一筆資料同時送多個目的地）
  // =========================================================================
  void issue_router_mcast_nowait(const NocMcastInstr &nm) {
    const int srcb = nm.src_buf;
    const int num = nm.num;

    if (num <= 0 || num > MAX_MCAST_DESTS ||
        static_cast<int>(nm.dests.size()) != num) {
      SC_REPORT_ERROR(name(), "NoC multicast: invalid num/dests (exceed "
                              "MAX_MCAST_DESTS or size mismatch)");
      return;
    }

    // wait_bufs_idle({srcb});
    lock_bufs({srcb}); // 確保後續不會有其他指令搶走這個 buf
    wait_router_send_idle_with_reservation();

    clear_router_mcast_outputs();

    if (!valid_buf_idx(nm.src_buf)) {
      std::ostringstream oss;
      oss << "Router MCAST: src_buf out of range (0.." << (NUM_BUFS - 1) << ")";
      SC_REPORT_ERROR(name(), oss.str().c_str());
      return;
    }
    if (nm.src_buf == BUF_ARRAY_IN || nm.src_buf == BUF_VEC_IN ||
        nm.src_buf == BUF_VEC_W) {
      SC_REPORT_ERROR(name(),
                      "Router MCAST: src_buf cannot be read-only buffers");
    }

    Router_is_send.write(true);
    Router_is_mcast.write(true);

    Router_src_buf.write(nm.src_buf);
    Router_src_offset.write(nm.src_offset);
    Router_length.write(nm.len);

    if constexpr (MODE >= 1 && MODE <= 5) {
      Router_hash.write(0);
    } else {
      Router_hash.write(nm.hash);
    }

    Router_dst_x.write(nm.dests[0].dst_x);
    Router_dst_y.write(nm.dests[0].dst_y);
    Router_dst_buf.write(nm.dests[0].dst_buf);
    Router_dst_offset.write(nm.dests[0].dst_offset);

    // ★ mcast fanout can be up to 64; keep full 7-bit width to avoid truncation
    Router_mcast_num.write(static_cast<sc_dt::sc_uint<7>>(num));
    for (int i = 0; i < MAX_MCAST_DESTS; ++i) {
      if (i < num) {
        Router_mcast_dst_x[i].write(
            static_cast<sc_dt::sc_uint<5>>(nm.dests[i].dst_x));
        Router_mcast_dst_y[i].write(
            static_cast<sc_dt::sc_uint<5>>(nm.dests[i].dst_y));
        Router_mcast_dst_buf[i].write(
            static_cast<sc_dt::sc_uint<4>>(nm.dests[i].dst_buf));
        Router_mcast_dst_offset[i].write(
            static_cast<sc_dt::sc_uint<16>>(nm.dests[i].dst_offset));
      } else {
        Router_mcast_dst_x[i].write(0);
        Router_mcast_dst_y[i].write(0);
        Router_mcast_dst_buf[i].write(0);
        Router_mcast_dst_offset[i].write(0);
      }
    }

    {
      std::ostringstream oss;
      oss << sc_core::sc_time_stamp() << " [controller] Router MCAST SEND"
          << " src_buf=" << nm.src_buf << " src_offset=" << nm.src_offset
          << " len=" << nm.len << " fanout=" << num;
      for (int i = 0; i < num; ++i) {
        oss << " ->(" << nm.dests[i].dst_x << "," << nm.dests[i].dst_y << ")"
            << " dst_buf=" << nm.dests[i].dst_buf
            << " dst_offset=" << nm.dests[i].dst_offset;
      }
      SC_REPORT_INFO(this->name(), oss.str().c_str());
    }

    // mark_bufs_busy({srcb});
    current_router_send_buf_ = srcb;
    router_send_reserved_ = true;

    Router_start.write(true);
    wait_one_cycle();
    Router_start.write(false);
    wait(sc_core::SC_ZERO_TIME); // ✅ delta-settle
  }

  // =========================================================================
  // Router SEND / RECV
  // =========================================================================
  void issue_router_and_wait(const NocInstr &n) {

    // ===============================
    // CASE A: Router RECEIVE（本 Tcore 當 receiver）
    // ===============================
    if (n.is_receive) {
      wait_router_recv_idle_with_reservation();

      std::ostringstream oss;
      oss << sc_core::sc_time_stamp()
          << " [controller] Router RECEIVE hash=" << n.hash
          << " (may overlap with A/V/DMA)";
      SC_REPORT_INFO(this->name(), oss.str().c_str());

      Router_is_send.write(false);
      clear_router_mcast_outputs();
      if constexpr (MODE >= 1 && MODE <= 5) {
        Router_hash.write(0);
      } else {
        Router_hash.write(n.hash);
      }

      Router_src_buf.write(0);
      Router_src_offset.write(0);
      Router_dst_buf.write(0);
      Router_dst_offset.write(0);
      Router_length.write(0);
      Router_dst_x.write(0);
      Router_dst_y.write(0);

      router_recv_reserved_ = true;

      Router_start.write(true);
      wait_one_cycle();
      Router_start.write(false);
      wait(sc_core::SC_ZERO_TIME); // ✅ delta-settle
      while (!router_recv_grant_mirror_.read()) {
        wait_one_cycle();
      }

      return;
    }

    // ===============================
    // CASE B: SEND
    // ===============================
    {
      int srcb = n.src_buf;

      // wait_bufs_idle({srcb});
      lock_bufs({srcb}); // 確保後續不會有其他指令搶走這個 buf
      wait_router_send_idle_with_reservation();

      clear_router_mcast_outputs();
      Router_is_send.write(true);
      Router_src_buf.write(n.src_buf);
      Router_src_offset.write(n.src_offset);
      Router_dst_x.write(n.dst_x);
      Router_dst_y.write(n.dst_y);
      Router_dst_buf.write(n.dst_buf);
      Router_dst_offset.write(n.dst_offset);
      Router_length.write(n.len);
      if constexpr (MODE >= 1 && MODE <= 5) {
        Router_hash.write(0);
      } else {
        Router_hash.write(n.hash);
      }

      if (n.src_buf < 0 || n.src_buf > NUM_BUFS - 1 || n.dst_buf < 0 ||
          n.dst_buf > NUM_BUFS - 1) {
        std::ostringstream oss;
        oss << "Router SEND: src_buf/dst_buf out of range (0.."
            << (NUM_BUFS - 1) << ")";
        SC_REPORT_ERROR(name(), oss.str().c_str());
      }
      // if (n.src_buf == BUF_ARRAY_IN || n.src_buf == BUF_VEC_IN ||
      //     n.src_buf == BUF_VEC_W) {
      //   SC_REPORT_ERROR(name(),
      //                   "Router SEND: src_buf cannot be read-only buffers");
      // }
      std::ostringstream oss;
      oss << sc_core::sc_time_stamp() << " [controller] Router SEND "
          << " src_buf=" << n.src_buf << " src_offset=" << n.src_offset
          << " -> (" << n.dst_x << "," << n.dst_y << ")"
          << " dst_buf=" << n.dst_buf << " dst_offset=" << n.dst_offset
          << " len=" << n.len << " hash=" << n.hash;
      SC_REPORT_INFO(this->name(), oss.str().c_str());

      // mark_bufs_busy({srcb});
      current_router_send_buf_ = srcb;
      router_send_reserved_ = true;

      Router_start.write(true);
      wait_one_cycle();
      Router_start.write(false);
    }
  }

  // =========================================================================
  // run() 主流程
  // =========================================================================
  void run() {
    Array_start.write(false);
    Array_op.write(0);
    // ===== Vector outputs init (避免 X/垃圾值) =====
    vector_start.write(false);
    vector_mode.write(0);
    vector_DIM.write(0);
    ArrayKey k0{};
    k0.set_weight_name("");
    k0.layer = 0;
    k0.slice_idx = 0;
    k0.head = 0;
    Array_key.write(k0);

    Array_row.write(0);
    Array_col.write(0);

    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      dma_start[ch].write(false);
      dma_src_sel[ch].write(0);
      dma_dst_sel[ch].write(0);
      dma_src_offset[ch].write(0);
      dma_dst_offset[ch].write(0);
      dma_length[ch].write(0);
      dma_kv_head[ch].write(0);
      dma_kv_is_prev[ch].write(false);
      // ★ 新增

      dma_kv_layer[ch].write(0); // ★ NEW
      ArrayKey mk{};
      mk.set_weight_name("");
      mk.layer = 0;
      mk.slice_idx = 0;
      mk.head = 0;
      dma_model_key[ch].write(mk);
    }

    Router_start.write(false);
    Router_is_send.write(false);
    Router_src_buf.write(0);
    Router_dst_buf.write(0);
    Router_src_offset.write(0);
    Router_dst_offset.write(0);
    Router_length.write(0);
    Router_dst_x.write(0);
    Router_dst_y.write(0);
    Router_hash.write(0);

    // Multicast ports init
    clear_router_mcast_outputs();

    all_done.write(false);
    wait_one_cycle(); // ✅ 很重要：確保所有 init 值落地、且對齊 clk
    std::ifstream fin(csv_path_);
    if (!fin.is_open()) {
      SC_REPORT_ERROR(name(), ("Cannot open CSV: " + csv_path_).c_str());
      while (true)
        wait_one_cycle();
    }

    std::string line;
    unsigned lineno = 0;

    while (std::getline(fin, line)) {
      ++lineno;
      auto s = trim(line);
      if (s.empty() || s[0] == '#')
        continue;

      Decoded inst{};
      if (!decode_line(s, inst)) {
        std::stringstream oss;
        oss << "Decode failed at line " << lineno << ": " << s;
        SC_REPORT_WARNING(name(), oss.str().c_str());
        SC_REPORT_ERROR(name(), "ISA decode error");
        continue;
      }

      switch (inst.kind) {
      case Kind::VECTOR: {
        issue_vector_nowait(inst.v.op, inst.v.dim);
        wait_n_cycles(3);
        break;
      }
      case Kind::ARRAY: {
        issue_array_nowait_isa(inst.a);
        wait_n_cycles(3);
        break;
      }

      case Kind::DRAM: {
        kick_dram_nowait(inst.m);
        wait_n_cycles(3);
        break;
      }
      case Kind::DMA: {
        kick_dma_nowait(inst.d.src_buf, inst.d.src_offset, inst.d.dst_buf,
                        inst.d.dst_offset, inst.d.len);
        wait_n_cycles(3);
        break;
      }

      case Kind::NOC_MCAST: {
        issue_router_mcast_nowait(inst.nm);
        wait_n_cycles(3);
        break;
      }

      case Kind::NOC: {
        issue_router_and_wait(inst.n);
        wait_n_cycles(3);
        break;
      }
      case Kind::KV: {
        issue_kv_write(inst.kv);
        wait_n_cycles(3);
        break;
      }

      default:
        break;
      }
    }

    do {
      wait_one_cycle();
    } while (!everything_idle());
    std::ostringstream oss;
    oss << sc_core::sc_time_stamp() << " [controller] ALL DONE"
        << " A_busy=" << Array_busy.read() << " V_busy=" << Vector_busy.read();

    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      oss << " dma" << ch << "_busy=" << dma_busy[ch].read();
    }

    oss << " Rs_busy=" << Router_send_busy.read()
        << " Rr_busy=" << Router_recv_busy.read()
        << " dram_busy=" << (dram_busy() ? 1 : 0);

    for (int b = 0; b < NUM_BUFS; ++b) {
      oss << " buf" << b << "=" << buf_busy_[b];
    }
    SC_REPORT_INFO(this->name(), oss.str().c_str());

    all_done.write(true);

    while (true)
      wait_one_cycle();
  }
};

#endif // CORE_CONTROLLER_H
