#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <ostream>
#include <string>
#include <systemc.h>
namespace kvfile {

// Fixed KV spec (must match DMA/Array)
static constexpr int NUM_LAYERS = 16;
static constexpr int NUM_HEADS = 8;
static constexpr int HEAD_DIM = 64; // FP32 dim
static constexpr int MAX_SEQ = 128; // token_pos 1..128

using scalar_t = float;
static constexpr std::uint64_t ELEM_BYTES = sizeof(scalar_t);

static constexpr std::uint64_t FILE_BYTES =
    static_cast<std::uint64_t>(MAX_SEQ) *
    static_cast<std::uint64_t>(NUM_HEADS) *
    static_cast<std::uint64_t>(HEAD_DIM) * ELEM_BYTES;

static constexpr const char *DEFAULT_ROOT_DIR = "../DRAM_json";

inline std::string make_relpath(bool is_k, int layer) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%c/L%02d.bin", is_k ? 'K' : 'V', layer);
  return std::string(buf);
}

inline std::string make_path(const std::string &root_dir, bool is_k,
                             int layer) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s/%c/L%02d.bin", root_dir.c_str(),
                is_k ? 'K' : 'V', layer);
  return std::string(buf);
}

inline std::uint64_t byte_offset_vec(int token_pos_1b, int head) {
  if (token_pos_1b <= 0 || token_pos_1b > MAX_SEQ || head < 0 ||
      head >= NUM_HEADS) {
    SC_REPORT_ERROR("kvfile", "byte_offset_vec: token/head out of range");
    return std::numeric_limits<std::uint64_t>::max();
  }
  const std::uint64_t token_slot = static_cast<std::uint64_t>(token_pos_1b - 1);
  const std::uint64_t entry =
      token_slot * static_cast<std::uint64_t>(NUM_HEADS) +
      static_cast<std::uint64_t>(head);
  return entry * static_cast<std::uint64_t>(HEAD_DIM) * ELEM_BYTES;
}

} // namespace kvfile
// ===============================
// Backward-compatible aliases (global)
// ===============================
static constexpr int KV_NUM_LAYERS = kvfile::NUM_LAYERS;
static constexpr int KV_NUM_HEADS = kvfile::NUM_HEADS;
static constexpr int KV_HEAD_DIM = kvfile::HEAD_DIM;
static constexpr int KV_MAX_SEQ = kvfile::MAX_SEQ;

static constexpr std::uint64_t KV_ELEM_BYTES = kvfile::ELEM_BYTES;
static constexpr std::uint64_t KV_FILE_BYTES = kvfile::FILE_BYTES;
// ============================================================
// DMA / DRAM endpoint selector encoding (shared by Controller/DMA)
// ============================================================
// 0..(SEL_EXT_BASE-1): local buffers
// SEL_EXT_BASE..31    : external endpoints (ROM/FILE/MEM)
// NOTE: src_sel/dst_sel are sc_uint<5> => max 31
namespace tcore_sel {

static constexpr int SEL_EXT_BASE = 24;

// ---- External endpoints (24..31) ----
static constexpr int SEL_COMPILER = 24;  // compiler "3d-dram"
static constexpr int SEL_TOKEN_IN = 25;  // token IN
static constexpr int SEL_TOKEN_OUT = 26; // token OUT
static constexpr int SEL_STATE =
    27; // STATE file (for saving/restoring Core state)

// File-backed KV endpoints (read/write via DMA)  ★跟你 DMA KVFILE case 對齊用
static constexpr int SEL_KV_V = 28; // KV(V) file
static constexpr int SEL_KV_K = 29; // KV(K) file

static constexpr int SEL_ROPE = 30; // RoPE ROM
static constexpr int SEL_MODEL = 31;

// ---- helpers ----
inline bool is_external_sel(int s) { return (s >= SEL_EXT_BASE && s <= 31); }
inline bool is_local_buf_sel(int s) { return (s >= 0 && s < 12); }
inline bool is_kvfile_sel(int s) { return (s == SEL_KV_K) || (s == SEL_KV_V); }

} // namespace tcore_sel
// ============================================================
// DRAM row-based requests
// ============================================================
// 語意：請求 [row, row + rows) 這個「列區間」
// 每一列都含有完整的 COLS 個元素（由 DRAM 的模板參數提供）
template <typename T> struct MemRowReadReq {
  std::uint32_t row;  // 起始列（0-based）
  std::uint32_t rows; // 要讀多少列（每列完整 COLS 個元素）
};

template <typename T> struct MemRowWriteReq {
  std::uint32_t row;  // 起始列（0-based）
  std::uint32_t rows; // 要寫多少列（每列完整 COLS 個元素）
};

// --- 為 SystemC 的 sc_fifo << 友好，定義可列印 ---
template <typename T>
inline std::ostream &operator<<(std::ostream &os, const MemRowReadReq<T> &r) {
  os << "{row=" << r.row << ", rows=" << r.rows << "}";
  return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const MemRowWriteReq<T> &r) {
  os << "{row=" << r.row << ", rows=" << r.rows << "}";
  return os;
}

// ============================================================
// NoC command types
// ============================================================

// ★ 統一 multicast 上限（要跟 Core_Controller / Router 對齊）
// 你現在 Core_Controller / Router 都用 64，我這裡也設 64。
static constexpr int NOC_MCAST_MAX_DESTS = 64;

// multicast 目的地項目
struct NocMcastDest {
  sc_dt::sc_uint<5> dst_x;
  sc_dt::sc_uint<5> dst_y;
  sc_dt::sc_uint<4> dst_buf;     // 0..11
  sc_dt::sc_uint<16> dst_offset; // 起始 index
};

inline std::ostream &operator<<(std::ostream &os, const NocMcastDest &d) {
  os << "{(" << d.dst_x << "," << d.dst_y << ")"
     << " dst_buf=" << d.dst_buf << " dst_offset=" << d.dst_offset << "}";
  return os;
}

// NoC cmd：同時支援 unicast 與 multicast
// - unicast: is_mcast=false，使用 dst_x/dst_y/dst_buf/dst_offset
// - multicast: is_mcast=true，mcast_num>0，使用 mcast_dsts[0..mcast_num-1]
template <typename T> struct NocCmd {
  // ---- src coordinate ----
  sc_dt::sc_uint<5> src_x;
  sc_dt::sc_uint<5> src_y;

  // ---- unicast dst coordinate ----
  sc_dt::sc_uint<5> dst_x;
  sc_dt::sc_uint<5> dst_y;

  // ---- buffer selection ----
  sc_dt::sc_uint<4> src_buf; // 0..11 對應 Tcore 裡的 bufs[]
  sc_dt::sc_uint<4> dst_buf;

  // ---- indices & length ----
  sc_dt::sc_uint<16> src_offset; // 起始 index（0..2047 或 0..511）
  sc_dt::sc_uint<16> dst_offset;
  sc_dt::sc_uint<16> length; // 要搬多少個 T 元素
  sc_dt::sc_uint<32> hash = 0;
  // ---- multicast extension ----
  bool is_mcast{false};           // 預設 unicast
  sc_dt::sc_uint<7> mcast_num{0}; // fanout（<= NOC_MCAST_MAX_DESTS）
  std::array<NocMcastDest, NOC_MCAST_MAX_DESTS> mcast_dsts{};
};

// --- 為 SystemC 的 sc_fifo << 友好，定義可列印 ---
template <typename T>
inline std::ostream &operator<<(std::ostream &os, const NocCmd<T> &cmd) {
  os << "NocCmd{"
     << "src=(" << cmd.src_x << "," << cmd.src_y << "), "
     << "src_buf=" << cmd.src_buf << ", src_offset=" << cmd.src_offset
     << ", length=" << cmd.length;

  if (!cmd.is_mcast) {
    os << ", dst=(" << cmd.dst_x << "," << cmd.dst_y << ")"
       << ", dst_buf=" << cmd.dst_buf << ", dst_offset=" << cmd.dst_offset;
  } else {
    os << ", MCAST num=" << cmd.mcast_num << " [";
    const int n = static_cast<int>(cmd.mcast_num.to_uint());
    for (int i = 0; i < n && i < NOC_MCAST_MAX_DESTS; ++i) {
      if (i)
        os << ", ";
      os << cmd.mcast_dsts[i];
    }
    os << "]";
  }

  os << "}";
  return os;
}

// ============================================================
// Array command key (weight_name / layer / slice_idx / head)
// ============================================================
// 用 fixed-size char array 來表示字串，才能穩定走 SystemC port
static constexpr int ARRAY_WEIGHT_NAME_MAX = 64;

struct ArrayKey {
  sc_dt::sc_uint<8> layer;
  sc_dt::sc_uint<16> slice_idx;
  sc_dt::sc_uint<8> head{0};

  char weight_name[ARRAY_WEIGHT_NAME_MAX]{};

  inline void set_weight_name(const std::string &s) {
    std::memset(weight_name, 0, sizeof(weight_name));
    std::strncpy(weight_name, s.c_str(), sizeof(weight_name) - 1);
  }

  // ✅ 改成 static，get_weight_name() const 才能呼叫
  static inline std::size_t safe_strnlen_(const char *s, std::size_t maxn) {
    std::size_t n = 0;
    while (n < maxn && s[n] != '\0')
      ++n;
    return n;
  }

  inline std::string get_weight_name() const {
    return std::string(weight_name,
                       safe_strnlen_(weight_name, sizeof(weight_name)));
  }
};

inline void sc_trace(sc_core::sc_trace_file *tf, const ArrayKey &v,
                     const std::string &n) {
  sc_trace(tf, v.layer, n + ".layer");
  sc_trace(tf, v.slice_idx, n + ".slice_idx");
  sc_trace(tf, v.head, n + ".head");

  // weight_name 是 char array，不好在 VCD 直接當字串看
  // 先不要 trace 它（需要的話看下面進階作法）
}

// sc_signal<T> 常用：提供 == / !=，避免比較問題
inline bool operator==(const ArrayKey &a, const ArrayKey &b) {
  return (std::strncmp(a.weight_name, b.weight_name, ARRAY_WEIGHT_NAME_MAX) ==
          0) &&
         (a.layer == b.layer) && (a.slice_idx == b.slice_idx) &&
         (a.head == b.head);
}
inline bool operator!=(const ArrayKey &a, const ArrayKey &b) {
  return !(a == b);
}

// 為了 SC_REPORT / sc_fifo << 友好
inline std::ostream &operator<<(std::ostream &os, const ArrayKey &k) {
  os << "ArrayKey{w=\"" << k.get_weight_name() << "\", layer=" << k.layer
     << ", slice=" << k.slice_idx << ", head=" << k.head << "}";
  return os;
}