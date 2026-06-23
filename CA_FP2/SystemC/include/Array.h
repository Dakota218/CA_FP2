#pragma once

#include "hw_config_selector.h"
#include "mem_types.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <sysc/utils/sc_report.h>
#include <systemc.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

// 需要 nlohmann/json 來讀 manifest
#include <nlohmann/json.hpp>

// ------------------------------------------------------------
// Array (新版 ports 對齊 Core_Controller)
//   - Array_start
//   - Array_op   (0=IS, 1=OS, 2=QK, 3=PV)
//   - Array_key  (layer/weight_name/slice/head)
//   - Array_row / Array_col : GEMM/GEMV shape (IS/OS/QK/PV 都使用)
// ------------------------------------------------------------
template <typename T, int MODE = 0> class Array : public sc_core::sc_module {
public:
  SC_HAS_PROCESS(Array);
  static constexpr int kArrayInSize = ActiveHWConfig::BUF_ARRAY_IN_SIZE;
  static constexpr int kArrayOutSize = ActiveHWConfig::BUF_ARRAY_OUT_SIZE;
  static constexpr int kKBufSize = ActiveHWConfig::BUF_K_SIZE;
  static constexpr int kVBufSize = ActiveHWConfig::BUF_V_SIZE;

  static_assert(kArrayInSize > 0, "BUF_ARRAY_IN_SIZE must be > 0");
  static_assert(kArrayOutSize > 0, "BUF_ARRAY_OUT_SIZE must be > 0");
  static_assert(kKBufSize > 0, "BUF_K_SIZE must be > 0");
  static_assert(kVBufSize > 0, "BUF_V_SIZE must be > 0");
  static_assert(ActiveHWConfig::ARRAY_MACS_PER_CYCLE > 0,
                "ARRAY_MACS_PER_CYCLE must be > 0");
  static_assert(ActiveHWConfig::DRAM_LANES > 0, "DRAM_LANES must be > 0");
  // ============= Ports =============
  sc_core::sc_in<bool> clk;

  // I/O buffers (size follows ActiveHWConfig)
  sc_core::sc_vector<sc_core::sc_in<T>> Array_input_buffer;

  // K/V staging buffers (size follows ActiveHWConfig)
  sc_core::sc_vector<sc_core::sc_in<T>> Array_K_buffer;
  sc_core::sc_vector<sc_core::sc_in<T>> Array_V_buffer;
  sc_core::sc_vector<sc_core::sc_out<T>> Array_output_buffer;

  // control/status
  sc_core::sc_out<bool> Array_busy;
  sc_core::sc_out<bool> Array_done;

  sc_core::sc_in<bool> Array_start;
  sc_core::sc_in<sc_dt::sc_uint<2>> Array_op; // 0=IS,1=OS,2=QK,3=PV
  sc_core::sc_in<ArrayKey> Array_key;

  sc_core::sc_in<int> Array_row;
  sc_core::sc_in<int> Array_col;

  // 可保留：debug
  sc_core::sc_in<int> token_pos;

  // ============= Ctor =============
  Array(sc_core::sc_module_name name, std::string manifest_path = "",
        std::string model_dir = "")
      : sc_core::sc_module(name), clk("clk"),
        Array_input_buffer("Array_input_buffer", kArrayInSize),
        Array_K_buffer("Array_K_buffer", kKBufSize),
        Array_V_buffer("Array_V_buffer", kVBufSize),
        Array_output_buffer("Array_output_buffer", kArrayOutSize),
        Array_busy("Array_busy"), Array_done("Array_done"),
        Array_start("Array_start"), Array_op("Array_op"),
        Array_key("Array_key"), Array_row("Array_row"), Array_col("Array_col"),
        token_pos("token_pos"),
        manifest_path_(manifest_path.empty() ? default_manifest_path_by_mode_()
                                             : std::move(manifest_path)),
        model_dir_(model_dir.empty() ? default_model_dir_by_mode_()
                                     : std::move(model_dir)) {
    SC_THREAD(run);
    sensitive << clk.pos();
    // dont_initialize();
    bin_buf_.resize(8 * 1024 * 1024);
    kv_buf_.resize(1 * 1024 * 1024); // 1MB for KV file stream buffer
    {
      std::ostringstream oss;
      oss << "Array: manifest_path=" << manifest_path_
          << ", model_dir=" << model_dir_;
      SC_REPORT_INFO(this->name(), oss.str().c_str());
    }
    // 讀 manifest（一次）
    if (!load_manifest()) {
      SC_REPORT_ERROR(this->name(), "Array: failed to load manifest");
    }
  }

private:
  using json = nlohmann::json;

  struct SliceInfo {
    std::string bin_file;
    std::uint64_t bin_offset_bytes{0};
    std::uint64_t bin_size_bytes{0};
    int rows{0}; // slice_shape[0]
    int cols{0}; // slice_shape[1]
  };

  // -------- config / state --------
  std::string manifest_path_;
  std::string model_dir_;
  json manifest_;

  // bin file handle cache
  std::string cur_bin_path_;
  std::ifstream bin_fin_;
  // std::array<T, 2048> outbuf_{};
  // --- file buffer: 必須是 member（避免 pubsetbuf dangling）---
  std::vector<char> bin_buf_;
  // -------- KV bin file handle cache (separate from weight bin_fin_) --------
  std::string cur_kv_path_;
  std::ifstream kv_fin_;
  std::vector<char> kv_buf_;
  // --- accumulator: size follows output SRAM capacity ---
  std::array<float, kArrayOutSize> acc_{};
  // 快取 slice 查詢結果：key = layer|weight_name|slice_idx
  struct SliceKey {
    int layer;
    std::string weight;
    int slice;
    bool operator==(const SliceKey &o) const {
      return layer == o.layer && slice == o.slice && weight == o.weight;
    }
  };
  struct SliceKeyHash {
    std::size_t operator()(const SliceKey &k) const {
      std::size_t h1 = std::hash<int>{}(k.layer);
      std::size_t h2 = std::hash<int>{}(k.slice);
      std::size_t h3 = std::hash<std::string>{}(k.weight);
      return (h1 * 1315423911u) ^ (h2 * 2654435761u) ^
             (h3 + 0x9e3779b97f4a7c15ULL);
    }
  };
  std::unordered_map<SliceKey, SliceInfo, SliceKeyHash> slice_cache_;

  // 用來判斷「是否已經在某個 GEMM 開頭清過 output」
  std::vector<float> wmat_;        // K*N contiguous weights (FP32)
  SliceKey last_wkey_{-1, "", -1}; // cache key
  bool wmat_valid_{false};

  // -------- helpers --------
  static inline std::string layer_key_with_dot(int layer) {
    std::ostringstream oss;
    if constexpr (MODE == 7) {
      oss << "model." << layer << ".";
    } else {
      oss << "layers." << layer << ".";
    }
    return oss.str();
  }

  static inline std::string layer_key_no_dot(int layer) {
    std::ostringstream oss;
    if constexpr (MODE == 7) {
      oss << "model." << layer;
    } else {
      oss << "layers." << layer;
    }
    return oss.str();
  }
  static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

  inline int calc_latency_cycles(int row, int col) {
    const int effective_parallelism = std::min(
        ActiveHWConfig::ARRAY_MACS_PER_CYCLE, ActiveHWConfig::DRAM_LANES);

    if (effective_parallelism < 1) {
      SC_REPORT_ERROR(this->name(),
                      "Array: effective_parallelism must be >= 1.");
      return 1;
    }

    const long long total_work = 1LL * row * col;
    // int lat = ceil_div_int((int)total_work, effective_parallelism);
    long long lat_ll =
        (total_work + effective_parallelism - 1) / effective_parallelism;
    int lat = (lat_ll > std::numeric_limits<int>::max())
                  ? std::numeric_limits<int>::max()
                  : static_cast<int>(lat_ll);

    if (lat < 1) {
      std::ostringstream oss;
      oss << "Array: calculated latency < 1 cycle for row=" << row
          << " col=" << col << ". Check if Array_row/Array_col are correct.";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      lat = 1;
    }

    lat = ceil_div_int(lat * 100, 80);
    lat += 5; // memory access + compute overhead
    return lat;
  }

  void commit_outputs_from_acc(int N) {
    if (N < 0 || N > kArrayOutSize) {
      std::ostringstream oss;
      oss << "Array: output length N=" << N
          << " exceeds BUF_ARRAY_OUT_SIZE=" << kArrayOutSize;
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      N = std::max(0, std::min(N, kArrayOutSize));
    }

    for (int j = 0; j < N; ++j) {
      Array_output_buffer[(std::size_t)j].write((T)acc_[(std::size_t)j]);
    }
    for (int j = N; j < kArrayOutSize; ++j) {
      Array_output_buffer[(std::size_t)j].write((T)0);
    }
  }

  void commit_zero_outputs() {
    for (int j = 0; j < kArrayOutSize; ++j) {
      Array_output_buffer[(std::size_t)j].write((T)0);
    }
  }

  bool load_manifest() {
    std::ifstream f(manifest_path_);
    if (!f.is_open()) {
      std::ostringstream oss;
      oss << "Array: cannot open manifest: " << manifest_path_;
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }
    try {
      f >> manifest_;
    } catch (const std::exception &e) {
      std::ostringstream oss;
      oss << "Array: manifest parse error: " << e.what();
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }
    return true;
  }
  static inline std::string join_path(const std::string &dir,
                                      const std::string &file) {
    if (!file.empty() && file[0] == '/')
      return file; // absolute path
    if (dir.empty())
      return file;
    if (!dir.empty() && dir.back() == '/')
      return dir + file;
    return dir + "/" + file;
  }
  static inline std::string getenv_or_default_(const char *name,
                                               const std::string &fallback) {
    const char *v = std::getenv(name);
    if (v && v[0] != '\0') {
      return std::string(v);
    }
    return fallback;
  }

  static inline std::string asmap_default_run_dir_() {
    return "/home/wilson/SystemC/auto_sched_map/runs/llama_ffn_plan000_v0";
  }

  static inline std::string asmap_dram_dir_() {
    return getenv_or_default_("ASMAP_DRAM_DIR",
                              join_path(asmap_default_run_dir_(), "dram"));
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
    const std::string path = asmap_dram_dir_();
    for (std::size_t i = path.size(); i >= 4; --i) {
      const std::size_t p = i - 4;
      if (path[p] == '_' && path[p + 1] == 'G' &&
          path[p + 2] >= '0' && path[p + 2] <= '9' &&
          path[p + 3] >= '0' && path[p + 3] <= '9') {
        return path.substr(p + 1, 3);
      }
    }
    return "G01";
  }

  static inline std::string mode7_manifest_path_() {
    const std::string group = mode7_group_suffix_();
    return join_path(asmap_dram_dir_(), "memory_map_sim_" + group + ".json");
  }
#ifndef FP1_SUBMISSION_ROOT
#define FP1_SUBMISSION_ROOT "/home/wilson/2026_Spring_CA/FP1/submission"
#endif

#ifndef FP1_GROUP_NUM
#define FP1_GROUP_NUM 1
#endif

  static std::string fp1_group_suffix_() {
    std::ostringstream oss;
    oss << 'G' << std::setw(2) << std::setfill('0') << FP1_GROUP_NUM;
    return oss.str();
  }

  static std::string fp1_group_dir_() {
    const std::string g = fp1_group_suffix_();
    return std::string(FP1_SUBMISSION_ROOT) + "/CA_FP1_" + g;
  }

  static std::string fp1_manifest_path_() {
    const std::string g = fp1_group_suffix_();
    return fp1_group_dir_() + "/mlp_dram_map_" + g + ".json";
  }
  static std::string default_manifest_path_by_mode_() {
    if constexpr (MODE == 0) {
      return "/home/wilson/SystemC/DRAM_json/Llama_manifest.json";
    } else if constexpr (MODE == 1) {
      return "/home/wilson/SystemC/Compiler_ISA/1x1_mlp/mlp_final_map.json";
    } else if constexpr (MODE == 2) {
      return "/home/wilson/SystemC/Compiler_ISA/2x2_mlp/mlp_final_map.json";
    } else if constexpr (MODE == 3) {
      return "/home/wilson/SystemC/Compiler_ISA/4x4_mlp/mlp_final_map.json";
    } else if constexpr (MODE == 4) {
      return "/home/wilson/SystemC/Compiler_ISA/8x8_mlp/mlp_final_map.json";
    } else if constexpr (MODE == 5) {
      return fp1_manifest_path_();
    } else if constexpr (MODE == 6) {
      return join_path(asmap_dram_dir_(), "Llama_manifest.json");
    } else if constexpr (MODE == 7) {
      return mode7_manifest_path_();
    } else {
      return "/home/wilson/SystemC/DRAM_json/Llama_manifest.json";
    }
  }

  static std::string default_model_dir_by_mode_() {
    if constexpr (MODE == 0) {
      return "/home/wilson/SystemC/DRAM_json";
    } else if constexpr (MODE == 1) {
      return "/home/wilson/SystemC/Compiler_ISA/1x1_mlp";
    } else if constexpr (MODE == 2) {
      return "/home/wilson/SystemC/Compiler_ISA/2x2_mlp";
    } else if constexpr (MODE == 3) {
      return "/home/wilson/SystemC/Compiler_ISA/4x4_mlp";
    } else if constexpr (MODE == 4) {
      return "/home/wilson/SystemC/Compiler_ISA/8x8_mlp";
    } else if constexpr (MODE == 5) {
      return fp1_group_dir_();
    } else if constexpr (MODE == 6) {
      return asmap_dram_dir_();
    } else if constexpr (MODE == 7) {
      return asmap_dram_dir_();
    } else {
      return "/home/wilson/SystemC/DRAM_json";
    }
  }
  // ===== KV buffer layout (must match Core_Controller) =====
  static inline int kv_local_idx(int kv_head) { return (kv_head & 1); }
  static inline int kv_buf_off(int layer, int kv_head) {
    return 128 * layer + 64 * kv_local_idx(kv_head);
  }

  bool open_kv_bin_if_needed(const std::string &kv_file_rel) {
    const std::string full = join_path(model_dir_, kv_file_rel);

    if (kv_fin_.is_open() && full == cur_kv_path_)
      return true;

    if (kv_fin_.is_open())
      kv_fin_.close();
    kv_fin_.clear();

    // setbuf before open
    if (!kv_buf_.empty()) {
      kv_fin_.rdbuf()->pubsetbuf(kv_buf_.data(),
                                 (std::streamsize)kv_buf_.size());
    }

    kv_fin_.open(full, std::ios::in | std::ios::binary);
    if (!kv_fin_.is_open()) {
      return false; // caller will try other candidates
    }

    cur_kv_path_ = full;
    return true;
  }

  bool read_kv_vec_fp32(bool is_k, int layer, int kv_head, int token_pos_1b,
                        std::array<float, 64> &out) {
    //  KV 檔案規格就是 1..MAX_SEQ
    if (token_pos_1b < 1 || token_pos_1b > kvfile::MAX_SEQ) {
      SC_REPORT_ERROR(this->name(),
                      "Array: token_pos out of range for KV read");
      return false;
    }
    if (layer < 0 || layer >= kvfile::NUM_LAYERS) {
      SC_REPORT_ERROR(this->name(), "Array: layer out of range for KV read");
      return false;
    }
    if (kv_head < 0 || kv_head >= kvfile::NUM_HEADS) {
      SC_REPORT_ERROR(this->name(), "Array: kv_head out of range for KV read");
      return false;
    }

    const std::string rel = kvfile::make_relpath(is_k, layer);
    if (!open_kv_bin_if_needed(rel)) {
      std::ostringstream oss;
      oss << "Array: cannot open KV file: " << join_path(model_dir_, rel);
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    const std::uint64_t byte_off =
        kvfile::byte_offset_vec(token_pos_1b, kv_head);

    //  擋住 mem_types.h 回傳的 sentinel
    if (byte_off == std::numeric_limits<std::uint64_t>::max()) {
      SC_REPORT_ERROR(this->name(), "Array: KV byte_offset_vec failed");
      return false;
    }

    //  額外保險：避免 seek/read 超出 KV 檔案規格容量
    const std::uint64_t bytes =
        (std::uint64_t)kvfile::HEAD_DIM * kvfile::ELEM_BYTES;
    if (byte_off + bytes > kvfile::FILE_BYTES) {
      SC_REPORT_ERROR(this->name(), "Array: KV read range exceeds FILE_BYTES");
      return false;
    }

    kv_fin_.clear();
    kv_fin_.seekg((std::streamoff)byte_off, std::ios::beg);
    if (!kv_fin_) {
      SC_REPORT_ERROR(this->name(), "Array: KV seekg failed");
      return false;
    }

    kv_fin_.read(reinterpret_cast<char *>(out.data()), (std::streamsize)bytes);
    if (!kv_fin_) {
      SC_REPORT_ERROR(this->name(), "Array: KV read failed");
      return false;
    }
    return true;
  }
  bool open_bin_if_needed(const std::string &bin_file) {
    const std::string full = join_path(model_dir_, bin_file);
    if (bin_fin_.is_open() && full == cur_bin_path_)
      return true;

    if (bin_fin_.is_open())
      bin_fin_.close();
    bin_fin_.clear();

    // 先 setbuf（在 open 之前）
    bin_fin_.rdbuf()->pubsetbuf(bin_buf_.data(),
                                (std::streamsize)bin_buf_.size());

    bin_fin_.open(full, std::ios::in | std::ios::binary);
    cur_bin_path_ = full;

    if (!bin_fin_.is_open()) {
      std::ostringstream oss;
      oss << "Array: cannot open bin_file: " << full;
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }
    return true;
  }

  bool query_slice_info(int layer, const std::string &weight_name,
                        int slice_idx, SliceInfo &out) {
    SliceKey sk{layer, weight_name, slice_idx};
    auto it = slice_cache_.find(sk);
    if (it != slice_cache_.end()) {
      out = it->second;
      return true;
    }

    json *layer_obj = nullptr;
    const std::string k1 = layer_key_with_dot(layer);
    const std::string k2 = layer_key_no_dot(layer);

    if (manifest_.contains(k1))
      layer_obj = &manifest_[k1];
    else if (manifest_.contains(k2))
      layer_obj = &manifest_[k2];

    if (!layer_obj) {
      std::ostringstream oss;
      oss << "Array: manifest missing layer key: " << k1 << " / " << k2;
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    if (!layer_obj->contains(weight_name)) {
      std::ostringstream oss;
      oss << "Array: manifest missing weight_name: " << weight_name
          << " under key=" << k1 << " / " << k2;
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    const std::string sidx = std::to_string(slice_idx);
    json &wobj = (*layer_obj)[weight_name];
    if (!wobj.contains(sidx)) {
      std::ostringstream oss;
      oss << "Array: manifest missing slice_idx=" << sidx
          << " for weight_name=" << weight_name << " layer=" << layer;
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    json &entry = wobj[sidx];

    SliceInfo info{};
    try {
      info.bin_file = entry.at("bin_file").get<std::string>();
      info.bin_offset_bytes = entry.at("bin_offset_bytes").get<std::uint64_t>();
      info.bin_size_bytes = entry.at("bin_size_bytes").get<std::uint64_t>();

      auto shape = entry.at("slice_shape");
      info.rows = shape.at(0).get<int>();
      info.cols = shape.at(1).get<int>();
    } catch (const std::exception &e) {
      std::ostringstream oss;
      oss << "Array: manifest entry parse error: " << e.what()
          << " (layer=" << layer << ", weight=" << weight_name
          << ", slice=" << slice_idx << ")";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    slice_cache_.emplace(sk, info);
    out = info;
    return true;
  }

  // IS/OS：tile 級的 y += x_tile * W_tile
  // input: x[r0..r0+7]
  // output: y[c0..c0+7]
  bool read_weight_row_fp32_lenN(const SliceInfo &si, int r, int ncols,
                                 std::vector<float> &wrow) {
    constexpr std::uint64_t EBYTES = 4; // FP32

    if (si.rows <= 0 || si.cols <= 0) {
      SC_REPORT_ERROR(this->name(), "Array: invalid slice shape");
      return false;
    }
    if (r < 0 || r >= si.rows) {
      SC_REPORT_ERROR(this->name(), "Array: weight row out of range");
      return false;
    }
    if (ncols < 0 || ncols > si.cols) {
      SC_REPORT_ERROR(this->name(), "Array: ncols out of range");
      return false;
    }

    if (!open_bin_if_needed(si.bin_file))
      return false;

    // row-major flatten: idx = r*si.cols + c
    const std::uint64_t elem_index =
        (std::uint64_t)r * (std::uint64_t)si.cols; // c=0
    const std::uint64_t byte_off = si.bin_offset_bytes + elem_index * EBYTES;

    wrow.resize((std::size_t)ncols);
    bin_fin_.clear();
    bin_fin_.seekg((std::streamoff)byte_off, std::ios::beg);
    if (!bin_fin_) {
      SC_REPORT_ERROR(this->name(), "Array: bin seekg failed");
      return false;
    }

    // 先做 slice-range 檢查，再讀（避免跨 slice 邏輯錯）
    const std::uint64_t end_off = byte_off + (std::uint64_t)ncols * EBYTES;
    const std::uint64_t limit = si.bin_offset_bytes + si.bin_size_bytes;
    if (end_off > limit) {
      SC_REPORT_ERROR(this->name(),
                      "Array: row read exceeds bin_size_bytes slice range");
      return false;
    }
    bin_fin_.read(reinterpret_cast<char *>(wrow.data()),
                  (std::streamsize)ncols * (std::streamsize)EBYTES);

    if (!bin_fin_) {
      SC_REPORT_ERROR(this->name(), "Array: bin read failed");
      return false;
    }

    return true;
  }

  bool read_weight_matrix_fp32_contig(const SliceInfo &si, int K, int N,
                                      std::vector<float> &W_out) {
    constexpr std::uint64_t EBYTES = 4; // FP32

    if (K <= 0 || N <= 0) {
      SC_REPORT_ERROR(this->name(), "Array: K/N must be >= 1");
      return false;
    }
    if (si.rows <= 0 || si.cols <= 0) {
      SC_REPORT_ERROR(this->name(), "Array: invalid slice shape");
      return false;
    }

    // 你本來就要求 shape match，所以這裡也保險檢查一次
    if (si.rows != K || si.cols != N) {
      SC_REPORT_ERROR(this->name(),
                      "Array: read_weight_matrix called with shape mismatch");
      return false;
    }

    if (!open_bin_if_needed(si.bin_file))
      return false;

    const std::uint64_t byte_off = si.bin_offset_bytes;
    const std::uint64_t need_bytes =
        (std::uint64_t)K * (std::uint64_t)N * EBYTES;
    const std::uint64_t limit = si.bin_offset_bytes + si.bin_size_bytes;

    // 先做 slice-range 檢查
    if (byte_off + need_bytes > limit) {
      SC_REPORT_ERROR(this->name(),
                      "Array: matrix read exceeds bin_size_bytes slice range");
      return false;
    }

    W_out.resize((std::size_t)K * (std::size_t)N);

    bin_fin_.clear();
    bin_fin_.seekg((std::streamoff)byte_off, std::ios::beg);
    if (!bin_fin_) {
      SC_REPORT_ERROR(this->name(), "Array: bin seekg failed");
      return false;
    }

    bin_fin_.read(reinterpret_cast<char *>(W_out.data()),
                  (std::streamsize)need_bytes);
    if (!bin_fin_) {
      SC_REPORT_ERROR(this->name(), "Array: bin read failed");
      return false;
    }
    return true;
  }

  bool exec_is_os_full_gemm(const ArrayKey &k, int K, int N) {
    const int layer = (int)k.layer.to_uint();
    const int slice_idx = (int)k.slice_idx.to_uint();
    const std::string wname = k.get_weight_name();

    if (K <= 0 || K > kArrayInSize || N <= 0 || N > kArrayOutSize) {
      std::ostringstream oss;
      oss << "Array: K/N exceeds configured SRAM capacity. "
          << "K=" << K << " (BUF_ARRAY_IN_SIZE=" << kArrayInSize << "), "
          << "N=" << N << " (BUF_ARRAY_OUT_SIZE=" << kArrayOutSize << ")";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    SliceInfo si{};
    if (!query_slice_info(layer, wname, slice_idx, si)) {
      SC_REPORT_ERROR(this->name(), "Array: query_slice_info failed");
      return false;
    }

    if (si.rows > kArrayInSize) {
      std::ostringstream oss;
      oss << "Array: weight rows exceed BUF_ARRAY_IN_SIZE. "
          << "weight rows=" << si.rows << ", BUF_ARRAY_IN_SIZE=" << kArrayInSize
          << " (layer=" << layer << ", weight=" << wname
          << ", slice=" << slice_idx << ")";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    if (si.cols > kArrayOutSize) {
      std::ostringstream oss;
      oss << "Array: weight cols exceed BUF_ARRAY_OUT_SIZE. "
          << "weight cols=" << si.cols
          << ", BUF_ARRAY_OUT_SIZE=" << kArrayOutSize << " (layer=" << layer
          << ", weight=" << wname << ", slice=" << slice_idx << ")";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    if (si.rows != K || si.cols != N) {
      std::ostringstream oss;
      oss << "Array: shape mismatch for IS/OS. "
          << "manifest slice_shape=(" << si.rows << "," << si.cols << ") "
          << "but Array_row/Array_col(K,N)=(" << K << "," << N << "). "
          << "key(layer=" << layer << ", weight=" << wname
          << ", slice=" << slice_idx << ")";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    constexpr std::uint64_t EBYTES = 4;
    const std::uint64_t need_bytes =
        (std::uint64_t)si.rows * (std::uint64_t)si.cols * EBYTES;
    if (si.bin_size_bytes < need_bytes) {
      std::ostringstream oss;
      oss << "Array: bin_size_bytes too small. "
          << "need >= " << need_bytes << " but got " << si.bin_size_bytes
          << " (rows=" << si.rows << ", cols=" << si.cols << ")";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      return false;
    }

    std::fill(acc_.begin(), acc_.begin() + N, 0.0f);

    std::vector<float> x((std::size_t)K, 0.0f);
    for (int i = 0; i < K; ++i) {
      x[(std::size_t)i] = (float)Array_input_buffer[(std::size_t)i].read();
    }

    SliceKey cur_key{layer, wname, slice_idx};
    if (!(cur_key == last_wkey_) || !wmat_valid_) {
      if (!read_weight_matrix_fp32_contig(si, K, N, wmat_))
        return false;
      last_wkey_ = cur_key;
      wmat_valid_ = true;
    }

    for (int i = 0; i < K; ++i) {
      const float xi = x[(std::size_t)i];
      const std::size_t base = (std::size_t)i * (std::size_t)N;
      const float *wrow = wmat_.data() + base;
      for (int j = 0; j < N; ++j) {
        acc_[(std::size_t)j] += xi * wrow[(std::size_t)j];
      }
    }

    return true;
  }
  // ------------------------------------------------------------
  // QK: scores[t] = dot(Q[64], K[t][64])
  // Q from Array_input_buffer[0..63]
  // K[t] from KV bin (even), and for odd token: last token K[tp] from
  // Array_K_buffer shape: row=64, col=tp output: acc_[0..tp-1]
  // ------------------------------------------------------------
  void exec_qk_kv(const ArrayKey &k, int tp) {
    const int layer = (int)k.layer.to_uint();
    const int kv_head = (int)k.head.to_uint();

    //  tp 應該跟 KV 檔案規格一致
    if (tp <= 0 || tp > kvfile::MAX_SEQ) {
      SC_REPORT_ERROR(this->name(),
                      "Array QK: tp out of range (1..KV_MAX_SEQ)");
      std::fill(acc_.begin(), acc_.end(), 0.0f);
      return;
    }

    const bool odd = (tp & 1);

    std::fill(acc_.begin(), acc_.begin() + tp, 0.0f);

    float q[64];
    for (int i = 0; i < 64; ++i) {
      q[i] = (float)Array_input_buffer[(std::size_t)i].read();
    }

    std::array<float, 64> kvec{};

    for (int t = 1; t <= tp; ++t) { // 1-based token_pos
      if (odd && t == tp) {
        const int off = kv_buf_off(layer, kv_head);
        if (off < 0 || off + 63 >= kKBufSize) {
          SC_REPORT_ERROR(this->name(), "Array QK: BUF_K offset out of range");
          acc_[(std::size_t)(t - 1)] = 0.0f;
          continue;
        }
        for (int i = 0; i < 64; ++i) {
          kvec[(std::size_t)i] =
              (float)Array_K_buffer[(std::size_t)(off + i)].read();
        }
      } else {
        if (!read_kv_vec_fp32(true, layer, kv_head, t, kvec)) {
          acc_[(std::size_t)(t - 1)] = 0.0f;
          continue;
        }
      }
      // 在 exec_qk_kv() 一開始（for t 之前）加：
      constexpr float inv_sqrt_hd =
          1.0f / 8.0f; // head_dim=64 -> 1/sqrt(64)=1/8
      float sum = 0.0f;
      for (int i = 0; i < 64; ++i)
        sum += q[i] * kvec[(std::size_t)i];
      // ★補上 attention scale
      sum *= inv_sqrt_hd;
      acc_[(std::size_t)(t - 1)] = sum;
    }
  }
  // ------------------------------------------------------------
  // PV: y[j] = sum_t P[t] * V[t][j]
  // P from Array_input_buffer[0..tp-1]
  // V[t] from KV bin (even), and for odd token: last token V[tp] from
  // Array_V_buffer shape: row=tp, col=64 output: acc_[0..63]
  // ------------------------------------------------------------
  void exec_pv_kv(const ArrayKey &k, int tp) {
    const int layer = (int)k.layer.to_uint();
    const int kv_head = (int)k.head.to_uint();

    if (tp <= 0 || tp > kvfile::MAX_SEQ) {
      SC_REPORT_ERROR(this->name(),
                      "Array PV: tp out of range (1..KV_MAX_SEQ)");
      std::fill(acc_.begin(), acc_.end(), 0.0f);
      return;
    }

    const bool odd = (tp & 1);

    std::fill(acc_.begin(), acc_.begin() + 64, 0.0f);

    std::vector<float> p((std::size_t)tp, 0.0f);
    for (int t = 0; t < tp; ++t) {
      p[(std::size_t)t] = (float)Array_input_buffer[(std::size_t)t].read();
    }

    std::array<float, 64> vvec{};

    for (int t = 1; t <= tp; ++t) {
      if (odd && t == tp) {
        const int off = kv_buf_off(layer, kv_head);
        if (off < 0 || off + 63 >= kVBufSize) {
          SC_REPORT_ERROR(this->name(), "Array PV: BUF_V offset out of range");
          continue;
        }
        for (int j = 0; j < 64; ++j) {
          vvec[(std::size_t)j] =
              (float)Array_V_buffer[(std::size_t)(off + j)].read();
        }
      } else {
        if (!read_kv_vec_fp32(false, layer, kv_head, t, vvec)) {
          continue;
        }
      }

      const float pt = p[(std::size_t)(t - 1)];
      for (int j = 0; j < 64; ++j) {
        acc_[(std::size_t)j] += pt * vvec[(std::size_t)j];
      }
    }
  }

  void run() {
    Array_busy.write(false);
    Array_done.write(false);

    bool busy_state = false;
    bool prev_start = false;

    while (true) {
      wait(); // clk posedge

      // done 預設每個 posedge 都清掉 => 保證 done one-pulse
      Array_done.write(false);

      const bool s = Array_start.read();
      const bool start_pulse = (s && !prev_start);
      prev_start = s;

      // idle 狀態才接受 start
      if (!start_pulse || busy_state) {
        if (start_pulse && busy_state) {
          SC_REPORT_WARNING(this->name(),
                            "Array: start received while busy; ignored");
        }
        continue;
      }

      // ===== 接受 start：busy 立刻拉高 =====
      busy_state = true;
      Array_busy.write(true);

      const std::uint32_t op = Array_op.read().to_uint();
      const ArrayKey k = Array_key.read();

      int K = Array_row.read();
      int N = Array_col.read();

      // 基本防呆
      if (K <= 0 || N <= 0) {
        SC_REPORT_ERROR(this->name(), "Array: Array_row/Array_col must be > 0");
        std::fill(acc_.begin(), acc_.end(), 0.0f);
        // latency 還是走 1 cycle 保底
        K = std::max(K, 1);
        N = std::max(N, 1);
      }

      const int latency = calc_latency_cycles(K, N);

      bool op_ok = false;

      if (op == 0 || op == 1) {
        std::ostringstream oss;
        oss << sc_core::sc_time_stamp() << " [Array] "
            << (op == 0 ? "IS" : "OS") << " key=" << k
            << " start, latency=" << latency << " (K=" << K << ", N=" << N
            << ")";
        SC_REPORT_INFO(this->name(), oss.str().c_str());

        op_ok = exec_is_os_full_gemm(k, K, N);
      } else if (op == 2) {
        // QK: expect row=64 col=tp
        if (K != 64) {
          SC_REPORT_ERROR(this->name(), "Array QK: expect row=64");
        }
        // 在 run() 裡 QK/PV 時加：
        int tp_port = token_pos.read();
        if (tp_port >= 1) {
          int tp_shape = (op == 2) ? N : K;
          if (tp_port != tp_shape) {
            SC_REPORT_WARNING(
                name(), "Array: token_pos port != tp inferred from shape");
          }
        }
        if (N > kvfile::MAX_SEQ) {
          SC_REPORT_ERROR(this->name(),
                          "Array QK: tp(shape=N) exceeds KV_MAX_SEQ");
        }
        exec_qk_kv(k, /*tp=*/N);

      } else if (op == 3) {
        // PV: expect row=tp col=64
        if (N != 64) {
          SC_REPORT_ERROR(this->name(), "Array PV: expect col=64");
        }
        // 在 run() 裡 QK/PV 時加：
        int tp_port = token_pos.read();
        if (tp_port >= 1) {
          int tp_shape = (op == 2) ? N : K;
          if (tp_port != tp_shape) {
            SC_REPORT_WARNING(
                name(), "Array: token_pos port != tp inferred from shape");
          }
        }
        if (K > kvfile::MAX_SEQ) {
          SC_REPORT_ERROR(this->name(),
                          "Array PV: tp(shape=K) exceeds KV_MAX_SEQ");
        }
        exec_pv_kv(k, /*tp=*/K);

      } else {
        SC_REPORT_ERROR(this->name(), "Array: unknown Array_op, output zeros");
        std::fill(acc_.begin(), acc_.end(), 0.0f);
      }

      // ===== busy 維持 latency cycles =====
      for (int c = 0; c < latency; ++c) {
        wait();
      }

      // ===== commit output（依 op 決定 commit 長度）=====
      if (!op_ok && (op == 0 || op == 1)) {
        commit_zero_outputs();
      } else if (op == 0 || op == 1) {
        commit_outputs_from_acc(N);
      } else if (op == 2) {
        commit_outputs_from_acc(N);
      } else if (op == 3) {
        commit_outputs_from_acc(64);
      } else {
        commit_zero_outputs();
      }
      //  重要：在結束 op 時重新取樣 start，讓 edge detector re-arm
      prev_start = Array_start.read();
      Array_busy.write(false);
      busy_state = false;
      Array_done.write(true); // done 拉高 1 cycle

      // 下一個 posedge 會在迴圈開頭自動清掉 done
    }
  }
};
