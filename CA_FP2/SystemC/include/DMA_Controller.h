#ifndef DMA_CONTROLLER_H
#define DMA_CONTROLLER_H

#include "hw_config_selector.h"
#include "mem_types.h"
// #include "rope_table_theta500000_pos0_127_d64_f32.h"
#include "rope_table_official_hf_pos0_127_d64_f32.h"
#include <algorithm> // std::min
#include <array>
#include <cctype>
#include <cstdint> // std::uint32_t, std::uint64_t
#include <cstdio>  // std::snprintf
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream> // std::ostringstream
#include <string>
#include <sysc/utils/sc_report.h>
#include <systemc.h>
#include <type_traits>
#include <vector> // std::vector

/**
 * DMA_Controller
 * ============================================================================
 * [Selector 編碼]
 *   - src_sel / dst_sel 指定搬運端點：
 *       0 .. (bufs_.size()-1) : Tcore 內部
 *
 * : RoPE ROM 端（只支援讀取到 buffer）
 *
 * [Length / Offset 語意（很重要）]
 *
 *   - src_offset / dst_offset 的單位也依端點而不同：
 *       * 端點是 BUF 時：offset = element index（buffer 內元素起始位置）
 *
 *
 * [備註]
 *   - bufs_ 是「DMA 讀取來源」的 sc_signal buffer；shadow_bufs_ 是「DMA
 * 寫入目的」 的 shadow array（實際寫回 sc_signal 由上層仲裁/commit）。
 */

// ====== 新增：報告輔助巨集（不改時序/行為，只改訊息內容）======
#define DMA_REPORT_INFO(expr_stream)                                           \
  do {                                                                         \
    std::ostringstream __oss;                                                  \
    __oss << expr_stream << " | t=" << sc_core::sc_time_stamp();               \
    SC_REPORT_INFO(name(), __oss.str().c_str());                               \
  } while (0)

#define DMA_REPORT_WARNING(expr_stream)                                        \
  do {                                                                         \
    std::ostringstream __oss;                                                  \
    __oss << expr_stream << " | t=" << sc_core::sc_time_stamp();               \
    SC_REPORT_WARNING(name(), __oss.str().c_str());                            \
  } while (0)

#define DMA_REPORT_ERROR(expr_stream)                                          \
  do {                                                                         \
    std::ostringstream __oss;                                                  \
    __oss << expr_stream << " | t=" << sc_core::sc_time_stamp();               \
    SC_REPORT_ERROR(name(), __oss.str().c_str());                              \
  } while (0)
// ============================================================

template <typename T, int RBW = 8, int WBW = 8, int overhead = 5, int MODE = 0>
class DMA_Controller : public sc_core::sc_module {
public:
  // ===== Ports =====
  sc_core::sc_in<bool> clk;

  // 控制訊號
  sc_core::sc_in<bool> dma_start; // 拉高表示開始搬運
  sc_core::sc_out<bool> dma_busy; // 正在搬運
  sc_core::sc_out<bool> dma_done; // 搬運完成（脈衝一拍）

  // ===== 新增：DRAM busy（外部端點活動）=====
  // non-buf -> buf：從 start 到 done 期間視為 DRAM read busy
  sc_core::sc_out<bool> dram_rd_busy;

  // buf -> non-buf：從 start 到 done 期間視為 DRAM write busy
  sc_core::sc_out<bool> dram_wr_busy;

  sc_core::sc_in<sc_dt::sc_uint<5>> src_sel;
  sc_core::sc_in<sc_dt::sc_uint<5>> dst_sel;

  // * 端點是 BUF：src_idx/dst_offset 表示 buffer 的「元素起始索引」（非 row）
  sc_core::sc_in<int> src_offset; // 當 src 是 BUF 時的元素起始索引；當 src 是
                                  // MEM 時視為 src_row
  sc_core::sc_in<int> dst_offset; // 當 dst 是 BUF 時的元素起始索引；當 dst 是
                                  // MEM 時視為 dst_row
  sc_core::sc_in<int> length; // MEM<->BUF: rows；BUF->BUF: N elements

  // ====== KV controls (Aligned with Core_Controller) ======

  sc_core::sc_in<int> kv_layer;
  sc_core::sc_in<int> token_pos;   // 1-based token position from Core
  sc_core::sc_in<int> kv_head;     // 0..7 (start head)
  sc_core::sc_in<bool> kv_is_prev; // 1: prev token, 0: current token
  // ★ NEW：接收 MODEL 的 weight_name/layer（固定 64B）
  sc_core::sc_in<ArrayKey> dma_model_key;
  SC_HAS_PROCESS(DMA_Controller);

  // ========== 建構子 (C): 通用（任意多組） ==========
  //
  // buffers        : 給 DMA 讀用的「真實 sc_signal buffer」
  // shadow_buffers : 給 DMA 寫用的「shadow array」，之後由 Tcore 仲裁寫回
  // sc_signal
  //
  static_assert(std::is_same_v<T, float>,
                "DMA_Controller: this version only supports FP32 (T=float).");
  DMA_Controller(
      sc_core::sc_module_name module_name,
      const std::vector<sc_core::sc_vector<sc_core::sc_signal<T>> *> &buffers,
      const std::vector<std::vector<T> *> &shadow_buffers,
      std::string model_dir = "",
      std::string manifest_path = "",
      std::string kv_root_dir = kvfile::DEFAULT_ROOT_DIR)
      : sc_core::sc_module(module_name), bufs_(buffers),
        shadow_bufs_(shadow_buffers),
        model_dir_(model_dir.empty() ? default_model_bin_dir_by_mode_("../DRAM_json")
                                     : std::move(model_dir)),
        model_bin_dir_(default_model_bin_dir_by_mode_(model_dir_)),
        manifest_path_(manifest_path.empty() ? default_manifest_path_by_mode_()
                                             : std::move(manifest_path)),
        kv_root_dir_(std::move(kv_root_dir)) {
    SC_THREAD(run);
    if (bufs_.size() > static_cast<std::size_t>(tcore_sel::SEL_EXT_BASE)) {
      SC_REPORT_FATAL(this->name(),
                      "DMA_Controller: bufs_.size() must be <= 24 because "
                      "24..31 are reserved external selectors.");
    }
    validate_registered_buffers_or_die_();
    {
      std::ostringstream oss;
      oss << "DMA_Controller paths:"
          << " model_dir=" << model_dir_ << " model_bin_dir=" << model_bin_dir_
          << " manifest_path=" << manifest_path_
          << " token_in=" << token_full_path_(true)
          << " token_out=" << token_full_path_(false);
      SC_REPORT_INFO(this->name(), oss.str().c_str());
    }
  }
  ~DMA_Controller() override {
    for (auto &p : kv_files_) {
      if (p && p->is_open()) {
        p->flush();
        p->close();
      }
    }
    for (auto &tf : token_files_) {
      if (tf && tf->is_open()) {
        tf->flush();
        tf->close();
      }
    }
    if (compiler_file_ && compiler_file_->is_open()) {
      compiler_file_->flush();
      compiler_file_->close();
    }
  }

  void run() {
    dma_busy.write(false);
    dma_done.write(false);
    // 新增
    dram_rd_busy.write(false);
    dram_wr_busy.write(false);
    bool prev_start = false;

    auto finish_txn = [&]() {
      dma_busy.write(false);

      // 新增：結束時一律清掉
      dram_rd_busy.write(false);
      dram_wr_busy.write(false);

      dma_done.write(true); // done pulse = 1 cycle
      wait(clk.posedge_event());
      dma_done.write(false);

      // ★★★ 關鍵：同步 start 歷史，避免下一筆 rise 被吃掉
      prev_start = dma_start.read();
    };
    while (true) {
      wait(clk.posedge_event()); // ★ 必須有：讓模擬時間前進
      dma_done.write(false); // 每個 cycle 先清掉 done pulse（可選但很推薦）

      bool cur = dma_start.read();
      bool rise = cur && !prev_start;
      prev_start = cur;

      if (!rise)
        continue;

      if (bufs_.empty()) {
        DMA_REPORT_ERROR("DMA bufs_ is empty (no buffers registered)"
                         << " src_buf=" << src_sel.read().to_uint()
                         << " dst_buf=" << dst_sel.read().to_uint()
                         << " src_offset=" << src_offset.read()
                         << " dst_offset=" << dst_offset.read()
                         << " L=" << length.read());
        finish_txn();
        continue;
      }

      // 安全檢查：shadow_bufs_ 必須跟 bufs_ 同大小
      if (shadow_bufs_.size() != bufs_.size()) {
        DMA_REPORT_ERROR("DMA: shadow_bufs_ size mismatch with bufs_"
                         << " shadow=" << shadow_bufs_.size()
                         << " bufs=" << bufs_.size());
        finish_txn();
        continue;
      }

      dma_busy.write(true);

      // 建議：index 用 s_idx/d_idx
      const unsigned s_idx = src_sel.read().to_uint();
      const unsigned d_idx = dst_sel.read().to_uint();
      const int L = length.read();
      // ===== 新增：DRAM busy 分類（從 start 到 done 保持）=====
      const bool src_is_buf = (s_idx < bufs_.size());
      const bool dst_is_buf = (d_idx < bufs_.size());

      // 非buf -> buf：read busy
      dram_rd_busy.write((!src_is_buf) && dst_is_buf);

      // buf -> 非buf：write busy
      dram_wr_busy.write(src_is_buf && (!dst_is_buf));

      // =========================================================
      // Case A0: ROPE(ROM) -> Buffer  (length = elements)
      //   - src_sel = tcore_sel::SEL_ROPE
      //   - src_offset = pos (1..128)
      //   - length  = N elements (typically 64)
      // =========================================================
      if (s_idx == tcore_sel::SEL_ROPE && d_idx < bufs_.size()) {
        // RoPE table is float32; allow T=float or convertible
        static_assert(std::is_arithmetic<T>::value,
                      "DMA: T must be arithmetic for RoPE path");

        const int pos_1b = this->src_offset.read(); // ISA 1-based
        const int pos = pos_1b - 1;                 // 轉 0-based
        const int dst_off = this->dst_offset.read();
        const int N = L;

        DMA_REPORT_INFO("DMA Case: ROPE->BUF"
                        << " pos=" << pos << " dst_buf=" << d_idx
                        << " dst_offset=" << dst_off << " N(L)=" << N);

        if (pos < 0 || pos >= static_cast<int>(rope::kPosCount)) {
          DMA_REPORT_ERROR("DMA ROPE->BUF: pos out of range"
                           << " pos=" << pos
                           << " kPosCount=" << rope::kPosCount);
          finish_txn();
          continue;
        }
        if (N != static_cast<int>(rope::kHeadDim)) {
          DMA_REPORT_ERROR("DMA ROPE->BUF: N must be equal to head_dim"
                           << " N=" << N << " kHeadDim=" << rope::kHeadDim);
          finish_txn();
          continue;
        }
        if (dst_off < 0) {
          DMA_REPORT_ERROR("DMA ROPE->BUF: invalid dst_offset"
                           << " dst_offset=" << dst_off);
          finish_txn();
          continue;
        }

        auto *dst_shadow = shadow_bufs_[d_idx];
        if (!dst_shadow) {
          DMA_REPORT_ERROR("DMA ROPE->BUF: dst_shadow is nullptr");
          finish_txn();
          continue;
        }
        if (static_cast<std::size_t>(dst_off) + static_cast<std::size_t>(N) >
            dst_shadow->size()) {
          DMA_REPORT_ERROR("DMA ROPE->BUF: dst_shadow overflow"
                           << " dst_offset=" << dst_off << " N=" << N
                           << " shadow_size=" << dst_shadow->size());
          finish_txn();
          continue;
        }

        // latency model: treat like write-only cost
        const int total_latency = latency_mem_to_sram_(N);
        for (int t = 0; t < total_latency; ++t)
          wait(clk.posedge_event());

        // Copy from ROM table into shadow buffer
        for (int i = 0; i < N; ++i) {
          (*dst_shadow)[static_cast<std::size_t>(dst_off + i)] =
              static_cast<T>(rope::kRopeCosSinTable[pos][i]);
        }

        finish_txn();
        continue;
      }

      // =========================================================
      // Case A1: MODEL(FILE/ROM) -> Buffer (length = elements)
      //   - src_sel    = tcore_sel::SEL_MODEL
      //   - src_offset = hash(weight_name) (Core 端目前仍保留)
      //   - kv_layer   = layer_id (Core 端用 kv_layer port 送)
      //   - dma_model_key = ArrayKey{weight_name[64], layer}
      // =========================================================
      if (s_idx == tcore_sel::SEL_MODEL && d_idx < bufs_.size()) {
        const int dst_off = this->dst_offset.read();
        const int N = L;

        // 從 dma_model_key 拿到固定 64B 字串
        const ArrayKey mk = dma_model_key.read();
        const std::string wname = mk.get_weight_name();
        const int layer_from_key = static_cast<int>(mk.layer.to_uint());
        const int layer_from_port = kv_layer.read();

        // hash 一致性檢查（非必要，但非常推薦用來抓 bug）
        const std::uint32_t h_from_core = static_cast<std::uint32_t>(
            static_cast<std::int32_t>(this->src_offset.read()));
        const std::uint32_t h_calc = fnv1a_32_(wname);

        DMA_REPORT_INFO("DMA Case: MODEL->BUF"
                        << " weight=\"" << wname << "\""
                        << " layer(key)=" << layer_from_key << " layer(port)="
                        << layer_from_port << " hash(core)=0x" << std::hex
                        << h_from_core << " hash(calc)=0x" << h_calc << std::dec
                        << " dst_buf=" << d_idx << " dst_offset=" << dst_off
                        << " N(L)=" << N);

        if (wname.empty()) {
          DMA_REPORT_ERROR("DMA MODEL->BUF: weight_name is empty "
                           "(dma_model_key not driven?)");
          for (int t = 0; t < overhead; ++t)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        }

        if (layer_from_port != layer_from_key) {
          DMA_REPORT_ERROR("DMA MODEL->BUF: layer mismatch between port and key"
                           << " layer(port)=" << layer_from_port
                           << " layer(key)=" << layer_from_key);
        }

        if (h_from_core != h_calc) {
          DMA_REPORT_ERROR("DMA MODEL->BUF: hash mismatch (still proceed)"
                           << " core=0x" << std::hex << h_from_core
                           << " calc=0x" << h_calc << std::dec);
        }

        auto *dst_shadow = shadow_bufs_[d_idx];
        if (!dst_shadow) {
          DMA_REPORT_ERROR("DMA MODEL->BUF: dst_shadow is nullptr");
          for (int t = 0; t < overhead; ++t)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        }

        if (N <= 0 || dst_off < 0 ||
            static_cast<std::size_t>(dst_off + N) > dst_shadow->size()) {
          DMA_REPORT_ERROR("DMA MODEL->BUF: dst range invalid"
                           << " dst_off=" << dst_off << " N=" << N
                           << " shadow_size=" << dst_shadow->size());
          for (int t = 0; t < overhead; ++t)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        }

        // latency model：MODEL->BUF 視作
        // write-only，因為通常是從快取（manifest）讀
        // metadata，然後從另一個快取（bin）讀資料；即使不完全命中，整體也不會慢到哪裡去
        const int total_latency = latency_mem_to_sram_(N);
        for (int t = 0; t < total_latency; ++t)
          wait(clk.posedge_event());

        // ---------------------------------------------------------
        // MODEL manifest lookup + bin read (fast path, cached)
        // ---------------------------------------------------------

        const int slice_idx = static_cast<int>(mk.slice_idx.to_uint());
        ModelSlice ms{};

        bool ready = load_model_manifest_once_();
        if (ready)
          ready = lookup_model_slice_(layer_from_key, wname, slice_idx, ms);

        const std::uint64_t need_bytes =
            static_cast<std::uint64_t>(N) * sizeof(T);

        if (ready && need_bytes > ms.size_bytes) {
          DMA_REPORT_ERROR("DMA MODEL->BUF: requested bytes exceed slice size"
                           << " need=" << need_bytes
                           << " slice_size=" << ms.size_bytes
                           << " weight=" << wname << " layer=" << layer_from_key
                           << " slice=" << slice_idx);
          ready = false;
        }

        if (ready) {
          const std::string full_bin = join_path_(model_bin_dir_, ms.bin_file);
          ready = open_model_bin_if_needed_(full_bin);

          if (ready) {
            if (model_read_buf_.size() < static_cast<std::size_t>(N))
              model_read_buf_.resize(static_cast<std::size_t>(N));

            model_bin_fin_.clear();
            model_bin_fin_.seekg(static_cast<std::streamoff>(ms.off_bytes),
                                 std::ios::beg);
            if (!model_bin_fin_) {
              DMA_REPORT_ERROR("DMA MODEL->BUF: seekg failed in bin file");
              ready = false;
            } else {
              model_bin_fin_.read(
                  reinterpret_cast<char *>(model_read_buf_.data()),
                  static_cast<std::streamsize>(need_bytes));
              if (!model_bin_fin_) {
                DMA_REPORT_ERROR("DMA MODEL->BUF: read failed from bin file");
                ready = false;
              } else {
                for (int i = 0; i < N; ++i) {
                  (*dst_shadow)[static_cast<std::size_t>(dst_off + i)] =
                      static_cast<T>(
                          model_read_buf_[static_cast<std::size_t>(i)]);
                }
              }
            }
          }
        }

        if (!ready) {
          for (int i = 0; i < N; ++i) {
            (*dst_shadow)[static_cast<std::size_t>(dst_off + i)] =
                static_cast<T>(0);
          }
        }

        finish_txn();
        continue;
      }
      // =========================================================
      // Case A2: TOKEN_IN(FILE, FP32) -> Buffer
      //   - token_pos (1-based) decides which token slot
      //   - each token slot = 2048 FP32
      //   - length must be 2048
      // =========================================================
      if ((s_idx == tcore_sel::SEL_TOKEN_IN ||
           s_idx == tcore_sel::SEL_COMPILER) &&
          d_idx < bufs_.size()) {

        static_assert(std::is_arithmetic<T>::value,
                      "DMA TOKEN_BIN->BUF: T must be arithmetic");

        const bool is_compiler = (s_idx == tcore_sel::SEL_COMPILER);
        const bool is_in = (s_idx == tcore_sel::SEL_TOKEN_IN);
        const char *which = is_compiler ? "COMPILER" : "IN";

        const int dst_off = this->dst_offset.read();
        const int N = L;

        // --- TOKEN_IN must be exactly 2048 elems ---
        if (!is_compiler) {
          if constexpr (MODE == 0) {
            if (N != TOKEN_ELEMS) {
              DMA_REPORT_ERROR("DMA TOKEN_IN->BUF: length must be 2048"
                               << " N=" << N
                               << " token_pos=" << token_pos.read());
              for (int t = 0; t < overhead; ++t)
                wait(clk.posedge_event());
              finish_txn();
              continue;
            }
          }
        }

        // Decide file offset
        std::uint64_t byte_off = 0;
        std::uint64_t need_bytes =
            static_cast<std::uint64_t>(N) * sizeof(float);

        if (!is_compiler) {
          if constexpr (MODE == 0) {
            const int pos1b_port = this->token_pos.read(); // authoritative
            const int pos1b_off =
                this->src_offset.read(); // optional debug mirror

            if (pos1b_port <= 0) {
              DMA_REPORT_ERROR(
                  "DMA TOKEN_IN->BUF: token_pos must be 1-based (>0)"
                  << " token_pos=" << pos1b_port);
              for (int t = 0; t < overhead; ++t)
                wait(clk.posedge_event());
              finish_txn();
              continue;
            }

            if (pos1b_off > 0 && pos1b_off != pos1b_port) {
              DMA_REPORT_ERROR("DMA TOKEN_IN->BUF: token_pos mismatch"
                               << " port=" << pos1b_port
                               << " offset=" << pos1b_off);
            }

            const std::uint64_t token_slot =
                static_cast<std::uint64_t>(pos1b_port - 1);
            byte_off = token_slot * TOKEN_BYTES_PER_TOKEN;
            need_bytes = TOKEN_BYTES_PER_TOKEN;
          } else {
            // MODE != 0:
            // src_offset is element offset inside TOKEN_I/input_tensor.bin.
            // Old Core_Controller legacy path passes 0, so old behavior is
            // preserved.
            const int file_elem_off = this->src_offset.read();

            if (file_elem_off < 0) {
              DMA_REPORT_ERROR(
                  "DMA TOKEN_IN->BUF: src_offset/token_io_offset must be >=0"
                  << " src_offset=" << file_elem_off);
              for (int t = 0; t < overhead; ++t)
                wait(clk.posedge_event());
              finish_txn();
              continue;
            }

            byte_off =
                static_cast<std::uint64_t>(file_elem_off) * sizeof(float);
            need_bytes = static_cast<std::uint64_t>(N) * sizeof(float);
          }
        } else {
          // COMPILER keeps original semantics: src_offset is 0-based float
          // index.
          const int file_elem_off = this->src_offset.read();

          if (file_elem_off < 0) {
            DMA_REPORT_ERROR("DMA COMPILER->BUF: src_offset must be >=0"
                             << " src_offset=" << file_elem_off);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();
            continue;
          }

          byte_off = static_cast<std::uint64_t>(file_elem_off) * sizeof(float);
          need_bytes = static_cast<std::uint64_t>(N) * sizeof(float);
        }

        const std::string path = is_compiler ? compiler_full_path_()
                                             : token_full_path_(/*is_in=*/true);

        DMA_REPORT_INFO(
            "DMA Case: EXT_BIN->BUF"
            << " which=" << which << " mode=" << MODE
            << " token_pos(1b)=" << (is_compiler ? -1 : token_pos.read())
            << " byte_off=" << byte_off << " dst_buf=" << d_idx
            << " dst_offset=" << dst_off << " N=" << N << " path=" << path);

        auto *dst_shadow = shadow_bufs_[d_idx];
        bool ok = true;
        if (!dst_shadow)
          ok = false;
        if (N <= 0 || dst_off < 0)
          ok = false;
        if (ok && static_cast<std::size_t>(dst_off + N) > dst_shadow->size())
          ok = false;

        if (!ok) {
          DMA_REPORT_ERROR("DMA EXT_BIN->BUF: invalid args"
                           << " which=" << which << " dst_buf=" << d_idx
                           << " dst_off=" << dst_off << " N=" << N
                           << " dst_shadow=" << (dst_shadow ? "ok" : "null")
                           << " dst_shadow_size="
                           << (dst_shadow ? dst_shadow->size() : 0));
          for (int t = 0; t < overhead; ++t)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        }

        const int total_latency = latency_mem_to_sram_(N);
        for (int t = 0; t < total_latency; ++t)
          wait(clk.posedge_event());

        if (token_io_buf_.size() < static_cast<std::size_t>(N))
          token_io_buf_.resize(static_cast<std::size_t>(N));

        bool ready = false;

        if (is_compiler) {
          std::lock_guard<std::mutex> lk(compiler_mtx_);
          ready = open_compiler_bin_if_needed_();
          if (ready && compiler_file_) {
            std::fstream &f = *compiler_file_;
            f.clear();
            f.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
            if (!f) {
              DMA_REPORT_ERROR("DMA COMPILER->BUF: seekg failed"
                               << " path=" << path << " byte_off=" << byte_off);
              ready = false;
            } else {
              f.read(reinterpret_cast<char *>(token_io_buf_.data()),
                     static_cast<std::streamsize>(need_bytes));
              if (!f) {
                DMA_REPORT_ERROR("DMA COMPILER->BUF: read failed (fill 0)"
                                 << " path=" << path << " byte_off=" << byte_off
                                 << " bytes=" << need_bytes);
                ready = false;
              }
            }
          } else {
            ready = false;
          }
        } else {
          // TOKEN_IN: read-only
          std::lock_guard<std::mutex> lk(token_mtx_);
          ready =
              open_token_bin_if_needed_(/*is_in=*/true, /*for_write=*/false);
          if (ready && token_files_[token_file_index_(true)]) {
            std::fstream &f = *token_files_[token_file_index_(true)];
            f.clear();
            f.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
            if (!f) {
              DMA_REPORT_ERROR("DMA TOKEN_IN->BUF: seekg failed"
                               << " path=" << path << " byte_off=" << byte_off);
              ready = false;
            } else {
              f.read(reinterpret_cast<char *>(token_io_buf_.data()),
                     static_cast<std::streamsize>(need_bytes));
              if (!f) {
                DMA_REPORT_ERROR("DMA TOKEN_IN->BUF: read failed (fill 0)"
                                 << " path=" << path << " byte_off=" << byte_off
                                 << " bytes=" << need_bytes);
                ready = false;
              }
            }
          } else {
            ready = false;
          }
        }

        if (ready) {
          for (int i = 0; i < N; ++i) {
            (*dst_shadow)[static_cast<std::size_t>(dst_off + i)] =
                static_cast<T>(token_io_buf_[static_cast<std::size_t>(i)]);
          }
        } else {
          for (int i = 0; i < N; ++i) {
            (*dst_shadow)[static_cast<std::size_t>(dst_off + i)] =
                static_cast<T>(0);
          }
        }

        finish_txn();
        continue;
      }
      // =========================================================
      // Case A3: Buffer -> TOKEN_OUT(FILE, FP32)
      //   - token_pos decides which token slot
      //   - each token slot = 2048 FP32
      //   - length must be 2048
      // =========================================================
      if (s_idx < bufs_.size() && (d_idx == tcore_sel::SEL_TOKEN_OUT ||
                                   d_idx == tcore_sel::SEL_COMPILER)) {

        static_assert(std::is_arithmetic<T>::value,
                      "DMA BUF->TOKEN_BIN: T must be arithmetic");

        const bool is_compiler = (d_idx == tcore_sel::SEL_COMPILER);
        const char *which = is_compiler ? "COMPILER" : "OUT";

        const int src_off = this->src_offset.read();
        const int N = L;

        // TOKEN_OUT must be exactly 2048
        if (!is_compiler) {
          if constexpr (MODE == 0) {
            if (N != TOKEN_ELEMS) {
              DMA_REPORT_ERROR("DMA BUF->TOKEN_OUT: length must be 2048"
                               << " N=" << N
                               << " token_pos=" << token_pos.read());
              for (int t = 0; t < overhead; ++t)
                wait(clk.posedge_event());
              finish_txn();
              continue;
            }
          }
        }

        std::uint64_t byte_off = 0;
        std::uint64_t need_bytes =
            static_cast<std::uint64_t>(N) * sizeof(float);

        if (!is_compiler) {
          if constexpr (MODE == 0) {
            const int pos1b_port = this->token_pos.read(); // authoritative
            const int pos1b_off =
                this->dst_offset.read(); // optional debug mirror

            if (pos1b_port <= 0) {
              DMA_REPORT_ERROR(
                  "DMA BUF->TOKEN_OUT: token_pos must be 1-based (>0)"
                  << " token_pos=" << pos1b_port);
              for (int t = 0; t < overhead; ++t)
                wait(clk.posedge_event());
              finish_txn();
              continue;
            }

            if (pos1b_off > 0 && pos1b_off != pos1b_port) {
              DMA_REPORT_ERROR("DMA BUF->TOKEN_OUT: token_pos mismatch"
                               << " port=" << pos1b_port
                               << " offset=" << pos1b_off);
            }

            const std::uint64_t token_slot =
                static_cast<std::uint64_t>(pos1b_port - 1);
            byte_off = token_slot * TOKEN_BYTES_PER_TOKEN;
            need_bytes = TOKEN_BYTES_PER_TOKEN;
          } else {
            // MODE != 0:
            // dst_offset is element offset inside TOKEN_O/output_tensor.bin.
            // Old Core_Controller legacy path passes 0, so old behavior is
            // preserved.
            const int file_elem_off = this->dst_offset.read();

            if (file_elem_off < 0) {
              DMA_REPORT_ERROR(
                  "DMA BUF->TOKEN_OUT: dst_offset/token_io_offset must be >=0"
                  << " dst_offset=" << file_elem_off);
              for (int t = 0; t < overhead; ++t)
                wait(clk.posedge_event());
              finish_txn();
              continue;
            }

            byte_off =
                static_cast<std::uint64_t>(file_elem_off) * sizeof(float);
            need_bytes = static_cast<std::uint64_t>(N) * sizeof(float);
          }
        } else {
          // COMPILER keeps original semantics: dst_offset is 0-based float
          // index.
          const int file_elem_off = this->dst_offset.read();

          if (file_elem_off < 0) {
            DMA_REPORT_ERROR("DMA BUF->COMPILER: dst_offset must be >=0"
                             << " dst_offset=" << file_elem_off);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();
            continue;
          }

          byte_off = static_cast<std::uint64_t>(file_elem_off) * sizeof(float);
          need_bytes = static_cast<std::uint64_t>(N) * sizeof(float);
        }

        auto *src_buf = bufs_[s_idx];
        const std::string path = is_compiler
                                     ? compiler_full_path_()
                                     : token_full_path_(/*is_in=*/false);

        DMA_REPORT_INFO(
            "DMA Case: BUF->EXT_BIN"
            << " which=" << which << " mode=" << MODE
            << " token_pos(1b)=" << (is_compiler ? -1 : token_pos.read())
            << " byte_off=" << byte_off << " src_buf=" << s_idx
            << " src_offset=" << src_off << " N=" << N << " path=" << path);

        bool ok = true;
        if (!src_buf)
          ok = false;
        if (N <= 0 || src_off < 0)
          ok = false;
        if (ok) {
          const std::size_t ssz = src_buf->size();
          if (static_cast<std::size_t>(src_off + N) > ssz)
            ok = false;
        }

        if (!ok) {
          DMA_REPORT_ERROR("DMA BUF->EXT_BIN: invalid args"
                           << " which=" << which << " src_buf=" << s_idx
                           << " src_off=" << src_off << " N=" << N
                           << " src_buf_ptr=" << (src_buf ? "ok" : "null"));
          for (int t = 0; t < overhead; ++t)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        }

        // latency model
        const int total_latency = latency_sram_to_mem_(N);
        for (int t = 0; t < total_latency; ++t)
          wait(clk.posedge_event());

        if (token_io_buf_.size() < static_cast<std::size_t>(N))
          token_io_buf_.resize(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
          token_io_buf_[static_cast<std::size_t>(i)] = static_cast<float>(
              (*src_buf)[static_cast<std::size_t>(src_off + i)].read());
        }

        bool ready = false;
        bool write_ok = true;

        if (is_compiler) {
          std::lock_guard<std::mutex> lk(compiler_mtx_);
          ready = open_compiler_bin_if_needed_();
          if (ready && compiler_file_) {
            std::fstream &f = *compiler_file_;
            ensure_file_size_(f, byte_off + need_bytes);
            f.clear();
            f.seekp(static_cast<std::streamoff>(byte_off), std::ios::beg);
            if (!f) {
              DMA_REPORT_ERROR("DMA BUF->COMPILER: seekp failed"
                               << " path=" << path << " byte_off=" << byte_off);
              write_ok = false;
            } else {
              f.write(reinterpret_cast<const char *>(token_io_buf_.data()),
                      static_cast<std::streamsize>(need_bytes));
              f.flush();
              if (!f)
                write_ok = false;
            }
          } else {
            ready = false;
            write_ok = false;
          }
        } else {
          // TOKEN_OUT: write-only (we still open in|out to avoid truncation)
          std::lock_guard<std::mutex> lk(token_mtx_);
          ready =
              open_token_bin_if_needed_(/*is_in=*/false, /*for_write=*/true);
          if (ready && token_files_[token_file_index_(false)]) {
            std::fstream &f = *token_files_[token_file_index_(false)];
            ensure_file_size_(f, byte_off + need_bytes);
            f.clear();
            f.seekp(static_cast<std::streamoff>(byte_off), std::ios::beg);
            if (!f) {
              DMA_REPORT_ERROR("DMA BUF->TOKEN_OUT: seekp failed"
                               << " path=" << path << " byte_off=" << byte_off);
              write_ok = false;
            } else {
              f.write(reinterpret_cast<const char *>(token_io_buf_.data()),
                      static_cast<std::streamsize>(need_bytes));
              f.flush();
              if (!f) {
                write_ok = false;
              } else {
                if constexpr (MODE == 5) {
                  const std::string mirror_path =
                      fp1_outcome_output_tensor_path_();

                  const bool mirror_ok = write_fp32_mirror_file_(
                      mirror_path, token_io_buf_.data(), byte_off, need_bytes);

                  if (!mirror_ok) {
                    DMA_REPORT_ERROR("DMA BUF->TOKEN_OUT: mirror write failed"
                                     << " mirror_path=" << mirror_path
                                     << " byte_off=" << byte_off
                                     << " bytes=" << need_bytes);
                  } else {
                    DMA_REPORT_INFO("DMA BUF->TOKEN_OUT: mirror write ok"
                                    << " mirror_path=" << mirror_path
                                    << " byte_off=" << byte_off
                                    << " bytes=" << need_bytes);
                  }
                }
              }
            }
          } else {
            ready = false;
            write_ok = false;
          }
        }
        if (!ready) {
          DMA_REPORT_ERROR("DMA BUF->EXT_BIN: file not ready"
                           << " which=" << which << " path=" << path);
        } else if (!write_ok) {
          DMA_REPORT_ERROR("DMA BUF->EXT_BIN: write failed"
                           << " which=" << which << " path=" << path);
        }

        finish_txn();
        continue;
      }
      // =========================================================
      // Case B: Buffer -> Buffer  (length = elements)
      // =========================================================
      if (s_idx < bufs_.size() && d_idx < bufs_.size() && s_idx != d_idx) {
        DMA_REPORT_INFO("DMA Case: BUF->BUF"
                        << " src_buf=" << s_idx << " dst_buf=" << d_idx
                        << " src_offset=" << src_offset.read() << " dst_offset="
                        << dst_offset.read() << " N(L)=" << L);

        const int src_off = src_offset.read(); // 來源 buffer 的元素起始索引
        const int dst_off = dst_offset.read(); // 目的 buffer 的元素起始索引
        const int N = L;                       // 要搬的元素數

        // --- 在 BUF->BUF case 一進來就先抓指標 ---
        auto *src_buf = bufs_[s_idx];
        auto *dst_shadow = shadow_bufs_[d_idx];

        bool ok = true;
        if (!src_buf || !dst_shadow) {
          ok = false;
        } else {

          if (N <= 0 || src_off < 0 || dst_off < 0) {
            ok = false;
          } else {
            const std::size_t ssz = src_buf->size();
            const std::size_t dsz =
                dst_shadow->size(); // ✅ 用 shadow 的 size 當目的端界線
            if (static_cast<std::size_t>(src_off) >= ssz ||
                static_cast<std::size_t>(src_off + N) > ssz ||
                static_cast<std::size_t>(dst_off) >= dsz ||
                static_cast<std::size_t>(dst_off + N) > dsz) {
              ok = false;
            }
          }
        }

        if (!ok) {
          DMA_REPORT_ERROR("DMA BUF->BUF: invalid arguments; skip"
                           << " src_buf=" << s_idx << " dst_buf=" << d_idx
                           << " src_offset=" << src_off
                           << " dst_offset=" << dst_off << " N(L)=" << L
                           << " src_buf=" << (src_buf ? "ok" : "null")
                           << " dst_shadow=" << (dst_shadow ? "ok" : "null"));
          for (int i = 0; i < overhead; ++i)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        }

        const std::size_t dshadow = dst_shadow->size();
        if (static_cast<std::size_t>(dst_off) >= dshadow ||
            static_cast<std::size_t>(dst_off + N) > dshadow) {
          DMA_REPORT_ERROR("DMA BUF->BUF: dst_shadow out of range; skip"
                           << " dst_buf=" << d_idx << " dst_offset=" << dst_off
                           << " N=" << N << " dst_shadow_size=" << dshadow);
          for (int i = 0; i < overhead; ++i)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        }
        const int total_latency = latency_sram_to_sram_(N);
        for (int t = 0; t < total_latency; ++t)
          wait(clk.posedge_event());

        // ★★★ 實際搬元素：從 sc_signal 讀，寫到 shadow buffer ★★★
        for (int i = 0; i < N; ++i) {
          T v = (*src_buf)[static_cast<std::size_t>(src_off + i)].read();
          (*dst_shadow)[static_cast<std::size_t>(dst_off + i)] = v;
        }

        finish_txn();
        continue;
      }

      // =========================================================
      // Case KV (Scheme-B): Buffer -> KV File (FP32), length = elements
      //   - dst_sel = tcore_sel::SEL_KV_K or tcore_sel::SEL_KV_V
      //   - token_pos / kv_is_prev / kv_head 由 Core 提供
      //   - length must be 64 (one head)
      //   contiguous)
      // =========================================================
      // ---------------- KV (Scheme-B): Buffer -> KV File ----------------
      if (s_idx < bufs_.size() &&
          (d_idx == tcore_sel::SEL_KV_K || d_idx == tcore_sel::SEL_KV_V)) {
        // KV path 支援用 dst_offset 傳遞 token_pos snapshot：
        //   - >0 : 以 dst_offset 為準（建議 Core 端傳入）
        //   - <=0: fallback 到 token_pos port（舊行為）
        const int token_pos_snap_1b = dst_offset.read();
        if constexpr (!std::is_same_v<T, float>) {
          DMA_REPORT_ERROR("DMA KVFILE: requires T=float (FP32).");
          for (int t = 0; t < overhead; ++t)
            wait(clk.posedge_event());
          finish_txn();
          continue;
        } else {
          const bool is_k = (d_idx == tcore_sel::SEL_KV_K);

          // layer comes from kv_layer port (aligned with Core_Controller) const
          int layer = kv_layer.read();
          const int head0 = kv_head.read();

          // Step1: token_pos（1-based）
          //   優先使用 Core 傳入的 snapshot，避免跨 cycle 讀到變動中的
          //   token_pos。
          const int token_pos_port_1b = this->token_pos.read();
          int token_pos_1b =
              (token_pos_snap_1b > 0) ? token_pos_snap_1b : token_pos_port_1b;

          if (token_pos_snap_1b > 0 && token_pos_snap_1b != token_pos_port_1b) {
            DMA_REPORT_WARNING(
                "DMA KVFILE: token_pos snapshot differs from port"
                << " snapshot=" << token_pos_snap_1b
                << " port=" << token_pos_port_1b << " -> use snapshot");
          }

          // Step1: 是否寫 prev token 由 Core 的 kv_is_prev 提供
          const bool write_prev = kv_is_prev.read();

          const int src_off = src_offset.read();
          const int N = L; // elements

          // --- basic checks: layer/head ---
          if (layer < 0 || layer >= kvfile::NUM_LAYERS) {
            DMA_REPORT_ERROR("DMA KVFILE: layer out of range"
                             << " layer=" << layer);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }
          if (head0 < 0 || head0 >= kvfile::NUM_HEADS) {
            DMA_REPORT_ERROR("DMA KVFILE: kv_head out of range"
                             << " head0=" << head0);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }

          // --- token_pos_1b is 1-based by ISA spec ---
          if (token_pos_1b <= 0) {
            DMA_REPORT_ERROR("DMA KVFILE: token_pos must be 1-based and > 0"
                             << " token_pos=" << token_pos_1b);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }
          if (token_pos_1b > kvfile::MAX_SEQ) {
            DMA_REPORT_ERROR("DMA KVFILE: token_pos exceeds KV_MAX_SEQ"
                             << " token_pos=" << token_pos_1b
                             << " KV_MAX_SEQ=" << kvfile::MAX_SEQ);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }

          // write_prev can make target_token = 0 when token_pos_1b == 1
          const int target_token =
              write_prev ? (token_pos_1b - 1) : token_pos_1b;
          if (target_token <= 0) {
            DMA_REPORT_ERROR("DMA KVFILE: target_token invalid"
                             << " token_pos=" << token_pos_1b << " write_prev="
                             << write_prev << " target_token=" << target_token);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }

          // convert to 0-based ONLY for storage indexing (addressing)
          const int token_slot = target_token - 1; // 0..KV_MAX_SEQ-1

          if (N != 64) {
            DMA_REPORT_ERROR("DMA KVFILE: length must be 64 (head_dim)"
                             << " N=" << N << " head_dim=" << KV_HEAD_DIM);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }

          const int heads_to_write = N / KV_HEAD_DIM;

          auto *src_buf = bufs_[s_idx];
          if (!src_buf) {
            DMA_REPORT_ERROR("DMA KVFILE: src_buf is nullptr"
                             << " src_buf=" << s_idx);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }
          const std::size_t ssz = src_buf->size();
          if (src_off < 0 || static_cast<std::size_t>(src_off + N) > ssz) {
            DMA_REPORT_ERROR("DMA KVFILE: source range out of buffer"
                             << " src_buf=" << src_buf << " src_offset="
                             << src_off << " N=" << N << " ssz=" << ssz);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }

          // latency model...
          const int total_latency = latency_sram_to_mem_(N);
          for (int t = 0; t < total_latency; ++t)
            wait(clk.posedge_event());

          // ---------------------------------------------------------
          // KV layout: [token_slot][head][dim]
          // ---------------------------------------------------------
          const std::uint64_t entry0 =
              static_cast<std::uint64_t>(token_slot) * KV_NUM_HEADS +
              static_cast<std::uint64_t>(head0);

          const std::uint64_t byte_off0 =
              kvfile::byte_offset_vec(target_token, head0);
          const std::uint64_t bytes =
              static_cast<std::uint64_t>(N) * kvfile::ELEM_BYTES;
          if (byte_off0 == std::numeric_limits<std::uint64_t>::max()) {
            DMA_REPORT_ERROR("DMA KVFILE: byte_offset_vec failed");
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }
          // Safety: must not exceed file capacity
          if (byte_off0 + bytes > kvfile::FILE_BYTES) {
            DMA_REPORT_ERROR("DMA KVFILE: write range exceeds KV_FILE_BYTES"
                             << " layer=" << layer << " K=" << is_k
                             << " byte_off0=" << byte_off0 << " bytes=" << bytes
                             << " KV_FILE_BYTES=" << kvfile::FILE_BYTES
                             << " token_slot=" << token_slot << " head0="
                             << head0 << " heads_to_write=" << heads_to_write);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }

          // (optional) conflict guard
          // (你原本那段可保留；若你前面已保留就不用再貼)

          // Pack N floats from src_buf into contiguous temp buffer
          std::vector<T> tmp_bulk;
          tmp_bulk.resize(static_cast<std::size_t>(N));
          for (int i = 0; i < N; ++i) {
            tmp_bulk[static_cast<std::size_t>(i)] =
                (*src_buf)[static_cast<std::size_t>(src_off + i)].read();
          }

          const std::string path = kv_make_path_(is_k, layer);

          DMA_REPORT_INFO(
              "DMA Case: BUF->KVFILE"
              << " K=" << is_k << " layer=" << layer << " token_pos(1b)="
              << token_pos_1b << " target_token(1b)=" << target_token
              << " token_slot(0b)=" << token_slot << " head0=" << head0
              << " heads_to_write=" << heads_to_write << " src_buf=" << s_idx
              << " src_offset=" << src_off << " N=" << N << " file=" << path);

          bool file_ready = true;
          bool write_ok = true;

          // lock per (K/V, layer)
          {
            const int fidx = kv_file_index_(is_k, layer);
            std::lock_guard<std::mutex> lk_file(kv_file_mtx_[fidx]);

            std::fstream &f = kv_open_file_(is_k, layer);
            if (!f) {
              file_ready = false;
            } else {
              kv_ensure_size_(f, byte_off0 +
                                     (std::uint64_t)N * kvfile::ELEM_BYTES);

              f.clear();
              f.seekp(static_cast<std::streamoff>(byte_off0), std::ios::beg);
              f.write(reinterpret_cast<const char *>(tmp_bulk.data()),
                      static_cast<std::streamsize>(
                          static_cast<std::uint64_t>(N) * sizeof(T)));
              f.flush();

              if (!f)
                write_ok = false;
            }
          } // unlock

          if (!file_ready) {
            DMA_REPORT_ERROR("DMA KVFILE: file not ready"
                             << " layer=" << layer << " K=" << is_k
                             << " path=" << path);
            for (int t = 0; t < overhead; ++t)
              wait(clk.posedge_event());
            finish_txn();

            continue;
          }

          if (!write_ok) {
            DMA_REPORT_ERROR("DMA KVFILE: write failed"
                             << " layer=" << layer << " K=" << is_k
                             << " token_slot(0b)=" << token_slot << " head0="
                             << head0 << " heads_to_write=" << heads_to_write
                             << " byte_off0=" << byte_off0 << " bytes="
                             << (static_cast<std::uint64_t>(N) * sizeof(T))
                             << " file=" << path);
          } else {
            DMA_REPORT_INFO(
                "KVFILE bulk write"
                << " layer=" << layer << " K=" << is_k
                << " token_slot(0b)=" << token_slot << " head0=" << head0
                << " heads_to_write=" << heads_to_write << " entry0=" << entry0
                << " byte_off0=" << byte_off0
                << " bytes=" << (static_cast<std::uint64_t>(N) * sizeof(T)));
          }

          finish_txn();
          continue;
        }
      }
      // =========================================================
      // 其它不支援的組合
      // =========================================================
      DMA_REPORT_ERROR("DMA: unsupported src/dst combination"
                       << " src_buf=" << s_idx << " dst_buf=" << d_idx
                       << " src_offset=" << src_offset.read()
                       << " dst_offset=" << dst_offset.read() << " L=" << L);
      for (int i = 0; i < overhead; ++i)
        wait(clk.posedge_event());
      finish_txn();
      continue;

    } // while
  }   // run()
  static constexpr int kNumBufs = ActiveHWConfig::NUM_BUFS;
  inline static constexpr auto kBufCaps = ActiveHWConfig::BUF_SIZES;

  static constexpr int kDramBw = ActiveHWConfig::DRAM_LANES;
  static constexpr int kSramBw = ActiveHWConfig::SRAM_DMA_WIDTH;

  static_assert(kNumBufs == 12, "DMA_Controller expects NUM_BUFS == 12.");
  static_assert(kDramBw > 0, "DRAM_LANES must be > 0.");
  static_assert(kSramBw > 0, "SRAM_DMA_WIDTH must be > 0.");

private:
  static int ceil_div_pos_(int a, int b) { return (a + b - 1) / b; }

  static int bw_mem_sram_() { return std::min(kDramBw, kSramBw); }

  static int bw_sram_sram_() { return kSramBw; }

  static int latency_mem_to_sram_(int n) {
    return overhead + ceil_div_pos_(n, bw_mem_sram_());
  }

  static int latency_sram_to_mem_(int n) {
    return overhead + ceil_div_pos_(n, bw_mem_sram_());
  }

  static int latency_sram_to_sram_(int n) {
    return overhead + ceil_div_pos_(n, bw_sram_sram_());
  }
  inline static std::array<std::mutex, 2 * kvfile::NUM_LAYERS> kv_file_mtx_;
  std::array<std::unique_ptr<std::fstream>, 2 * kvfile::NUM_LAYERS> kv_files_;
  std::string kv_root_dir_ = kvfile::DEFAULT_ROOT_DIR;
  static constexpr int TOKEN_ELEMS = 2048;
  static constexpr std::uint64_t TOKEN_BYTES_PER_TOKEN =
      static_cast<std::uint64_t>(TOKEN_ELEMS) * sizeof(float);
  std::ifstream model_bin_fin_;
  std::string model_cur_bin_path_;

  // stream buffer cache (optional)
  static constexpr std::size_t kStreamBufBytes = (std::size_t{1} << 20);
  std::vector<char> model_bin_buf_ = std::vector<char>(kStreamBufBytes);
  std::vector<char> token_bin_buf_ = std::vector<char>(kStreamBufBytes);

  // temp buffers
  std::vector<T> model_read_buf_;
  std::vector<float> token_io_buf_;
  // MODEL backend config/cache
  std::string model_dir_ = "../DRAM_json";     // 給 TOKEN/COMPILER 用
  std::string model_bin_dir_ = "../DRAM_json"; // 給 MODEL bin_file 用
  std::string manifest_path_ = "../DRAM_json/Llama_manifest.json";
  nlohmann::json model_manifest_;
  bool model_manifest_loaded_ = false;
  std::mutex model_mtx_;
  // 可變長度：指向多組 buffer 的指標（每組為一條 sc_vector<sc_signal<T>>）
  // 注意：順序定義了 src_sel/dst_sel 的語意！
  std::vector<sc_core::sc_vector<sc_core::sc_signal<T>> *> bufs_;

  struct ModelSlice {
    std::string bin_file;
    std::uint64_t off_bytes{0};
    std::uint64_t size_bytes{0};
  };
  void validate_registered_buffers_or_die_() {
    if (bufs_.size() != static_cast<std::size_t>(kNumBufs)) {
      std::ostringstream oss;
      oss << "DMA_Controller: bufs_.size() mismatch. expected=" << kNumBufs
          << " actual=" << bufs_.size();
      SC_REPORT_FATAL(this->name(), oss.str().c_str());
    }

    if (shadow_bufs_.size() != static_cast<std::size_t>(kNumBufs)) {
      std::ostringstream oss;
      oss << "DMA_Controller: shadow_bufs_.size() mismatch. expected="
          << kNumBufs << " actual=" << shadow_bufs_.size();
      SC_REPORT_FATAL(this->name(), oss.str().c_str());
    }

    for (int i = 0; i < kNumBufs; ++i) {
      if (!bufs_[i]) {
        std::ostringstream oss;
        oss << "DMA_Controller: bufs_[" << i << "] is nullptr";
        SC_REPORT_FATAL(this->name(), oss.str().c_str());
      }
      if (!shadow_bufs_[i]) {
        std::ostringstream oss;
        oss << "DMA_Controller: shadow_bufs_[" << i << "] is nullptr";
        SC_REPORT_FATAL(this->name(), oss.str().c_str());
      }

      const std::size_t exp = static_cast<std::size_t>(kBufCaps[i]);
      const std::size_t got_buf = bufs_[i]->size();
      const std::size_t got_shadow = shadow_bufs_[i]->size();

      if (got_buf != exp) {
        std::ostringstream oss;
        oss << "DMA_Controller: bufs_[" << i
            << "] size mismatch. expected=" << exp << " actual=" << got_buf;
        SC_REPORT_FATAL(this->name(), oss.str().c_str());
      }

      if (got_shadow != exp) {
        std::ostringstream oss;
        oss << "DMA_Controller: shadow_bufs_[" << i
            << "] size mismatch. expected=" << exp << " actual=" << got_shadow;
        SC_REPORT_FATAL(this->name(), oss.str().c_str());
      }
    }
  }
#ifndef FP1_SUBMISSION_ROOT
#define FP1_SUBMISSION_ROOT "/home/wilson/2026_Spring_CA/FP1/submission"
#endif

#ifndef FP1_OUTCOME_ROOT
#define FP1_OUTCOME_ROOT "/home/wilson/2026_Spring_CA/FP1/outcome"
#endif

#ifndef FP1_GROUP_NUM
#define FP1_GROUP_NUM 1
#endif

#if FP1_GROUP_NUM < 1
#error "FP1_GROUP_NUM must be >= 1"
#endif

  static std::string fp1_group_suffix_() {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "G%02d", FP1_GROUP_NUM);
    return std::string(buf);
  }

  static std::string fp1_group_dir_() {
    const std::string g = fp1_group_suffix_();
    return std::string(FP1_SUBMISSION_ROOT) + "/CA_FP1_" + g;
  }

  static std::string fp1_manifest_path_() {
    const std::string g = fp1_group_suffix_();
    return fp1_group_dir_() + "/mlp_dram_map_" + g + ".json";
  }
  static std::string fp1_outcome_group_dir_() {
    const std::string g = fp1_group_suffix_();
    return std::string(FP1_OUTCOME_ROOT) + "/CA_FP1_" + g;
  }

  static std::string fp1_outcome_output_tensor_path_() {
    return fp1_outcome_group_dir_() + "/output_tensor.bin";
  }
  static std::string join_path_(const std::string &dir,
                                const std::string &file) {
    if (!file.empty() && file[0] == '/')
      return file;
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
                              join_path_(asmap_default_run_dir_(), "dram"));
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
      if (path[p] == '_' && path[p + 1] == 'G' && path[p + 2] >= '0' &&
          path[p + 2] <= '9' && path[p + 3] >= '0' && path[p + 3] <= '9') {
        return path.substr(p + 1, 3);
      }
    }
    return "G01";
  }

  static inline std::string asmap_ref_dir_() {
    return getenv_or_default_("ASMAP_REF_DIR",
                              join_path_(asmap_default_run_dir_(), "ref"));
  }

  static inline std::string asmap_sim_out_dir_() {
    return getenv_or_default_("ASMAP_SIM_OUT_DIR",
                              join_path_(asmap_default_run_dir_(), "sim_out"));
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
      return join_path_(asmap_dram_dir_(), "Llama_manifest.json");
    } else if constexpr (MODE == 7) {
      const std::string group = mode7_group_suffix_();
      return join_path_(asmap_dram_dir_(), "memory_map_sim_" + group + ".json");
    } else {
      return "/home/wilson/SystemC/DRAM_json/Llama_manifest.json";
    }
  }

  static std::string
  default_model_bin_dir_by_mode_(const std::string &model_dir_fallback) {
    if constexpr (MODE == 0) {
      return model_dir_fallback;
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
      return model_dir_fallback;
    }
  }
  bool load_model_manifest_once_() {
    if (model_manifest_loaded_)
      return true;
    std::lock_guard<std::mutex> lk(model_mtx_);
    if (model_manifest_loaded_)
      return true;

    std::ifstream mf(manifest_path_);
    if (!mf.is_open()) {
      DMA_REPORT_ERROR("DMA MODEL: cannot open manifest: " << manifest_path_);
      return false;
    }
    try {
      mf >> model_manifest_;
      model_manifest_loaded_ = true;
    } catch (const std::exception &e) {
      DMA_REPORT_ERROR("DMA MODEL: manifest parse error: " << e.what());
      return false;
    }
    return true;
  }

  bool lookup_model_slice_(int layer, const std::string &weight, int slice_idx,
                           ModelSlice &out) {
    nlohmann::json *layer_obj = find_layer_obj_(layer);
    if (!layer_obj) {
      DMA_REPORT_ERROR("DMA MODEL: manifest missing layer id="
                       << layer << " (expect key layers.<id> or layers.<id>.)");
      return false;
    }

    auto it_w = layer_obj->find(weight);
    if (it_w == layer_obj->end()) {
      DMA_REPORT_ERROR("DMA MODEL: manifest missing weight_name="
                       << weight << " layer=" << layer);
      return false;
    }

    const auto &wobj = *it_w;
    if (!wobj.is_object()) {
      DMA_REPORT_ERROR("DMA MODEL: manifest weight entry not object"
                       << " weight=" << weight << " layer=" << layer);
      return false;
    }

    // 你的 manifest 是 "input_layernorm.weight": { "0": { ... } }
    // 也就是一定走 sliced case
    const std::string s = std::to_string(slice_idx);
    auto it_s = wobj.find(s);
    if (it_s == wobj.end()) {
      DMA_REPORT_ERROR("DMA MODEL: manifest missing slice_idx="
                       << s << " weight=" << weight << " layer=" << layer);
      return false;
    }

    try {
      out.bin_file = it_s->at("bin_file").get<std::string>();
      out.off_bytes = it_s->at("bin_offset_bytes").get<std::uint64_t>();
      out.size_bytes = it_s->at("bin_size_bytes").get<std::uint64_t>();
    } catch (const std::exception &e) {
      DMA_REPORT_ERROR("DMA MODEL: manifest entry parse error: " << e.what());
      return false;
    }
    return !out.bin_file.empty();
  }
  bool open_model_bin_if_needed_(const std::string &full_path) {
    if (model_bin_fin_.is_open() && model_cur_bin_path_ == full_path)
      return true;

    if (model_bin_fin_.is_open())
      model_bin_fin_.close();
    model_bin_fin_.clear();

    // optional: set read buffer
    if (!model_bin_buf_.empty()) {
      model_bin_fin_.rdbuf()->pubsetbuf(
          model_bin_buf_.data(),
          static_cast<std::streamsize>(model_bin_buf_.size()));
    }

    model_bin_fin_.open(full_path, std::ios::in | std::ios::binary);
    if (!model_bin_fin_.is_open()) {
      DMA_REPORT_ERROR("DMA MODEL: cannot open bin_file: " << full_path);
      return false;
    }
    model_cur_bin_path_ = full_path;
    return true;
  }
  // ===== TOKEN_IN / TOKEN_OUT bin backend (FP32) =====
  inline static std::mutex token_mtx_;
  std::array<std::unique_ptr<std::fstream>, 2> token_files_; // 0:IN, 1:OUT
  std::array<std::string, 2> token_cur_paths_{};             // cache full path

  static int token_file_index_(bool is_in) { return is_in ? 0 : 1; }

  std::string token_full_path_(bool is_in) const {
    if constexpr (MODE == 6) {
      if (is_in) {
        // generated by generate_llama_ffn_correctness_assets.py
        return join_path_(asmap_ref_dir_(), "input_tensor.bin");
        // 如果您之後想明確用 TOKEN_I.bin，也可以改成：
        // return join_path_(asmap_ref_dir_(), "TOKEN_I.bin");
      } else {
        return join_path_(asmap_sim_out_dir_(), "output_tensor.bin");
      }
    } else if constexpr (MODE == 7) {
      if (is_in) {
        return getenv_or_default_(
            "ASMAP_TOKEN_IN_PATH",
            "/home/wilson/SystemC/DRAM_json/input_tensor.bin");
      } else {
        return join_path_(asmap_sim_out_dir_(), "output_tensor.bin");
      }
    } else {
      return join_path_(model_dir_,
                        is_in ? "input_tensor.bin" : "output_tensor.bin");
    }
  }

  // ensure size for random write (like kv_ensure_size_ but generic)
  static void ensure_file_size_(std::fstream &f, std::uint64_t need_bytes) {
    if (!f)
      return;
    f.clear();
    f.seekp(0, std::ios::end);
    std::uint64_t cur = 0;
    auto tp = f.tellp();
    if (tp != std::streampos(-1))
      cur = static_cast<std::uint64_t>(tp);
    if (cur >= need_bytes)
      return;

    f.clear();
    f.seekp(static_cast<std::streamoff>(need_bytes - 1), std::ios::beg);
    char z = 0;
    f.write(&z, 1);
    f.flush();
  }
  bool write_fp32_mirror_file_(const std::string &full_path, const float *data,
                               std::uint64_t byte_off,
                               std::uint64_t need_bytes) {
    try {
      std::filesystem::path p(full_path);
      if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
      }
    } catch (const std::exception &e) {
      DMA_REPORT_ERROR("DMA mirror write: mkdir failed"
                       << " path=" << full_path << " what=" << e.what());
      return false;
    } catch (...) {
      DMA_REPORT_ERROR("DMA mirror write: mkdir failed (unknown)"
                       << " path=" << full_path);
      return false;
    }

    std::fstream f(full_path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) {
      std::ofstream create(full_path, std::ios::out | std::ios::binary);
      create.close();
      f.clear();
      f.open(full_path, std::ios::in | std::ios::out | std::ios::binary);
    }

    if (!f) {
      DMA_REPORT_ERROR("DMA mirror write: cannot open file"
                       << " path=" << full_path);
      return false;
    }

    ensure_file_size_(f, byte_off + need_bytes);

    f.clear();
    f.seekp(static_cast<std::streamoff>(byte_off), std::ios::beg);
    if (!f) {
      DMA_REPORT_ERROR("DMA mirror write: seekp failed"
                       << " path=" << full_path << " byte_off=" << byte_off);
      return false;
    }

    f.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(need_bytes));
    f.flush();

    if (!f) {
      DMA_REPORT_ERROR("DMA mirror write: write failed"
                       << " path=" << full_path << " bytes=" << need_bytes);
      return false;
    }

    return true;
  }

  // Open TOKEN file (IN: default read-only; OUT: read+write and auto-create)
  bool open_token_bin_if_needed_(bool is_in, bool for_write) {
    const int idx = token_file_index_(is_in);
    const std::string full_path = token_full_path_(is_in);
    if (is_in && for_write) {
      DMA_REPORT_ERROR("DMA TOKEN_IN: write is not allowed (read-only)");
      return false;
    }
    if (!is_in && !for_write) {
      DMA_REPORT_ERROR("DMA TOKEN_OUT: read is not allowed (write-only)");
      return false;
    }
    // already open and same path
    if (token_files_[idx] && token_files_[idx]->is_open() &&
        token_cur_paths_[idx] == full_path) {
      return true;
    }

    // close old
    if (token_files_[idx] && token_files_[idx]->is_open())
      token_files_[idx]->close();

    token_files_[idx] = std::make_unique<std::fstream>();
    std::fstream &f = *token_files_[idx];

    // optional stream buffer
    if (!token_bin_buf_.empty()) {
      f.rdbuf()->pubsetbuf(token_bin_buf_.data(),
                           static_cast<std::streamsize>(token_bin_buf_.size()));
    }

    if (is_in) {
      // input_tensor.bin：預期存在；讀為主
      std::ios::openmode mode = std::ios::binary | std::ios::in;
      if (for_write)
        mode = std::ios::binary | std::ios::in | std::ios::out;
      f.open(full_path, mode);
      if (!f) {
        DMA_REPORT_ERROR("DMA TOKEN_IN: cannot open " << full_path);
        return false;
      }
    } else {
      // output_tensor.bin：可不存在，幫你 create
      std::ios::openmode mode = std::ios::binary | std::ios::in | std::ios::out;
      f.open(full_path, mode);
      if (!f) {
        // create then reopen
        std::ofstream create(full_path, std::ios::out | std::ios::binary);
        create.close();
        f.clear();
        f.open(full_path, mode);
      }
      if (!f) {
        DMA_REPORT_ERROR("DMA TOKEN_OUT: cannot open " << full_path);
        return false;
      }
    }

    token_cur_paths_[idx] = full_path;
    return true;
  }
  // ===== COMPILER.bin backend (fixed existing file) =====
  inline static std::mutex compiler_mtx_;
  std::unique_ptr<std::fstream> compiler_file_;
  std::string compiler_cur_path_;

  std::string compiler_full_path_() const {
    if constexpr (MODE == 6) {
      return join_path_(asmap_dram_dir_(), "Compiler.bin");
    }
    // model_dir_ 預設就是 ../DRAM_json，所以這裡會變 ../DRAM_json/Compiler.bin
    return join_path_(model_dir_, "Compiler.bin");
  }

  bool open_compiler_bin_if_needed_() {
    const std::string full_path = compiler_full_path_();

    if (compiler_file_ && compiler_file_->is_open() &&
        compiler_cur_path_ == full_path) {
      return true;
    }

    if (compiler_file_ && compiler_file_->is_open())
      compiler_file_->close();

    compiler_file_ = std::make_unique<std::fstream>();
    std::fstream &f = *compiler_file_;

    // optional stream buffer（沿用 token_bin_buf_，你原本就有）
    if (!token_bin_buf_.empty()) {
      f.rdbuf()->pubsetbuf(token_bin_buf_.data(),
                           static_cast<std::streamsize>(token_bin_buf_.size()));
    }

    // 你說固定存在，所以直接用 in|out 開啟；打不開就報錯
    f.open(full_path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) {
      DMA_REPORT_ERROR("DMA COMPILER_BIN: cannot open "
                       << full_path << " (file must exist)");
      return false;
    }

    compiler_cur_path_ = full_path;
    return true;
  }
  // ★ 新增：對應每一個 buffer 的「shadow 寫入緩衝區」
  //   - DMA 只會寫這裡
  //   - 實際的 sc_signal 寫入由 Tcore 仲裁後統一寫入
  std::vector<std::vector<T> *> shadow_bufs_;
  // ===== Scheme-B KV file backend =====

  int kv_file_index_(bool is_k, int layer) const {
    return (is_k ? 0 : kvfile::NUM_LAYERS) + layer;
  }

  std::string kv_make_path_(bool is_k, int layer) const {
    return kvfile::make_path(kv_root_dir_, is_k, layer);
  }

  std::fstream &kv_open_file_(bool is_k, int layer) {
    const int idx = kv_file_index_(is_k, layer);
    if (kv_files_[idx] && kv_files_[idx]->is_open()) {
      return *kv_files_[idx];
    }

    // ensure directories exist: root/K and root/V
    try {
      std::filesystem::create_directories(kv_root_dir_ + "/K");
      std::filesystem::create_directories(kv_root_dir_ + "/V");
    } catch (const std::exception &e) {
      DMA_REPORT_ERROR("KV mkdir failed (std::exception): "
                       << e.what() << " root=" << kv_root_dir_);
    } catch (...) {
      DMA_REPORT_ERROR("KV mkdir failed (unknown exception)"
                       << " root=" << kv_root_dir_);
    }

    const std::string path = kv_make_path_(is_k, layer);

    // Try open existing (in|out). If not exist, create then reopen.
    auto f = std::make_unique<std::fstream>();
    f->open(path, std::ios::in | std::ios::out | std::ios::binary);

    if (!(*f)) {
      // create then reopen
      std::ofstream create(path, std::ios::out | std::ios::binary);
      create.close();

      f->clear(); // ✅ 清掉 failbit
      f->open(path, std::ios::in | std::ios::out | std::ios::binary);
    }

    if (!(*f)) {
      DMA_REPORT_ERROR("KV file open failed: path=" << path);
      // create a dummy stream object anyway to avoid crash; but will error on
      // use
    }

    kv_files_[idx] = std::move(f);
    // Pre-allocate file to fixed capacity so random seek/write/read never hits
    // EOF. Unwritten areas are 0x00 bytes => float 0.0f.
    if (kv_files_[idx] && (*kv_files_[idx])) {
      kv_ensure_size_(*kv_files_[idx], kvfile::FILE_BYTES);
    }
    return *kv_files_[idx];
  }
  // ===== KVFILE conflict detector (same timestamp overlap check) =====
  struct KvRange {
    std::uint64_t b; // begin (inclusive)
    std::uint64_t e; // end   (exclusive)
  };

  static bool kv_overlap_(const KvRange &x, const KvRange &y) {
    return !(x.e <= y.b || y.e <= x.b);
  }

  static std::uint32_t kv_filekey_(bool is_k, int layer) {
    // unique key for (K/V, layer)
    return (static_cast<std::uint32_t>(is_k) << 8) |
           static_cast<std::uint32_t>(layer & 0xFF);
  }
  static inline std::uint32_t fnv1a_32_(const std::string &s) {
    std::uint32_t h = 2166136261u;
    for (unsigned char c : s) {
      h ^= static_cast<std::uint32_t>(c);
      h *= 16777619u;
    }
    return h;
  }
  // ================================================================
  void kv_ensure_size_(std::fstream &f, std::uint64_t need_bytes) {
    if (!f)
      return;
    f.clear();
    f.seekp(0, std::ios::end);
    std::uint64_t cur = 0;
    auto tp = f.tellp();
    if (tp != std::streampos(-1))
      cur = static_cast<std::uint64_t>(tp);

    if (cur >= need_bytes)
      return;

    // extend file (may create sparse file)
    f.clear();
    f.seekp(static_cast<std::streamoff>(need_bytes - 1), std::ios::beg);
    char z = 0;
    f.write(&z, 1);
    f.flush();
  }
  // 找到對應 layer 的 json object：支援 "layers.0" 與 "layers.0."
  nlohmann::json *find_layer_obj_(int layer) {
    auto try_prefix = [&](const std::string &prefix) -> nlohmann::json * {
      const std::string k_plain = prefix + std::to_string(layer);
      const std::string k_dot = k_plain + ".";

      if (model_manifest_.contains(k_plain) &&
          model_manifest_[k_plain].is_object())
        return &model_manifest_[k_plain];

      if (model_manifest_.contains(k_dot) && model_manifest_[k_dot].is_object())
        return &model_manifest_[k_dot];

      for (auto it = model_manifest_.begin(); it != model_manifest_.end();
           ++it) {
        const std::string &k = it.key();
        if (k.rfind(prefix, 0) != 0)
          continue;

        size_t i = prefix.size();
        size_t j = i;
        while (j < k.size() && std::isdigit((unsigned char)k[j]))
          ++j;
        if (j == i)
          continue;

        int id = 0;
        try {
          id = std::stoi(k.substr(i, j - i));
        } catch (...) {
          continue;
        }

        if (id == layer && it.value().is_object())
          return &model_manifest_[k];
      }

      return nullptr;
    };

    if constexpr (MODE == 7) {
      if (auto *p = try_prefix("model."))
        return p;

      // fallback，方便舊 manifest debug
      return try_prefix("layers.");
    } else {
      if (auto *p = try_prefix("layers."))
        return p;

      // fallback，避免之後共用測試資料時太嚴格
      return try_prefix("model.");
    }
  }
};

#endif // DMA_CONTROLLER_H
