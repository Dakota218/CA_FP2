// /home/wilson/SystemC/include/Tcore.h
#ifndef TCORE_H
#define TCORE_H
#include "Array.h"
#include "Core_Controller.h"
#include "DMA_Controller.h"
#include "Router.h"
#include "Vector.h"
#include "hw_config_selector.h"
#include "mem_types.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sysc/utils/sc_report.h>
#include <systemc.h>
#include <vector>

template <typename T, int MODE = 0> class Tcore : public sc_core::sc_module {
public:
  // ===== Buffer 索引 =====
  static constexpr int BUF_ARRAY_IN = 0; // 2048
  static constexpr int BUF_ARRAY_OUT =
      1; // 2048  (compute output, 不建議 DMA/Router 寫)
  static constexpr int BUF_VEC_IN = 2; // 2048
  static constexpr int BUF_VEC_W = 3;  // 2048
  static constexpr int BUF_VEC_OUT =
      4; // 2048  (compute output, 不建議 DMA/Router 寫)
  static constexpr int BUF_OTHER_IN = 5;  // 512
  static constexpr int BUF_OTHER_OUT = 6; // 512

  static constexpr int BUF_K = 7;                // 2048
  static constexpr int BUF_V = 8;                // 2048
  static constexpr int BUF_Q_SV = 9;             // 2048
  static constexpr int BUF_RMS = 10;             // 2048
  static constexpr int BUF_CORE_CONTROLLER = 11; // 512

  static constexpr int NUM_BUFS = 12;
  static constexpr int NUM_DMA_CH = 2;

  using HW = ActiveHWConfig;
  static_assert(HW::NUM_BUFS == NUM_BUFS,
                "ActiveHWConfig::NUM_BUFS must be 12.");

  inline static constexpr std::array<int, NUM_BUFS> BUF_SIZES = HW::BUF_SIZES;

  static constexpr int BUF_ARRAY_IN_CAP = HW::BUF_ARRAY_IN_SIZE;
  static constexpr int BUF_ARRAY_OUT_CAP = HW::BUF_ARRAY_OUT_SIZE;
  static constexpr int BUF_VEC_IN_CAP = HW::BUF_VEC_IN_SIZE;
  static constexpr int BUF_VEC_W_CAP = HW::BUF_VEC_W_SIZE;
  static constexpr int BUF_VEC_OUT_CAP = HW::BUF_VEC_OUT_SIZE;
  static constexpr int BUF_K_CAP = HW::BUF_K_SIZE;
  static constexpr int BUF_V_CAP = HW::BUF_V_SIZE;

  static constexpr int VECTOR_IN_CAP = HW::BUF_VEC_IN_SIZE;
  static constexpr int VECTOR_W_CAP = HW::BUF_VEC_W_SIZE;
  static constexpr int VECTOR_OUT_CAP = HW::BUF_VEC_OUT_SIZE;
  // ===== Ports =====
  sc_core::sc_in<bool> clk;

  // NoC ports
  sc_core::sc_fifo_in<NocCmd<T>> noc_cmd_in;
  sc_core::sc_fifo_out<NocCmd<T>> noc_cmd_out;
  sc_core::sc_fifo_in<T> noc_data_in;
  sc_core::sc_fifo_out<T> noc_data_out;

  sc_core::sc_out<bool> all_done;

  // monitors
  sc_core::sc_out<bool> mon_vector_busy{"mon_vector_busy"};
  sc_core::sc_out<bool> mon_array_busy{"mon_array_busy"};
  sc_core::sc_out<bool> mon_dma_busy{"mon_dma_busy"};
  sc_core::sc_out<bool> mon_Router_send_busy{"mon_Router_send_busy"};
  sc_core::sc_out<bool> mon_Router_recv_busy{"mon_Router_recv_busy"};
  // DMA DRAM busy (OR of 3 channels)
  sc_core::sc_out<bool> mon_dram_rd_busy{"mon_dram_rd_busy"};
  sc_core::sc_out<bool> mon_dram_wr_busy{"mon_dram_wr_busy"};

  // ===== submodules =====
  DMA_Controller<T, 8, 8, 5, MODE> *dma[NUM_DMA_CH]{};
  Array<T, MODE> *array{nullptr};
  Vector<T, 0, MODE> *vector_unit{nullptr};
  Core_Controller<MODE> *controller{nullptr};
  Router<T, MODE> *router{nullptr};

  // ===== real buffers =====
  std::array<std::unique_ptr<sc_core::sc_vector<sc_core::sc_signal<T>>>,
             NUM_BUFS>
      bufs;

  // ===== unified shadow buffers (DMA + Router share the SAME one) =====
  std::vector<std::vector<T>> shadow_buf;

  // ===== Controller <-> Vector =====
  sc_core::sc_signal<bool> sig_vector_start, sig_vector_busy, sig_vector_done;
  sc_core::sc_signal<int> sig_vector_mode, sig_vector_DIM;
  sc_core::sc_signal<int> sig_vector_step;

  // ===== Controller <-> Array (NEW interface) =====
  sc_core::sc_signal<bool> sig_Array_start, sig_Array_busy, sig_Array_done;
  sc_core::sc_signal<sc_dt::sc_uint<2>> sig_Array_op;
  sc_core::sc_signal<ArrayKey> sig_Array_key;
  sc_core::sc_signal<int> sig_Array_row, sig_Array_col;

  // token_pos：改成「由 Tcore state 驅動」（Array/DMA 用）
  sc_core::sc_signal<int> sig_token_pos;

  // ===== Controller <-> DMA (NEW) =====
  sc_core::sc_signal<bool> sig_dma_start[NUM_DMA_CH];
  sc_core::sc_signal<bool> sig_dma_busy[NUM_DMA_CH];
  sc_core::sc_signal<bool> sig_dma_done[NUM_DMA_CH];

  // ===== DMA -> Tcore (DRAM busy, per channel) =====
  sc_core::sc_signal<bool> sig_dma_dram_rd_busy[NUM_DMA_CH];
  sc_core::sc_signal<bool> sig_dma_dram_wr_busy[NUM_DMA_CH];

  sc_core::sc_signal<sc_dt::sc_uint<5>> sig_dma_src_sel[NUM_DMA_CH];
  sc_core::sc_signal<sc_dt::sc_uint<5>> sig_dma_dst_sel[NUM_DMA_CH];
  sc_core::sc_signal<int> sig_dma_src_offset[NUM_DMA_CH];
  sc_core::sc_signal<int> sig_dma_dst_offset[NUM_DMA_CH];
  sc_core::sc_signal<int> sig_dma_length[NUM_DMA_CH];

  // KV / model metadata (per channel)
  sc_core::sc_signal<int> sig_dma_kv_head[NUM_DMA_CH];
  sc_core::sc_signal<bool> sig_dma_kv_is_prev[NUM_DMA_CH];

  sc_core::sc_signal<int> sig_dma_kv_layer[NUM_DMA_CH];
  sc_core::sc_signal<ArrayKey> sig_dma_model_key[NUM_DMA_CH];

  // ===== Router wires =====
  sc_core::sc_signal<sc_dt::sc_uint<5>> tcore_x;
  sc_core::sc_signal<sc_dt::sc_uint<5>> tcore_y;

  sc_core::sc_signal<bool> sig_Router_start;
  sc_core::sc_signal<bool> sig_Router_is_send;
  sc_core::sc_signal<sc_dt::sc_uint<4>> sig_Router_src_buf;
  sc_core::sc_signal<sc_dt::sc_uint<4>> sig_Router_dst_buf;
  sc_core::sc_signal<sc_dt::sc_uint<16>>
      sig_Router_src_offset; // ✅ idx -> offset
  sc_core::sc_signal<sc_dt::sc_uint<16>>
      sig_Router_dst_offset; // ✅ idx -> offset
  sc_core::sc_signal<sc_dt::sc_uint<16>> sig_Router_length;
  sc_core::sc_signal<sc_dt::sc_uint<5>> sig_Router_dst_x;
  sc_core::sc_signal<sc_dt::sc_uint<5>> sig_Router_dst_y;

  static constexpr int MAX_MCAST_DESTS = Core_Controller<MODE>::MAX_MCAST_DESTS;
  sc_core::sc_signal<bool> sig_Router_is_mcast;
  sc_core::sc_signal<sc_dt::sc_uint<7>> sig_Router_mcast_num;
  sc_core::sc_signal<sc_dt::sc_uint<5>> sig_Router_mcast_dst_x[MAX_MCAST_DESTS];
  sc_core::sc_signal<sc_dt::sc_uint<5>> sig_Router_mcast_dst_y[MAX_MCAST_DESTS];
  sc_core::sc_signal<sc_dt::sc_uint<4>>
      sig_Router_mcast_dst_buf[MAX_MCAST_DESTS];
  sc_core::sc_signal<sc_dt::sc_uint<16>>
      sig_Router_mcast_dst_offset[MAX_MCAST_DESTS]; // ✅ idx -> offset

  sc_core::sc_signal<bool> sig_Router_send_busy;
  sc_core::sc_signal<bool> sig_Router_send_done;
  sc_core::sc_signal<bool> sig_Router_recv_busy;
  sc_core::sc_signal<bool> sig_Router_recv_done;

  sc_core::sc_signal<sc_dt::sc_uint<4>> sig_Router_recv_dst_buf;
  sc_core::sc_signal<bool> sig_Router_recv_dst_valid;
  sc_core::sc_signal<bool> sig_Router_recv_buf_grant;
  sc_core::sc_signal<sc_dt::sc_uint<32>> sig_Router_hash;

  struct DmaCommitMeta {
    int dst_buf = -1, off = 0, len = 0;
  };
  DmaCommitMeta last_dma_meta[NUM_DMA_CH];

  SC_HAS_PROCESS(Tcore);

  Tcore(sc_core::sc_module_name name, int x, int y, int tcore_id,
        const std::string &csv_path = "../ISA/Tcore0.csv")
      : sc_core::sc_module(name), clk("clk"), noc_cmd_in("noc_cmd_in"),
        noc_cmd_out("noc_cmd_out"), noc_data_in("noc_data_in"),
        noc_data_out("noc_data_out"), all_done("all_done") {

    // ---------- 建立 buffers + unified shadow ----------
    for (std::size_t i = 0; i < BUF_SIZES.size(); ++i) {
      std::string nm = std::string("BUF") + std::to_string(i);
      bufs[i] = std::make_unique<sc_core::sc_vector<sc_core::sc_signal<T>>>(
          nm.c_str(), BUF_SIZES[i]);
    }

    shadow_buf.resize(BUF_SIZES.size());
    for (std::size_t i = 0; i < BUF_SIZES.size(); ++i) {
      shadow_buf[i].assign(BUF_SIZES[i], T(0));
    }

    // ====== [STATE BIN] init + load ======
    init_state_path_();
    load_or_init_state_bin_();             // 讀進 shadow_buf + state_token_pos_
    sig_token_pos.write(state_token_pos_); // token_pos 由 Tcore state 驅動

    // ---------- Router ----------
    tcore_x.write(sc_dt::sc_uint<5>(x));
    tcore_y.write(sc_dt::sc_uint<5>(y));

    router = new Router<T, MODE>("Router");
    router->clk(clk);

    for (std::size_t i = 0; i < BUF_SIZES.size(); ++i) {
      router->bufs[i] = bufs[i].get();         // Router read
      router->shadow_bufs[i] = &shadow_buf[i]; // Router write (unified shadow)
    }

    router->Router_cmd_in(noc_cmd_in);
    router->Router_cmd_out(noc_cmd_out);
    router->Router_data_in(noc_data_in);
    router->Router_data_out(noc_data_out);

    router->start(sig_Router_start);
    router->is_send(sig_Router_is_send);
    router->src_buf(sig_Router_src_buf);
    router->dst_buf(sig_Router_dst_buf);
    router->src_offset(sig_Router_src_offset); // ✅ idx -> offset
    router->dst_offset(sig_Router_dst_offset); // ✅ idx -> offset
    router->length(sig_Router_length);
    router->src_x(tcore_x);
    router->src_y(tcore_y);
    router->dst_x(sig_Router_dst_x);
    router->dst_y(sig_Router_dst_y);

    router->is_mcast(sig_Router_is_mcast);
    router->mcast_num(sig_Router_mcast_num);
    for (int i = 0; i < MAX_MCAST_DESTS; ++i) {
      router->mcast_dst_x[i](sig_Router_mcast_dst_x[i]);
      router->mcast_dst_y[i](sig_Router_mcast_dst_y[i]);
      router->mcast_dst_buf[i](sig_Router_mcast_dst_buf[i]);
      router->mcast_dst_offset[i](
          sig_Router_mcast_dst_offset[i]); // ✅ idx -> offset
    }

    router->Router_send_busy(sig_Router_send_busy);
    router->Router_send_done(sig_Router_send_done);
    router->Router_recv_busy(sig_Router_recv_busy);
    router->Router_recv_done(sig_Router_recv_done);

    router->recv_dst_buf(sig_Router_recv_dst_buf);
    router->recv_dst_buf_valid(sig_Router_recv_dst_valid);
    router->recv_buf_grant(sig_Router_recv_buf_grant);
    router->hash(sig_Router_hash);
    // ---------- Array (NEW ports) ----------
    array = new Array<T, MODE>("Array");
    array->clk(clk);

    for (std::size_t k = 0; k < static_cast<std::size_t>(BUF_ARRAY_IN_CAP);
         ++k) {
      array->Array_input_buffer[k]((*bufs[BUF_ARRAY_IN])[k]);
    }

    for (std::size_t k = 0; k < static_cast<std::size_t>(BUF_ARRAY_OUT_CAP);
         ++k) {
      array->Array_output_buffer[k]((*bufs[BUF_ARRAY_OUT])[k]);
    }

    for (std::size_t k = 0; k < static_cast<std::size_t>(BUF_K_CAP); ++k) {
      array->Array_K_buffer[k]((*bufs[BUF_K])[k]);
    }

    for (std::size_t k = 0; k < static_cast<std::size_t>(BUF_V_CAP); ++k) {
      array->Array_V_buffer[k]((*bufs[BUF_V])[k]);
    }

    // Array 用 Tcore 的 token_pos（state）
    array->token_pos(sig_token_pos);

    array->Array_start(sig_Array_start);
    array->Array_op(sig_Array_op);
    array->Array_key(sig_Array_key);
    array->Array_row(sig_Array_row);
    array->Array_col(sig_Array_col);
    array->Array_busy(sig_Array_busy);
    array->Array_done(sig_Array_done);

    // ---------- Vector ----------
    vector_unit = new Vector<T, 0, MODE>("Vector");
    vector_unit->clk(clk);
    vector_unit->Vector_step(sig_vector_step);
    sig_vector_step.write(0);

    for (std::size_t j = 0; j < static_cast<std::size_t>(VECTOR_IN_CAP); ++j) {
      vector_unit->input_buffer[j]((*bufs[BUF_VEC_IN])[j]);
    }
    for (std::size_t j = 0; j < static_cast<std::size_t>(VECTOR_W_CAP); ++j) {
      vector_unit->weight_buffer[j]((*bufs[BUF_VEC_W])[j]);
    }
    for (std::size_t j = 0; j < static_cast<std::size_t>(VECTOR_OUT_CAP); ++j) {
      vector_unit->output_buffer[j]((*bufs[BUF_VEC_OUT])[j]);
    }

    vector_unit->vector_start(sig_vector_start);
    vector_unit->vector_mode(sig_vector_mode);
    vector_unit->DIM(sig_vector_DIM);
    vector_unit->Vector_busy(sig_vector_busy);
    vector_unit->Vector_done(sig_vector_done);

    // ---------- DMA (3 ch) ----------
    std::vector<sc_core::sc_vector<sc_core::sc_signal<T>> *> dma_bufs;
    std::vector<std::vector<T> *> dma_shadow_ptrs;
    dma_bufs.reserve(BUF_SIZES.size());
    dma_shadow_ptrs.reserve(BUF_SIZES.size());
    for (std::size_t i = 0; i < BUF_SIZES.size(); ++i) {
      dma_bufs.push_back(bufs[i].get());
      dma_shadow_ptrs.push_back(&shadow_buf[i]); // unified shadow
    }

    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      std::ostringstream nm;
      nm << "dma" << ch;
      dma[ch] = new DMA_Controller<T, 8, 8, 5, MODE>(nm.str().c_str(), dma_bufs,
                                                     dma_shadow_ptrs);
      dma[ch]->clk(clk);

      dma[ch]->dma_start(sig_dma_start[ch]);
      dma[ch]->dma_busy(sig_dma_busy[ch]);
      dma[ch]->dma_done(sig_dma_done[ch]);

      // NEW: DRAM busy ports
      dma[ch]->dram_rd_busy(sig_dma_dram_rd_busy[ch]);
      dma[ch]->dram_wr_busy(sig_dma_dram_wr_busy[ch]);

      dma[ch]->src_sel(sig_dma_src_sel[ch]);
      dma[ch]->dst_sel(sig_dma_dst_sel[ch]);
      dma[ch]->src_offset(sig_dma_src_offset[ch]);
      dma[ch]->dst_offset(sig_dma_dst_offset[ch]);
      dma[ch]->length(sig_dma_length[ch]);

      dma[ch]->kv_head(sig_dma_kv_head[ch]);
      dma[ch]->kv_is_prev(sig_dma_kv_is_prev[ch]);

      // DMA 的 token_pos 直接看 Tcore state token_pos（所有 channel 同一個）
      dma[ch]->token_pos(sig_token_pos);

      dma[ch]->kv_layer(sig_dma_kv_layer[ch]);
      dma[ch]->dma_model_key(sig_dma_model_key[ch]);
    }

    // ---------- Core_Controller ----------
    controller =
        new Core_Controller<MODE>("Core_Controller", csv_path, tcore_id);
    controller->clk(clk);

    // Controller <-> Vector
    controller->vector_start(sig_vector_start);
    controller->vector_mode(sig_vector_mode);
    controller->vector_DIM(sig_vector_DIM);
    controller->Vector_busy(sig_vector_busy);
    controller->Vector_done(sig_vector_done);

    // Controller <-> Array (NEW)
    controller->Array_start(sig_Array_start);
    controller->Array_op(sig_Array_op);
    controller->Array_key(sig_Array_key);
    controller->Array_row(sig_Array_row);
    controller->Array_col(sig_Array_col);
    controller->Array_busy(sig_Array_busy);
    controller->Array_done(sig_Array_done);

    // ★ controller 的 token_pos 改綁 dummy（token_pos 改由 Tcore state 驅動）
    controller->token_pos(sig_token_pos);

    // Controller <-> DMA (NEW)
    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      controller->dma_start[ch](sig_dma_start[ch]);
      controller->dma_busy[ch](sig_dma_busy[ch]);
      controller->dma_done[ch](sig_dma_done[ch]);

      controller->dma_src_sel[ch](sig_dma_src_sel[ch]);
      controller->dma_src_offset[ch](sig_dma_src_offset[ch]);
      controller->dma_dst_sel[ch](sig_dma_dst_sel[ch]);
      controller->dma_dst_offset[ch](sig_dma_dst_offset[ch]);
      controller->dma_length[ch](sig_dma_length[ch]);

      controller->dma_kv_head[ch](sig_dma_kv_head[ch]);
      controller->dma_kv_is_prev[ch](sig_dma_kv_is_prev[ch]);

      controller->dma_kv_layer[ch](sig_dma_kv_layer[ch]);
      controller->dma_model_key[ch](sig_dma_model_key[ch]);
    }

    // Controller <-> Router
    controller->Router_start(sig_Router_start);
    controller->Router_is_send(sig_Router_is_send);
    controller->Router_src_buf(sig_Router_src_buf);
    controller->Router_dst_buf(sig_Router_dst_buf);
    controller->Router_src_offset(sig_Router_src_offset);
    controller->Router_dst_offset(sig_Router_dst_offset);
    controller->Router_length(sig_Router_length);
    controller->Router_dst_x(sig_Router_dst_x);
    controller->Router_dst_y(sig_Router_dst_y);

    controller->Router_is_mcast(sig_Router_is_mcast);
    controller->Router_mcast_num(sig_Router_mcast_num);
    for (int i = 0; i < MAX_MCAST_DESTS; ++i) {
      controller->Router_mcast_dst_x[i](sig_Router_mcast_dst_x[i]);
      controller->Router_mcast_dst_y[i](sig_Router_mcast_dst_y[i]);
      controller->Router_mcast_dst_buf[i](sig_Router_mcast_dst_buf[i]);
      controller->Router_mcast_dst_offset[i](
          sig_Router_mcast_dst_offset[i]); // ✅ idx -> offset
    }

    controller->Router_send_busy(sig_Router_send_busy);
    controller->Router_send_done(sig_Router_send_done);
    controller->Router_recv_busy(sig_Router_recv_busy);
    controller->Router_recv_done(sig_Router_recv_done);

    controller->Router_recv_dst_buf(sig_Router_recv_dst_buf);
    controller->Router_recv_dst_valid(sig_Router_recv_dst_valid);
    controller->Router_recv_buf_grant(sig_Router_recv_buf_grant);
    controller->Router_hash(sig_Router_hash);
    controller->all_done(all_done);

    // ---------- Methods ----------
    SC_METHOD(relay_monitors);
    sensitive << sig_vector_busy << sig_Array_busy << sig_Router_send_busy
              << sig_Router_recv_busy;
    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      sensitive << sig_dma_busy[ch];
      sensitive << sig_dma_dram_rd_busy[ch];
      sensitive << sig_dma_dram_wr_busy[ch];
    }
    dont_initialize();

    // commit shadow -> real buffers
    // ★ 這個 method 會在 time 0 先做一次 state init commit（同一個 writer
    // process，避免 multi-writer）
    SC_METHOD(update_real_bufs);
    sensitive << sig_Router_recv_done;
    for (int ch = 0; ch < NUM_DMA_CH; ++ch)
      sensitive << sig_dma_done[ch];
    // ★不要 dont_initialize()

    // ★ all_done 且 idle 時，把 token_pos+1 + 12 bufs 覆寫回 state.bin
    SC_METHOD(autosave_state_when_done_);
    sensitive << all_done << sig_vector_busy << sig_Array_busy
              << sig_Router_send_busy << sig_Router_recv_busy
              << sig_Router_recv_done;
    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      sensitive << sig_dma_busy[ch];
      sensitive << sig_dma_done[ch];
    }
    dont_initialize();
  }

private:
  // =======================
  // [STATE BIN] members
  // =======================
  std::string state_bin_path_;
  bool state_loaded_{false};
  bool state_init_committed_{false};
  bool state_saved_{false};
  std::int32_t state_token_pos_{0};

  // 解析 "top.Tcore0" / "Tcore0" → 0
  static int parse_tcore_id_from_name_(const std::string &hier_name) {
    std::string leaf = hier_name;
    auto p = leaf.find_last_of('.');
    if (p != std::string::npos)
      leaf = leaf.substr(p + 1);

    // 優先找 "Tcore" 後面的數字
    std::size_t tpos = leaf.find("Tcore");
    if (tpos != std::string::npos) {
      std::size_t i = tpos + 5;
      std::size_t j = i;
      while (j < leaf.size() &&
             std::isdigit(static_cast<unsigned char>(leaf[j])))
        ++j;
      if (j > i)
        return std::stoi(leaf.substr(i, j - i));
    }

    // 否則抓尾巴數字
    std::size_t end = leaf.size();
    std::size_t start = end;
    while (start > 0 &&
           std::isdigit(static_cast<unsigned char>(leaf[start - 1])))
      --start;
    if (start < end)
      return std::stoi(leaf.substr(start, end - start));
    return 0;
  }

  using StateStorageT = float;

  static std::size_t total_state_elems_() {
    std::size_t total = 0;
    for (int n : BUF_SIZES) {
      total += static_cast<std::size_t>(n);
    }
    return total;
  }

  static std::size_t expected_state_bytes_() {
    return sizeof(std::int32_t) + total_state_elems_() * sizeof(StateStorageT);
  }

  void init_state_path_() {
    const int id = parse_tcore_id_from_name_(this->name());
    const char *state_dir_env = std::getenv("ASMAP_STATE_DIR");
    std::string state_dir =
        (state_dir_env && state_dir_env[0] != '\0')
            ? std::string(state_dir_env)
            : std::string("../DRAM_json");

    if (!state_dir.empty() && state_dir.back() == '/') {
      state_dir.pop_back();
    }

    state_bin_path_ =
        state_dir + "/Tcore" + std::to_string(id) + "_state.bin";
  }

  void load_or_init_state_bin_() {
    namespace fs = std::filesystem;
    fs::path p(state_bin_path_);
    fs::create_directories(p.parent_path());

    const std::size_t expect = expected_state_bytes_();

    bool ok = false;
    if (fs::exists(p)) {
      std::error_code ec;
      auto sz = fs::file_size(p, ec);
      if (!ec && sz == expect)
        ok = true;
    }

    if (!ok) {
      // 建新檔（token_pos=0，12 buf 全 0）
      state_token_pos_ = 1;
      for (std::size_t i = 0; i < shadow_buf.size(); ++i) {
        std::fill(shadow_buf[i].begin(), shadow_buf[i].end(), T(0));
      }
      write_state_bin_(/*token_pos_to_write*/ state_token_pos_,
                       /*read_from_real*/ false);
      state_loaded_ = true;

      std::ostringstream msg;
      msg << "[STATE] init new state bin: " << state_bin_path_
          << " (bytes=" << expect << ")";
      SC_REPORT_INFO(this->name(), msg.str().c_str());
      return;
    }

    // 讀檔 → shadow_buf + state_token_pos_
    std::ifstream ifs(state_bin_path_, std::ios::binary);
    if (!ifs) {
      SC_REPORT_WARNING(this->name(),
                        "[STATE] open state bin failed, fallback to zeros");
      state_token_pos_ = 0;
      state_loaded_ = true;
      return;
    }

    std::int32_t tp = 0;
    ifs.read(reinterpret_cast<char *>(&tp), sizeof(tp));
    if (!ifs)
      tp = 1;
    if (tp < 1) {
      SC_REPORT_WARNING(this->name(),
                        "[STATE] token_pos<1 in state.bin, clamp to 1");
      tp = 1;
    }
    state_token_pos_ = tp;

    std::vector<StateStorageT> flat(total_state_elems_(), StateStorageT(0));

    ifs.read(reinterpret_cast<char *>(flat.data()),
             static_cast<std::streamsize>(flat.size() * sizeof(StateStorageT)));

    if (!ifs) {
      SC_REPORT_WARNING(
          this->name(),
          "[STATE] state payload read incomplete, missing values use 0");
    }

    std::size_t idx = 0;
    for (std::size_t b = 0; b < shadow_buf.size(); ++b) {
      for (int i = 0; i < BUF_SIZES[b]; ++i) {
        shadow_buf[b][static_cast<std::size_t>(i)] =
            static_cast<T>(flat[idx++]);
      }
    }

    state_loaded_ = true;

    std::ostringstream msg;
    msg << "[STATE] loaded: " << state_bin_path_
        << " token_pos=" << state_token_pos_;
    SC_REPORT_INFO(this->name(), msg.str().c_str());
  }

  // 寫檔：token_pos(int32) + 12*buf(float32)
  // read_from_real=true：從 real sc_signal 讀值存（用於結束時存檔）
  // read_from_real=false：從 shadow_buf 存（用於初始化建檔/修復檔）
  void write_state_bin_(std::int32_t token_pos_to_write, bool read_from_real) {
    namespace fs = std::filesystem;
    fs::path p(state_bin_path_);
    fs::create_directories(p.parent_path());

    fs::path tmp = p;
    tmp += ".tmp";

    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs) {
      SC_REPORT_ERROR(this->name(), "[STATE] write_state_bin_ open tmp failed");
      return;
    }

    ofs.write(reinterpret_cast<const char *>(&token_pos_to_write),
              sizeof(token_pos_to_write));

    std::vector<StateStorageT> flat(total_state_elems_(), StateStorageT(0));

    std::size_t idx = 0;
    for (int b = 0; b < NUM_BUFS; ++b) {
      for (int i = 0; i < BUF_SIZES[b]; ++i) {
        if (read_from_real) {
          // 從 real sc_signal 讀（包含 BUF_ARRAY_OUT/BUF_VEC_OUT）
          flat[idx++] = static_cast<StateStorageT>(
              (*bufs[static_cast<std::size_t>(b)])[static_cast<std::size_t>(i)]
                  .read());
        } else {
          flat[idx++] = static_cast<StateStorageT>(
              shadow_buf[static_cast<std::size_t>(b)]
                        [static_cast<std::size_t>(i)]);
        }
      }
    }

    ofs.write(
        reinterpret_cast<const char *>(flat.data()),
        static_cast<std::streamsize>(flat.size() * sizeof(StateStorageT)));

    if (!ofs) {
      SC_REPORT_ERROR(this->name(), "[STATE] write_state_bin_ payload failed");
      return;
    }
    ofs.flush();
    ofs.close();

    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) {
      // windows/某些情況 rename 會失敗：先 remove 再 rename
      fs::remove(p, ec);
      ec.clear();
      fs::rename(tmp, p, ec);
    }
    if (ec) {
      SC_REPORT_ERROR(this->name(), "[STATE] rename tmp -> state bin failed");
      return;
    }
  }

  // =======================
  // existing methods
  // =======================
  void relay_monitors() {
    mon_vector_busy.write(sig_vector_busy.read());
    mon_array_busy.write(sig_Array_busy.read());

    bool dma_or = false;
    bool dram_rd_or = false;
    bool dram_wr_or = false;

    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      dma_or = dma_or || sig_dma_busy[ch].read();
      dram_rd_or = dram_rd_or || sig_dma_dram_rd_busy[ch].read();
      dram_wr_or = dram_wr_or || sig_dma_dram_wr_busy[ch].read();
    }

    mon_dma_busy.write(dma_or);

    // ✅ DRAM RD busy = (OR of 3 DMA channels) OR (Array busy)
    mon_dram_rd_busy.write(dram_rd_or || sig_Array_busy.read());

    // (保持原本定義：只看 DMA channels 的 wr busy)
    mon_dram_wr_busy.write(dram_wr_or);

    mon_Router_send_busy.write(sig_Router_send_busy.read());
    mon_Router_recv_busy.write(sig_Router_recv_busy.read());
  }
  // all_done 且完全 idle → 存 state.bin（token_pos+1 + 12 bufs）
  void autosave_state_when_done_() {
    if (state_saved_)
      return;
    if (!all_done.read())
      return;

    bool idle = true;
    idle = idle && !sig_vector_busy.read();
    idle = idle && !sig_Array_busy.read();
    idle = idle && !sig_Router_send_busy.read();
    idle = idle && !sig_Router_recv_busy.read();
    for (int ch = 0; ch < NUM_DMA_CH; ++ch)
      idle = idle && !sig_dma_busy[ch].read();

    if (!idle)
      return;

    const std::int32_t next_tp =
        static_cast<std::int32_t>(sig_token_pos.read()) + 1;

    // 這裡用 read_from_real=true：把「當下 sc_signal 的 buf」存回去（含 output
    // bufs）
    write_state_bin_(next_tp, /*read_from_real*/ true);
    state_saved_ = true;

    std::ostringstream msg;
    msg << "[STATE] saved: " << state_bin_path_
        << " token_pos(next)=" << next_tp << " | t=" << sc_time_stamp();
    SC_REPORT_INFO(this->name(), msg.str().c_str());
  }

  // commit shadow -> real buffers
  void update_real_bufs() {
    // ★ time 0：如果有 loaded state，先做一次「全 buf commit（避開 output
    // bufs）」
    if (state_loaded_ && !state_init_committed_) {
      auto is_forbidden_buf = [&](int b) {
        return (b == BUF_ARRAY_OUT || b == BUF_VEC_OUT);
      };
      for (int b = 0; b < NUM_BUFS; ++b) {
        if (is_forbidden_buf(b))
          continue;
        auto &dst_vec = *bufs[static_cast<std::size_t>(b)];
        auto &src_vec = shadow_buf[static_cast<std::size_t>(b)];
        const int N = static_cast<int>(dst_vec.size());
        for (int i = 0; i < N; ++i)
          dst_vec[i].write(src_vec[i]);
      }
      state_init_committed_ = true;
      return;
    }

    const bool router_finished = sig_Router_recv_done.read();

    bool dma_finished[NUM_DMA_CH]{};
    bool any_dma_finished = false;
    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      dma_finished[ch] = sig_dma_done[ch].read();
      any_dma_finished = any_dma_finished || dma_finished[ch];
    }
    if (!router_finished && !any_dma_finished)
      return;

    auto is_forbidden_buf = [&](int b) {
      return (b == BUF_ARRAY_OUT || b == BUF_VEC_OUT);
    };

    auto commit_range = [&](int buf_id, int off, int len) {
      if (buf_id < 0 || buf_id >= NUM_BUFS)
        return;
      if (is_forbidden_buf(buf_id))
        return;

      auto &dst_vec = *bufs[static_cast<std::size_t>(buf_id)];
      auto &src_vec = shadow_buf[static_cast<std::size_t>(buf_id)];

      const int N = static_cast<int>(dst_vec.size());
      if (len <= 0)
        return;
      if (off < 0 || off + len > N) {
        SC_REPORT_ERROR(this->name(),
                        "update_real_bufs: commit_range out of bound");
        return;
      }
      for (int i = 0; i < len; ++i) {
        dst_vec[off + i].write(src_vec[off + i]);
      }
    };

    auto commit_full = [&](int buf_id) {
      if (buf_id < 0 || buf_id >= NUM_BUFS)
        return;
      if (is_forbidden_buf(buf_id))
        return;

      auto &dst_vec = *bufs[static_cast<std::size_t>(buf_id)];
      auto &src_vec = shadow_buf[static_cast<std::size_t>(buf_id)];
      const int N = static_cast<int>(dst_vec.size());
      for (int i = 0; i < N; ++i)
        dst_vec[i].write(src_vec[i]);
    };

    // DMA commit (range)
    for (int ch = 0; ch < NUM_DMA_CH; ++ch) {
      if (!dma_finished[ch])
        continue;

      const int dst_buf =
          static_cast<int>(sig_dma_dst_sel[ch].read().to_uint());
      const int off = sig_dma_dst_offset[ch].read();
      const int len = sig_dma_length[ch].read();

      std::ostringstream oss;
      oss << "[DMA" << ch << " COMMIT] dst_buf=" << dst_buf << " off=" << off
          << " len=" << len << " | t=" << sc_time_stamp();
      SC_REPORT_INFO(this->name(), oss.str().c_str());

      commit_range(dst_buf, off, len);
    }

    // Router RECV commit
    if (router_finished) {
      const int dst_buf =
          static_cast<int>(sig_Router_recv_dst_buf.read().to_uint());

      std::ostringstream oss;
      oss << "[Router RECV COMMIT] dst_buf=" << dst_buf
          << " (full commit) | t=" << sc_time_stamp();
      SC_REPORT_INFO(this->name(), oss.str().c_str());

      commit_full(dst_buf);
    }
  }
};

#endif // TCORE_H
