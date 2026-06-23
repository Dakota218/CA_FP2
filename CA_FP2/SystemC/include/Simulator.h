#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "NoC.h"
#include "Tcore.h"
#include "hw_config_selector.h"
#include "mem_types.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <systemc.h>
template <typename T, int NX = 2, int NY = 2,
          int LANES =
              (ActiveHWConfig::SRAM_DMA_WIDTH < ActiveHWConfig::NOC_LINK_WIDTH
                   ? ActiveHWConfig::SRAM_DMA_WIDTH
                   : ActiveHWConfig::NOC_LINK_WIDTH),
          int MODE = 0>
class Simulator : public sc_core::sc_module {
public:
  static constexpr int NUM_NODES = NX * NY;
  static constexpr int NCORES = NUM_NODES;
  using HW = ActiveHWConfig;

  static constexpr int max_noc_packet_len_() {
    int mx = 1;
    for (int v : HW::BUF_SIZES) {
      if (v > mx)
        mx = v;
    }
    return mx;
  }

  static constexpr int MAX_NOC_PACKET_LEN = max_noc_packet_len_();

  // =====================================================================
  // FIFO 深度參數（NoC 端有 full-packet buffering 檢查）
  // ★注意：NoC.h 會檢查 fifo_depth >= cmd.length，否則直接 SC_REPORT_ERROR +
  // sc_stop() 所以 FIFO_NOC_DATA_*_DEPTH 必須 >= 你最大可能送的 length
  // =====================================================================
  static constexpr int FIFO_NOC_CMD_TO_NOC_DEPTH = 128;
  static constexpr int FIFO_NOC_CMD_FROM_NOC_DEPTH = 128;

  // data fifo depth：建議 >= 2048（你目前常用 token=2048）
  static constexpr int FIFO_NOC_DATA_TO_NOC_DEPTH = 2 * MAX_NOC_PACKET_LEN;
  static constexpr int FIFO_NOC_DATA_FROM_NOC_DEPTH = 2 * MAX_NOC_PACKET_LEN;

  // ===== 對外 Ports =====
  sc_core::sc_in<bool> clk;       // 外部 clock
  sc_core::sc_out<bool> all_done; // 所有 Tcore 都完成 → true

  // 每顆 core done
  sc_core::sc_vector<sc_core::sc_signal<bool>> tcore_done_sig;

  // ===== 子模組 =====
  NoC<T, LANES, NX, NY> *noc{nullptr};
  Tcore<T, MODE> *tcore[NUM_NODES]{}; // ★init nullptr

  // ===== NoC FIFO（Tcore ↔ NoC）=====
  sc_core::sc_fifo<NocCmd<T>> *noc_cmd_to_noc[NUM_NODES]{};
  sc_core::sc_fifo<T> *noc_data_to_noc[NUM_NODES]{};

  sc_core::sc_fifo<NocCmd<T>> *noc_cmd_from_noc[NUM_NODES]{};
  sc_core::sc_fifo<T> *noc_data_from_noc[NUM_NODES]{};

  // ===== 監視用 signals（每顆 Tcore 的 busy / done）=====
  sc_core::sc_signal<bool> tcore_vector_busy_sig[NCORES]{};
  sc_core::sc_signal<bool> tcore_array_busy_sig[NCORES]{};
  sc_core::sc_signal<bool> tcore_dma_busy_sig[NCORES]{};
  sc_core::sc_signal<bool> tcore_Router_send_busy_sig[NCORES]{};
  sc_core::sc_signal<bool> tcore_Router_recv_busy_sig[NCORES]{};

  // ★你 Tcore.h 新增的 DRAM busy monitor（OR of 3 DMA channels）
  sc_core::sc_signal<bool> tcore_dram_rd_busy_sig[NCORES]{};
  sc_core::sc_signal<bool> tcore_dram_wr_busy_sig[NCORES]{};

  // NoC busy（NoC.h 裡有 sc_out<bool> noc_busy）
  sc_core::sc_signal<bool> noc_busy_sig;

  SC_HAS_PROCESS(Simulator);

  Simulator(sc_core::sc_module_name name)
      : sc_core::sc_module(name), clk("clk"), all_done("all_done"),
        tcore_done_sig("tcore_done_sig", NCORES) {

    // =====================================================================
    // 1) 建立 NoC ↔ Tcore 的 FIFO（用指定深度）
    // =====================================================================
    for (int id = 0; id < NUM_NODES; ++id) {
      {
        std::ostringstream nm;
        nm << "noc_cmd_to_noc_" << id;
        noc_cmd_to_noc[id] = new sc_core::sc_fifo<NocCmd<T>>(
            nm.str().c_str(), FIFO_NOC_CMD_TO_NOC_DEPTH);
      }
      {
        std::ostringstream nm;
        nm << "noc_data_to_noc_" << id;
        noc_data_to_noc[id] = new sc_core::sc_fifo<T>(
            nm.str().c_str(), FIFO_NOC_DATA_TO_NOC_DEPTH);
      }
      {
        std::ostringstream nm;
        nm << "noc_cmd_from_noc_" << id;
        noc_cmd_from_noc[id] = new sc_core::sc_fifo<NocCmd<T>>(
            nm.str().c_str(), FIFO_NOC_CMD_FROM_NOC_DEPTH);
      }
      {
        std::ostringstream nm;
        nm << "noc_data_from_noc_" << id;
        noc_data_from_noc[id] = new sc_core::sc_fifo<T>(
            nm.str().c_str(), FIFO_NOC_DATA_FROM_NOC_DEPTH);
      }
    }

    // =====================================================================
    // 2) 建立 NoC
    // =====================================================================
    noc = new NoC<T, LANES, NX, NY>("NoC");
    noc->clk(clk);
    noc->noc_busy(noc_busy_sig);

    // =====================================================================
    // 3) 建立 NX × NY 顆 Tcore，並把「相同 id」的 FIFO 綁在一起
    //    node_id = y * NX + x
    // =====================================================================
    for (int y = 0; y < NY; ++y) {
      for (int x = 0; x < NX; ++x) {
        const int id = y * NX + x;

        // ISA path
        std::string isa_path;
        if constexpr (MODE == 0) {
          isa_path = "/home/wilson/SystemC/Compiler_ISA/core_" +
                     std::to_string(id) + "_16_layer_simulator_code.csv";
        } else if constexpr (MODE == 1) {
          isa_path =
              "/home/wilson/SystemC/Compiler_ISA/1x1_mlp/simulator_code/core_" +
              std::to_string(id) + "_1_layer_simulator_code.csv";
        } else if constexpr (MODE == 2) {
          isa_path =
              "/home/wilson/SystemC/Compiler_ISA/2x2_mlp/simulator_code/core_" +
              std::to_string(id) + "_1_layer_simulator_code.csv";
        } else if constexpr (MODE == 3) {
          isa_path =
              "/home/wilson/SystemC/Compiler_ISA/4x4_mlp/simulator_code/core_" +
              std::to_string(id) + "_1_layer_simulator_code.csv";
        } else if constexpr (MODE == 4) {
          isa_path =
              "/home/wilson/SystemC/Compiler_ISA/8x8_mlp/simulator_code/core_" +
              std::to_string(id) + "_1_layer_simulator_code.csv";
        } else if constexpr (MODE == 5) {
          isa_path = fp1_core_isa_path_(id);
        } else if constexpr (MODE == 6 || MODE == 7) {
          isa_path = asmap_core_isa_path_(id);
        } else {
          SC_REPORT_ERROR("Simulator", "Unsupported MODE");
          sc_core::sc_stop();
          return;
        }
        std::string tcore_name = "Tcore_" + std::to_string(id);

        tcore[id] = new Tcore<T, MODE>(tcore_name.c_str(), x, y, id, isa_path);
        tcore[id]->clk(clk);

        // ---------------------------
        // Tcore ↔ NoC 的 FIFO 連線
        // ---------------------------
        // cmd: Tcore.out -> NoC.in[id]
        tcore[id]->noc_cmd_out(*noc_cmd_to_noc[id]);
        noc->noc_cmd_in[id](*noc_cmd_to_noc[id]);

        // cmd: NoC.out[id] -> Tcore.in
        noc->noc_cmd_out[id](*noc_cmd_from_noc[id]);
        tcore[id]->noc_cmd_in(*noc_cmd_from_noc[id]);

        // data: Tcore.out -> NoC.in[id]
        tcore[id]->noc_data_out(*noc_data_to_noc[id]);
        noc->noc_data_in[id](*noc_data_to_noc[id]);

        // data: NoC.out[id] -> Tcore.in
        noc->noc_data_out[id](*noc_data_from_noc[id]);
        tcore[id]->noc_data_in(*noc_data_from_noc[id]);

        // ---------------------------
        // done + busy monitors
        // ---------------------------
        tcore[id]->all_done(tcore_done_sig[id]);

        tcore[id]->mon_vector_busy(tcore_vector_busy_sig[id]);
        tcore[id]->mon_array_busy(tcore_array_busy_sig[id]);
        tcore[id]->mon_dma_busy(tcore_dma_busy_sig[id]);

        tcore[id]->mon_Router_send_busy(tcore_Router_send_busy_sig[id]);
        tcore[id]->mon_Router_recv_busy(tcore_Router_recv_busy_sig[id]);

        // ★Tcore.h 新增的 DRAM busy monitor
        tcore[id]->mon_dram_rd_busy(tcore_dram_rd_busy_sig[id]);
        tcore[id]->mon_dram_wr_busy(tcore_dram_wr_busy_sig[id]);
      }
    }

    // =====================================================================
    // 4) all_done = AND(tcore_done_sig[])
    // =====================================================================
    SC_METHOD(combine_all_done);
    for (int i = 0; i < NCORES; ++i) {
      sensitive << tcore_done_sig[i];
    }
    dont_initialize();
  }

  ~Simulator() override {
    for (int i = 0; i < NUM_NODES; ++i) {
      if (tcore[i])
        delete tcore[i];

      if (noc_cmd_to_noc[i])
        delete noc_cmd_to_noc[i];
      if (noc_data_to_noc[i])
        delete noc_data_to_noc[i];
      if (noc_cmd_from_noc[i])
        delete noc_cmd_from_noc[i];
      if (noc_data_from_noc[i])
        delete noc_data_from_noc[i];
    }
    if (noc)
      delete noc;
  }

private:
#ifndef FP1_SUBMISSION_ROOT
#define FP1_SUBMISSION_ROOT "/home/wilson/2026_Spring_CA/FP1/submission"
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

  static std::string fp1_core_isa_path_(int id) {
    const std::string g = fp1_group_suffix_();
    return fp1_group_dir_() + "/core_" + std::to_string(id) + "_" + g + ".csv";
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

  static std::string getenv_or_default_(const char *name,
                                        const std::string &fallback) {
    const char *v = std::getenv(name);
    if (v && v[0] != '\0') {
      return std::string(v);
    }
    return fallback;
  }

  static std::string asmap_default_run_dir_() {
    return "/home/wilson/SystemC/auto_sched_map/runs/llama_ffn_plan000_v0";
  }

  static std::string asmap_instr_dir_() {
    return getenv_or_default_("ASMAP_INSTR_DIR",
                              join_path_(asmap_default_run_dir_(), "instr"));
  }

  static std::string fp2_group_suffix_from_path_(const std::string &path) {
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

  static std::string asmap_core_isa_path_(int id) {
    if constexpr (MODE == 7) {
      const std::string group = fp2_group_suffix_from_path_(asmap_instr_dir_());
      return join_path_(asmap_instr_dir_(),
                        "core_" + std::to_string(id) + "_" + group + ".csv");
    } else {
      return join_path_(asmap_instr_dir_(),
                        "core_" + std::to_string(id) + ".csv");
    }
  }
  void combine_all_done() {
    bool ok = true;
    for (int i = 0; i < NCORES; ++i) {
      if (!tcore_done_sig[i].read()) {
        ok = false;
        break;
      }
    }
    all_done.write(ok);
  }
};

#endif // SIMULATOR_H