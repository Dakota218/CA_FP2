// /home/wilson/SystemC/test_tool/main.cpp
#include "Simulator.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <cctype>
#include <sysc/utils/sc_report_handler.h>
#include <systemc.h>
/*
./build/main 2>&1 | tee ./logs/sim.log
./build/main --no-vcd 2>&1 | tee ./logs/sim.log
./build/main --vcd mywave 2>&1 | tee ./logs/sim.log
*/

static constexpr int MODE = 7; // 0,1,2,3,4,5,6
// MODE = 5, FP1 mode (2x2 Tcore)
// MODE = 6, auto_sched_map 的 LLaMA FFN 模式（目前是 2×2）
// MODE = 7, FP2 mode (8x8 Tcore)
using T = float;
// ===== logging control =====
static constexpr bool ENABLE_SC_INFO =
    true; // false: 關掉所有模組的 SC_REPORT_INFO
static constexpr bool ENABLE_MAIN_INFO = true; // true : 保留 main.cpp 的訊息

static void main_info(const std::string &msg) {
  if constexpr (ENABLE_MAIN_INFO) {
    std::cout << "Info: main: " << msg << std::endl;
  }
}
static std::string join_path(std::string base, const std::string &leaf) {
  if (base.empty())
    return leaf;
  if (base.back() == '/')
    return base + leaf;
  return base + "/" + leaf;
}

static void set_env_path(const char *name, const std::string &value) {
  setenv(name, value.c_str(), 1);
  if constexpr (ENABLE_MAIN_INFO) {
    std::cout << "Info: main: " << name << "=" << value << std::endl;
  }
}
static std::string getenv_or_default(const char *name,
                                     const std::string &fallback) {
  const char *v = std::getenv(name);
  if (v && v[0] != '\0') {
    return std::string(v);
  }
  return fallback;
}
static std::string repo_root_from_cwd() {
  namespace fs = std::filesystem;
  fs::path cwd = fs::current_path();

  if (cwd.filename() == "test_tool" && cwd.has_parent_path() &&
      cwd.parent_path().filename() == "SystemC") {
    return cwd.parent_path().parent_path().generic_string();
  }

  if (cwd.filename() == "SystemC") {
    return cwd.parent_path().generic_string();
  }

  return cwd.generic_string();
}
static std::string fp2_group_suffix_from_path(const std::string &path) {
  for (std::size_t i = path.size(); i >= 4; --i) {
    const std::size_t p = i - 4;
    if (path[p] == '_' && path[p + 1] == 'G' &&
        std::isdigit(static_cast<unsigned char>(path[p + 2])) &&
        std::isdigit(static_cast<unsigned char>(path[p + 3]))) {
      return path.substr(p + 1, 3);
    }
  }
  return "G01";
}
static std::string default_instr_dir_for_mode(const std::string &run_dir) {
  if constexpr (MODE == 1) {
    return "/home/wilson/SystemC/Compiler_ISA/1x1_mlp/simulator_code";
  } else if constexpr (MODE == 2) {
    return "/home/wilson/SystemC/Compiler_ISA/2x2_mlp/simulator_code";
  } else if constexpr (MODE == 3) {
    return "/home/wilson/SystemC/Compiler_ISA/4x4_mlp/simulator_code";
  } else if constexpr (MODE == 4) {
    return "/home/wilson/SystemC/Compiler_ISA/8x8_mlp/simulator_code";
  } else if constexpr (MODE == 7) {
    return run_dir;
  } else {
    return "";
  }
}
static std::string default_dram_dir_for_mode(const std::string &run_dir) {
  if constexpr (MODE >= 0 && MODE <= 4) {
    return "/home/wilson/SystemC/DRAM_json";
  } else if constexpr (MODE == 7) {
    return join_path(run_dir, "weights_" + fp2_group_suffix_from_path(run_dir));
  } else {
    return "";
  }
}

static std::string default_sim_out_dir_for_mode() {
  if constexpr (MODE >= 1 && MODE <= 4) {
    return "/home/wilson/SystemC/MNIST_picture";
  } else {
    return "";
  }
}
static constexpr int NX = (MODE == 0)   ? 2
                          : (MODE == 1) ? 1
                          : (MODE == 2) ? 2
                          : (MODE == 3) ? 4
                          : (MODE == 4) ? 8
                          : (MODE == 5) ? 2
                          : (MODE == 6) ? 4
                          : (MODE == 7) ? 8
                                        : 2;

static constexpr int NY = (MODE == 0)   ? 2
                          : (MODE == 1) ? 1
                          : (MODE == 2) ? 2
                          : (MODE == 3) ? 4
                          : (MODE == 4) ? 8
                          : (MODE == 5) ? 2
                          : (MODE == 6) ? 4
                          : (MODE == 7) ? 8
                                        : 2;

static constexpr int NCORES = NX * NY;

// ========================================================
// BusyMeter：針對「一顆 Tcore + 它自己的 DRAM」做統計
// ========================================================
SC_MODULE(BusyMeter) {
  sc_in<bool> clk;
  sc_in<bool> all_done;

  // ★ 新增：該顆 Tcore 自己的 done
  sc_in<bool> core_done;

  // 來自該 Tcore 的 busy
  sc_in<bool> vbusy;
  sc_in<bool> abusy;
  sc_in<bool> dbusy;
  sc_in<bool> rsbusy;
  sc_in<bool> rrbusy;

  // 來自該 Tcore 對應 DRAM 的 busy
  sc_in<bool> dram_rbusy;
  sc_in<bool> dram_wbusy;

  // 累計（只算到該 core_done）
  unsigned long long total_cycles{0};
  unsigned long long vbusy_cycles{0};
  unsigned long long abusy_cycles{0};
  unsigned long long dbusy_cycles{0};
  unsigned long long rsbusy_cycles{0};
  unsigned long long rrbusy_cycles{0};
  unsigned long long dram_rbusy_cycles{0};
  unsigned long long dram_wbusy_cycles{0};

  // ★ 重新定義這兩個
  unsigned long long tcore_busy_cycles{0};  // V/A/D/Router (4個類別)
  unsigned long long system_busy_cycles{0}; // V/A/D/R/DRAM (5個訊號)

  // 控制：停止計數 vs 已輸出
  bool stop_counting{false};
  bool printed{false};

  // 用 prev 避免「core_done 當拍」的 off-by-one
  bool core_done_prev{false};

  void tick() {
    if (stop_counting)
      return;

    const bool cd = core_done.read();

    // 若「上一拍」就已經 done，這拍開始不再算
    if (core_done_prev) {
      stop_counting = true;
      return;
    }

    // 這拍仍算進總 cycle（包含 done 上升那一拍）
    total_cycles++;

    const bool vb = vbusy.read();
    const bool ab = abusy.read();
    const bool db = dbusy.read();
    const bool rs = rsbusy.read();
    const bool rr = rrbusy.read();
    const bool rb = dram_rbusy.read();
    const bool wb = dram_wbusy.read();

    if (vb)
      ++vbusy_cycles;
    if (ab)
      ++abusy_cycles;
    if (db)
      ++dbusy_cycles;
    if (rs)
      ++rsbusy_cycles;
    if (rr)
      ++rrbusy_cycles;
    if (rb)
      ++dram_rbusy_cycles;
    if (wb)
      ++dram_wbusy_cycles;

    // ★ 你指定的新定義：
    // Tcore busy = V/A/D/Router (Router = send||recv)
    if (vb || ab || db || rs || rr)
      ++tcore_busy_cycles;

    // System busy = V/A/D/Router/DRAM  (五個訊號)
    if (vb || ab || db || rs || rr || rb || wb)
      ++system_busy_cycles;

    core_done_prev = cd;
  }

  static double pct(unsigned long long x, unsigned long long total) {
    if (total == 0)
      return 0.0;
    return 100.0 * static_cast<long double>(x) /
           static_cast<long double>(total);
  }

  void on_all_done() {
    if (printed)
      return;
    printed = true;

    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss << std::setprecision(2);

    oss << "\n===== Busy UTIL Report (per-core total_cycles) =====\n"
        << "Total cycles: " << total_cycles << "\n"
        << "Vector busy:  " << vbusy_cycles << " ("
        << pct(vbusy_cycles, total_cycles) << "%)\n"
        << "Array busy:   " << abusy_cycles << " ("
        << pct(abusy_cycles, total_cycles) << "%)\n"
        << "DMA busy:     " << dbusy_cycles << " ("
        << pct(dbusy_cycles, total_cycles) << "%)\n"
        << "Rtr send:     " << rsbusy_cycles << " ("
        << pct(rsbusy_cycles, total_cycles) << "%)\n"
        << "Rtr recv:     " << rrbusy_cycles << " ("
        << pct(rrbusy_cycles, total_cycles) << "%)\n"
        << "DRAM rd busy: " << dram_rbusy_cycles << " ("
        << pct(dram_rbusy_cycles, total_cycles) << "%)\n"
        << "DRAM wr busy: " << dram_wbusy_cycles << " ("
        << pct(dram_wbusy_cycles, total_cycles) << "%)\n"
        << "----------------------------------------------------\n"
        << "Tcore busy (Vector/Array/DMA/Router): " << tcore_busy_cycles << " ("
        << pct(tcore_busy_cycles, total_cycles)
        << "%)\n"
        // << "System busy (V/A/D/R/DRAM): " << system_busy_cycles << " ("
        // << pct(system_busy_cycles, total_cycles) << "%)\n"
        << "====================================================\n";

    SC_REPORT_INFO(name(), oss.str().c_str());
  }

  SC_CTOR(BusyMeter) {
    SC_METHOD(tick);
    sensitive << clk.pos();
    dont_initialize();

    // ★ 報告仍然在 all_done 出來時統一印
    SC_METHOD(on_all_done);
    sensitive << all_done.pos();
    dont_initialize();
  }
};

// ========================================================
// NoC BusyMeter：只看 NoC 的 busy 佔比
// ========================================================
SC_MODULE(NocBusyMeter) {
  sc_in<bool> clk;
  sc_in<bool> all_done;
  sc_in<bool> noc_busy;

  unsigned long long total_cycles{0};
  unsigned long long busy_cycles{0};
  bool done_seen{false};

  void tick() {
    if (done_seen)
      return;
    total_cycles++;
    if (noc_busy.read())
      ++busy_cycles;
  }

  static double pct(unsigned long long x, unsigned long long total) {
    if (total == 0)
      return 0.0;
    return 100.0 * static_cast<long double>(x) /
           static_cast<long double>(total);
  }

  void on_done() {
    if (done_seen)
      return;
    done_seen = true;

    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss << std::setprecision(2);

    oss << "\n===== NoC Busy Utilization (up to all_done) =====\n"
        << "Total cycles: " << total_cycles << "\n"
        << "NoC busy:     " << busy_cycles << " ("
        << pct(busy_cycles, total_cycles) << "%)\n"
        << "=================================================\n";

    SC_REPORT_INFO(name(), oss.str().c_str());
  }

  SC_CTOR(NocBusyMeter) {
    SC_METHOD(tick);
    sensitive << clk.pos();
    dont_initialize();

    SC_METHOD(on_done);
    sensitive << all_done.pos();
    dont_initialize();
  }
};

// ========================================================
// DoneWatcher：偵測 all_done 正緣後，負責 sc_stop()
// ========================================================
SC_MODULE(DoneWatcher) {
  sc_in<bool> all_done;

  void on_done() {
    if (all_done.read()) {
      std::ostringstream oss;
      oss << "all_done asserted at " << sc_time_stamp();
      SC_REPORT_INFO("DoneWatcher", oss.str().c_str());
      sc_stop();
    }
  }

  SC_CTOR(DoneWatcher) {
    SC_METHOD(on_done);
    sensitive << all_done.pos();
    dont_initialize();
  }
};

// ========================================================
// sc_main
// ========================================================
int sc_main(int argc, char *argv[]) {
  if (!ENABLE_SC_INFO) {
    sc_core::sc_report_handler::set_actions(sc_core::SC_INFO,
                                            sc_core::SC_DO_NOTHING);
  }
  // ----------------------------
  // Args
  //   用法：
  //     ./build/main [--vcd [name] | --no-vcd]
  //
  // Examples:
  //   ./build/main
  //   ./build/main --no-vcd
  //   ./build/main --vcd mywave
  // ----------------------------
  bool vcd_enable = (MODE == 7) ? false : true;
  // FP2 batch 預設關 VCD，避免 64-core 波形檔過大
  std::string vcd_name = "simulator_wave";
  const std::string vcd_dir = "/home/wilson/SystemC/test_tool";

  std::string run_dir =
      (MODE == 7)
          ? getenv_or_default(
                "ASMAP_RUN_DIR",
                join_path(repo_root_from_cwd(), "FP2/submission/CA_FP2_G01"))
          : getenv_or_default(
                "ASMAP_RUN_DIR",
                join_path(repo_root_from_cwd(),
                          "SystemC/auto_sched_map/runs/llama_ffn_plan000_v0"));
  std::string instr_dir;
  std::string dram_dir;
  std::string ref_dir;
  std::string sim_out_dir;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];

    if (a == "--no-vcd") {
      vcd_enable = false;
    } else if (a == "--vcd") {
      vcd_enable = true;
      // 可選：下一個參數當作檔名（不要含副檔名）
      if ((i + 1) < argc) {
        std::string next = argv[i + 1];
        if (!next.empty() && next[0] != '-') {
          vcd_name = next;
          ++i;
        }
      }
    } else if (a.rfind("--vcd=", 0) == 0) {
      vcd_enable = true;
      vcd_name = a.substr(std::string("--vcd=").size());
      if (vcd_name.empty())
        vcd_name = "simulator_wave";
    } else if (a == "--run-dir") {
      if ((i + 1) < argc) {
        run_dir = argv[++i];
      }
    } else if (a.rfind("--run-dir=", 0) == 0) {
      run_dir = a.substr(std::string("--run-dir=").size());
    } else if (a == "--instr-dir") {
      if ((i + 1) < argc) {
        instr_dir = argv[++i];
      }
    } else if (a.rfind("--instr-dir=", 0) == 0) {
      instr_dir = a.substr(std::string("--instr-dir=").size());
    } else if (a == "--dram-dir") {
      if ((i + 1) < argc) {
        dram_dir = argv[++i];
      }
    } else if (a.rfind("--dram-dir=", 0) == 0) {
      dram_dir = a.substr(std::string("--dram-dir=").size());
    } else if (a == "--ref-dir") {
      if ((i + 1) < argc) {
        ref_dir = argv[++i];
      }
    } else if (a.rfind("--ref-dir=", 0) == 0) {
      ref_dir = a.substr(std::string("--ref-dir=").size());
    } else if (a == "--sim-out-dir") {
      if ((i + 1) < argc) {
        sim_out_dir = argv[++i];
      }
    } else if (a.rfind("--sim-out-dir=", 0) == 0) {
      sim_out_dir = a.substr(std::string("--sim-out-dir=").size());
    } else {
      // ignore unknown flags
    }
  }
  if (instr_dir.empty()) {
    const std::string mode_instr_dir =
        getenv_or_default("ASMAP_INSTR_DIR", default_instr_dir_for_mode(run_dir));

    if (!mode_instr_dir.empty()) {
      instr_dir = mode_instr_dir;
    } else {
      instr_dir = join_path(run_dir, "instr");
    }
  }

  if (dram_dir.empty()) {
    const std::string mode_dram_dir =
        getenv_or_default("ASMAP_DRAM_DIR", default_dram_dir_for_mode(run_dir));

    if (!mode_dram_dir.empty()) {
      dram_dir = mode_dram_dir;
    } else {
      dram_dir = join_path(run_dir, "dram");
    }
  }

  if (ref_dir.empty())
    ref_dir = getenv_or_default("ASMAP_REF_DIR", join_path(run_dir, "ref"));

  if (sim_out_dir.empty()) {
    std::string mode_sim_out_dir =
        getenv_or_default("ASMAP_SIM_OUT_DIR", default_sim_out_dir_for_mode());

    if (mode_sim_out_dir.empty() && MODE == 7) {
      mode_sim_out_dir = join_path(repo_root_from_cwd(),
                                   "FP2/outcome/" +
                                       fp2_group_suffix_from_path(run_dir));
    }

    if (!mode_sim_out_dir.empty()) {
      sim_out_dir = mode_sim_out_dir;
    } else {
      sim_out_dir = join_path(run_dir, "sim_out");
    }
  }

  set_env_path("ASMAP_RUN_DIR", run_dir);
  set_env_path("ASMAP_INSTR_DIR", instr_dir);
  set_env_path("ASMAP_DRAM_DIR", dram_dir);
  set_env_path("ASMAP_REF_DIR", ref_dir);
  set_env_path("ASMAP_SIM_OUT_DIR", sim_out_dir);
  set_env_path("ASMAP_STATE_DIR",
               getenv_or_default("ASMAP_STATE_DIR",
                                 join_path(sim_out_dir, "state")));
  set_env_path("ASMAP_TOKEN_IN_PATH",
               getenv_or_default(
                   "ASMAP_TOKEN_IN_PATH",
                   join_path(repo_root_from_cwd(),
                             "SystemC/DRAM_json/input_tensor.bin")));
  // 2) clock / signals
  sc_clock clk("clk", sc_time(1, SC_NS)); // 1ns 週期
  sc_signal<bool> all_done_sig("all_done_sig");

  // 3) 建立頂層 Simulator（內含 NX×NY 個 Tcore + DRAM + NoC）
  static constexpr int LANES = 8;
  using SimType = Simulator<T, NX, NY, LANES, MODE>;
  SimType sim("Simulator");
  sim.clk(clk);
  sim.all_done(all_done_sig);

  // 4) DoneWatcher：負責在 all_done 正緣時呼叫 sc_stop()
  DoneWatcher watcher("DoneWatcher");
  watcher.all_done(all_done_sig);

  // 5) 每顆 Tcore 一個 BusyMeter
  BusyMeter *tcore_meter[NCORES];
  for (int cid = 0; cid < NCORES; ++cid) {
    std::ostringstream nm;
    nm << "BusyMeter_Tcore" << cid;

    tcore_meter[cid] = new BusyMeter(nm.str().c_str());
    tcore_meter[cid]->clk(clk);
    tcore_meter[cid]->all_done(all_done_sig);

    tcore_meter[cid]->core_done(sim.tcore_done_sig[cid]);
    tcore_meter[cid]->vbusy(sim.tcore_vector_busy_sig[cid]);
    tcore_meter[cid]->abusy(sim.tcore_array_busy_sig[cid]);
    tcore_meter[cid]->dbusy(sim.tcore_dma_busy_sig[cid]);
    tcore_meter[cid]->rsbusy(sim.tcore_Router_send_busy_sig[cid]);
    tcore_meter[cid]->rrbusy(sim.tcore_Router_recv_busy_sig[cid]);
    tcore_meter[cid]->dram_rbusy(sim.tcore_dram_rd_busy_sig[cid]);
    tcore_meter[cid]->dram_wbusy(sim.tcore_dram_wr_busy_sig[cid]);
  }

  // 6) NoC BusyMeter
  NocBusyMeter noc_meter("NocBusyMeter");
  noc_meter.clk(clk);
  noc_meter.all_done(all_done_sig);
  noc_meter.noc_busy(sim.noc_busy_sig);

  // 7) VCD 波形（可選）
  sc_trace_file *tf = nullptr;
  if (vcd_enable) {
    std::string vcd_path = vcd_dir + "/" + vcd_name;
    tf = sc_create_vcd_trace_file(vcd_path.c_str());
    tf->set_time_unit(100, SC_PS);

    sc_trace(tf, clk, "clk");
    sc_trace(tf, all_done_sig, "all_done_sig");

    for (int cid = 0; cid < NCORES; ++cid) {
      std::string base = "tcore" + std::to_string(cid);
      sc_trace(tf, sim.tcore_vector_busy_sig[cid],
               (base + "_vector_busy").c_str());
      sc_trace(tf, sim.tcore_array_busy_sig[cid],
               (base + "_array_busy").c_str());
      sc_trace(tf, sim.tcore_dma_busy_sig[cid], (base + "_dma_busy").c_str());
      sc_trace(tf, sim.tcore_Router_send_busy_sig[cid],
               (base + "_router_send_busy").c_str());
      sc_trace(tf, sim.tcore_Router_recv_busy_sig[cid],
               (base + "_router_recv_busy").c_str());
      sc_trace(tf, sim.tcore_dram_rd_busy_sig[cid],
               (base + "_dram_rd_busy").c_str());
      sc_trace(tf, sim.tcore_dram_wr_busy_sig[cid],
               (base + "_dram_wr_busy").c_str());
    }

    sc_trace(tf, sim.noc_busy_sig, "noc_busy");
  }

  // 8) 跑模擬 + execution time
  {
    std::ostringstream oss;
    oss << "VCD: " << (vcd_enable ? "ON" : "OFF");
    if (vcd_enable)
      oss << " (" << vcd_dir << "/" << vcd_name << ".vcd)";
    main_info(oss.str());
  }

  const auto wall_t0 = std::chrono::steady_clock::now();
  sc_start(); // 直到 DoneWatcher 呼叫 sc_stop()
  const auto wall_t1 = std::chrono::steady_clock::now();

  const std::chrono::duration<double> wall_sec = wall_t1 - wall_t0;
  const sc_time sim_t = sc_time_stamp();

  // 模擬秒數（sc_time 轉 seconds）
  const double sim_sec = sim_t.to_seconds();
  const double speed =
      (wall_sec.count() > 0.0) ? (sim_sec / wall_sec.count()) : 0.0;

  {
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss << std::setprecision(6);
    oss << "Execution time (wall): " << wall_sec.count() << " s\n"
        << "Simulated time: " << sim_t << " (" << sim_sec << " s)\n"
        << "Sim speed: " << speed << " sim_sec / real_sec";
    main_info(oss.str());
  }

  // 9) 收尾
  if (tf)
    sc_close_vcd_trace_file(tf);

  main_info("Simulation finished.");
  return 0;
}
