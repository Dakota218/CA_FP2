// /home/wilson/SystemC/include/NoC.h
#pragma once

#include "mem_types.h"
#include <systemc.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <sstream>
#include <vector>
#include "hw_config_selector.h"

/*
           +------------------------ NoC --------------------------+
Tcore(src) → noc_data_in[src_id] ---搬資料---> noc_data_out[dst_id] → Tcore(dst)
                    ^                                 ^
                    |                                 |
              read from FIFO                   write to FIFO

【本版本重點（你要的行為）】
  1) multicast 的每個目的地都有自己的 Manhattan hop（3 * dist）
     - dst1/dst2(1 hop) 可以先到、dst3(2 hop) 晚到
  2) deliver 改為 per-dst：某個 dst hop 到了、且該 dst 的 FIFO 有空間，就先送該
dst
     - 不再「等所有 dst 都 ready 才一起送」
  3) payload 仍維持「只從 src 讀一次」：
     - NoC 會在內部先把 payload 讀出存起來 (payload_ready)
     - 然後複製投遞到各 dst
  4) full-duplex 規則：
     - 每個 node 同一時間最多一條「當 src」(send)
     - 每個 node 同一時間最多一條「當 dst」(recv)
     - ★但 dst 的 recv_busy 會在「該 dst deliver 完成」後就立刻釋放（不用等整筆
multicast 全部完成）

【Congestion（新增、最小侵入）】
  - 保留你原本的 hop_left = 3*Manhattan 的 timing 語意
  - 但在啟動 transfer 時，額外估算 congestion_wait（排隊等待）並加到 hop_left
  - congestion_wait 的估算依據：每個 router 的每個 output direction 有一個抽象
FIFO(outq)
    - outq 的 service time 會考慮「整條 service time」：3(每 hop pipeline) +
flit_cycles
    - 只把 flit 用在「佔用資源 / 排隊等待」(讓長封包更容易造成塞車)
    - 不把 flit 再加到 base hop（避免與 Router 端搬運重複計算）
*/

template <typename T,    
          int LANES = (ActiveHWConfig::SRAM_DMA_WIDTH < ActiveHWConfig::NOC_LINK_WIDTH
                     ? ActiveHWConfig::SRAM_DMA_WIDTH
                     : ActiveHWConfig::NOC_LINK_WIDTH),
          int NX = 2, int NY = 2>
class NoC : public sc_core::sc_module {
public:
  static constexpr int NUM_NODES = NX * NY;

  sc_core::sc_in<bool> clk;

  sc_core::sc_fifo_in<NocCmd<T>> noc_cmd_in[NUM_NODES];   // from src Router
  sc_core::sc_fifo_out<NocCmd<T>> noc_cmd_out[NUM_NODES]; // to dst Router

  sc_core::sc_fifo_in<T> noc_data_in[NUM_NODES];   // from src Router
  sc_core::sc_fifo_out<T> noc_data_out[NUM_NODES]; // to dst Router

  sc_core::sc_out<bool> noc_busy;

  SC_HAS_PROCESS(NoC);

  NoC(sc_core::sc_module_name name)
      : sc_core::sc_module(name), noc_busy("noc_busy") {
    SC_THREAD(run);
    sensitive << clk.pos();
    dont_initialize();
  }

private:
  struct DestState {
    int dst_id = -1;
    int hop_left = 0; // 3 * manhattan_dist(src, dst) + congestion_wait
    bool delivered = false;
  };

  struct TransferState {
    NocCmd<T> cmd;
    int src_id = -1;

    bool is_mcast = false;
    int dst_count = 0;
    std::array<DestState, NOC_MCAST_MAX_DESTS> dsts;

    bool payload_ready = false;
    std::vector<T> payload; // len elements
  };

  // -----------------------------
  // Congestion model parameters
  // -----------------------------
  static constexpr int NUM_DIRS = 4; // E, W, S, N
  static constexpr int OUTBUF_DEPTH = 4;

  enum Dir : int { E = 0, W = 1, S = 2, N = 3 };

  int coord_to_id(sc_dt::sc_uint<5> x, sc_dt::sc_uint<5> y) const {
    int xi = static_cast<int>(x.to_uint());
    int yi = static_cast<int>(y.to_uint());
    if (xi < 0 || xi >= NX || yi < 0 || yi >= NY) {
      std::stringstream ss;
      ss << "coord_to_id: (x,y)=(" << xi << "," << yi << ") out of mesh range";
      SC_REPORT_ERROR(this->name(), ss.str().c_str());
    }
    return yi * NX + xi;
  }

  void id_to_xy(int id, int &x, int &y) const {
    if (id < 0 || id >= NUM_NODES) {
      std::ostringstream oss;
      oss << "id_to_xy: id=" << id << " out of range";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      x = y = 0;
      return;
    }
    x = id % NX;
    y = id / NX;
  }

  int manhattan_dist_xy(sc_dt::sc_uint<5> sx, sc_dt::sc_uint<5> sy,
                        sc_dt::sc_uint<5> dx, sc_dt::sc_uint<5> dy) const {
    int sxi = static_cast<int>(sx.to_uint());
    int syi = static_cast<int>(sy.to_uint());
    int dxi = static_cast<int>(dx.to_uint());
    int dyi = static_cast<int>(dy.to_uint());
    return std::abs(sxi - dxi) + std::abs(syi - dyi);
  }

  int hop_latency_cycles_unicast(const NocCmd<T> &cmd, sc_dt::sc_uint<5> dx,
                                 sc_dt::sc_uint<5> dy) const {
    return 3 * manhattan_dist_xy(cmd.src_x, cmd.src_y, dx, dy);
  }

  // 每 hop 的「資源佔用 service time」
  // - 3 cycles：對應你原本每 hop pipeline latency
  // - flit_cycles：讓長 payload 在 network resource
  // 上佔用更久，造成更合理的塞車
  int hop_service_cycles(int len) const {
    const int L = std::max(1, len);
    const int flit_cycles = (L + LANES - 1) / LANES; // ceil(len / LANES)
    return 3 + flit_cycles;
  }

  struct HopRef {
    int router_id = -1; // current router id
    int dir = 0;        // output direction used at this hop
  };

  // XY routing：先走 X，再走 Y
  std::vector<HopRef> build_xy_path(int src_id, int dst_id) const {
    std::vector<HopRef> path;
    int x, y, dx, dy;
    id_to_xy(src_id, x, y);
    id_to_xy(dst_id, dx, dy);

    int cur_id = src_id;

    // X dimension
    while (x != dx) {
      if (dx > x) {
        path.push_back(HopRef{cur_id, Dir::E});
        ++x;
      } else {
        path.push_back(HopRef{cur_id, Dir::W});
        --x;
      }
      cur_id = y * NX + x;
    }

    // Y dimension
    while (y != dy) {
      if (dy > y) {
        path.push_back(HopRef{cur_id, Dir::S});
        ++y;
      } else {
        path.push_back(HopRef{cur_id, Dir::N});
        --y;
      }
      cur_id = y * NX + x;
    }

    return path;
  }

  // ----------------------------------------------------------
  // OutQ entry (方案一): ready_at + remaining
  // ----------------------------------------------------------
  struct OutEntry {
    long long ready_at = 0; // 此 hop 最早會使用此 output port 的 cycle
    int remaining = 0;      // service cycles remaining (svc)
  };

  // outq[r][dir]：該 router 的該輸出方向，FIFO 排隊的封包
  using OutQ =
      std::array<std::array<std::deque<OutEntry>, NUM_DIRS>, NUM_NODES>;

  // 給定目前 cycle=now，計算此 output port 把目前 queue
  // 全部服務完後，最早變空的時間點 服務規則：
  //   - 只有 now >= front.ready_at 才能服務（每 cycle 扣 1）
  //   - 一筆服務完 pop，下一筆同理
  long long port_finish_time(const std::deque<OutEntry> &q,
                             long long now) const {
    long long t = now;
    for (const auto &e : q) {
      const int rem = std::max(0, e.remaining);
      if (rem == 0) {
        // remaining=0 視為已完成；正常情況不應該留在 queue，但這裡保守處理
        continue;
      }
      long long start = std::max(t, e.ready_at);
      t = start + rem;
    }
    return t;
  }

  // 估算 congestion wait（排隊等待）並在 outq 上「預約」資源（push entry）
  int estimate_and_reserve_congestion(OutQ &outq, long long now, int src_id,
                                      int dst_id, int len) const {
    (void)src_id; // src_id 目前不直接用於 congestion 計算（由 path 決定
                  // router_id）

    const auto path = build_xy_path(src_id, dst_id);
    if (path.empty())
      return 0;

    const int svc = hop_service_cycles(len);
    long long total_wait = 0;

    for (int hi = 0; hi < static_cast<int>(path.size()); ++hi) {
      const int r = path[hi].router_id;
      const int d = path[hi].dir;
      if (r < 0 || r >= NUM_NODES || d < 0 || d >= NUM_DIRS)
        continue;

      // 這一 hop 最早會到達此 output port 的時間點（absolute cycle）
      const long long arrival = now + 3LL * hi + total_wait;

      // 目前 queue 全部服務完的時間點（absolute cycle）
      const long long finish = port_finish_time(outq[r][d], now);

      // FIFO：本 hop 若 arrival 早於 finish，就必須等到 finish 才能開始服務
      const long long start = std::max(arrival, finish);
      const long long wait_here = start - arrival;

      total_wait += wait_here;

      // 預約資源：把這個 hop 加入 outq
      // ★重點：用 ready_at=arrival，確保「遠 hop 晚到晚開始，晚釋放」
      if (static_cast<int>(outq[r][d].size()) < OUTBUF_DEPTH) {
        outq[r][d].push_back(OutEntry{arrival, svc});
      } else {
        outq[r][d].push_back(OutEntry{arrival, svc});
        // 若想更 aggressive，可在此加入額外 penalty，例如：
        // total_wait += svc;
      }
    }

    if (total_wait < 0)
      total_wait = 0;
    if (total_wait > std::numeric_limits<int>::max())
      return std::numeric_limits<int>::max();
    return static_cast<int>(total_wait);
  }

  // --- helper: get bound sc_fifo capacity; return -1 if not a plain sc_fifo ---
  template <typename U>
  int fifo_capacity(sc_core::sc_fifo_in<U> &p) const {
    auto *ifc = p.get_interface();
    auto *fifo = dynamic_cast<sc_core::sc_fifo<U> *>(ifc);
    if (!fifo) return -1;
    // sc_fifo 沒有公開 size(); 用 free+avail 推回容量
    return fifo->num_free() + fifo->num_available();
  }

  template <typename U>
  int fifo_capacity(sc_core::sc_fifo_out<U> &p) const {
    auto *ifc = p.get_interface();
    auto *fifo = dynamic_cast<sc_core::sc_fifo<U> *>(ifc);
    if (!fifo) return -1;
    return fifo->num_free() + fifo->num_available();
  }

  void check_fifo_depth_or_error(int cap, int len, const char *tag, int nid) const {
    if (cap >= 0 && cap < len) {
      std::ostringstream oss;
      oss << "NoC FIFO depth check failed: " << tag
          << " nid=" << nid << " depth=" << cap << " < len=" << len
          << " (this will deadlock because NoC requires full-packet buffering)";
      SC_REPORT_ERROR(this->name(), oss.str().c_str());
      sc_core::sc_stop(); 
    }
  }
  // ----------------------------------------------------------
  // Congestion helpers for MULTICAST (shared prefix only reserve once)
  // ----------------------------------------------------------

  // 只估算 congestion wait，不 reserve
  int estimate_congestion_wait_only(const OutQ &outq, long long now, int src_id,
                                    int dst_id, int len) const {
    (void)len; // len 在此版本 wait-only 不直接用；service 長度已在 queue
               // remaining 內反映

    const auto path = build_xy_path(src_id, dst_id);
    if (path.empty())
      return 0;

    long long total_wait = 0;

    for (int hi = 0; hi < static_cast<int>(path.size()); ++hi) {
      const int r = path[hi].router_id;
      const int d = path[hi].dir;
      if (r < 0 || r >= NUM_NODES || d < 0 || d >= NUM_DIRS)
        continue;

      const long long arrival = now + 3LL * hi + total_wait;

      const long long finish = port_finish_time(outq[r][d], now);
      const long long start = std::max(arrival, finish);
      const long long wait_here = start - arrival;

      total_wait += wait_here;
    }

    if (total_wait < 0)
      total_wait = 0;
    if (total_wait > std::numeric_limits<int>::max())
      return std::numeric_limits<int>::max();
    return static_cast<int>(total_wait);
  }

  // multicast：把所有目的地的 XY path 做 union
  // - 同一條 (router, dir) 只 reserve 一次（shared prefix 不重複佔用）
  // - 方案一的 ready_at：用「shared hop 在路徑中最早的 hi」決定 ready_at = now
  // + 3*min_hi
  void reserve_congestion_mcast_union(
      OutQ &outq, long long now, int src_id,
      const std::array<int, NOC_MCAST_MAX_DESTS> &dst_ids, int dst_count,
      int len) const {
    const int svc = hop_service_cycles(len);

    std::array<std::array<bool, NUM_DIRS>, NUM_NODES> reserved{};
    std::array<std::array<int, NUM_DIRS>, NUM_NODES> min_hi{};

    for (int r = 0; r < NUM_NODES; ++r) {
      for (int d = 0; d < NUM_DIRS; ++d) {
        reserved[r][d] = false;
        min_hi[r][d] = std::numeric_limits<int>::max();
      }
    }

    // 先收集 union + 以及每條 (r,d) 的最早 hop index
    for (int di = 0; di < dst_count; ++di) {
      const int dst_id = dst_ids[di];
      if (dst_id < 0)
        continue;

      const auto path = build_xy_path(src_id, dst_id);
      for (int hi = 0; hi < static_cast<int>(path.size()); ++hi) {
        const int r = path[hi].router_id;
        const int d = path[hi].dir;
        if (r < 0 || r >= NUM_NODES || d < 0 || d >= NUM_DIRS)
          continue;

        reserved[r][d] = true;
        if (hi < min_hi[r][d]) {
          min_hi[r][d] = hi;
        }
      }
    }

    // 再依 union 做 reserve（shared hop 只 push 一次）
    for (int r = 0; r < NUM_NODES; ++r) {
      for (int d = 0; d < NUM_DIRS; ++d) {
        if (!reserved[r][d])
          continue;

        const int hi = min_hi[r][d];
        if (hi == std::numeric_limits<int>::max())
          continue;

        const long long arrival = now + 3LL * hi;

        if (static_cast<int>(outq[r][d].size()) < OUTBUF_DEPTH) {
          outq[r][d].push_back(OutEntry{arrival, svc});
        } else {
          outq[r][d].push_back(OutEntry{arrival, svc});
          // buffer 滿了：仍 push（避免死鎖），可在此加入額外 penalty
        }
      }
    }
  }

  // 將 multicast cmd 轉成某一個目的地的 unicast cmd
  NocCmd<T> make_unicast_from_mcast(const NocCmd<T> &mc, int di) const {
    NocCmd<T> uc = mc;

    const auto &d = mc.mcast_dsts[di];
    uc.dst_x = d.dst_x;
    uc.dst_y = d.dst_y;
    uc.dst_buf = d.dst_buf;
    uc.dst_offset = d.dst_offset;

    uc.is_mcast = false;
    uc.mcast_num = 0;
    return uc;
  }

  void run() {
    std::deque<NocCmd<T>> pending_cmds[NUM_NODES];
    std::vector<TransferState> active;

    bool node_send_busy[NUM_NODES];
    bool node_recv_busy[NUM_NODES];
    for (int i = 0; i < NUM_NODES; ++i) {
      node_send_busy[i] = false;
      node_recv_busy[i] = false;
    }

    // Congestion state (abstract output queues)
    OutQ outq; // default-constructed; deques empty

    // 抽象 cycle 計數器（每次 wait() 後 +1）
    long long cycle = 0;

    noc_busy.write(false);

    while (true) {
      wait();
      ++cycle;

      // ----------------------------------------------------------
      // (方案一) Congestion service：
      // 每拍每個 output port 服務 1 cycle，但必須 now>=ready_at 才能扣
      // remaining
      // ----------------------------------------------------------
      for (int r = 0; r < NUM_NODES; ++r) {
        for (int d = 0; d < NUM_DIRS; ++d) {
          auto &q = outq[r][d];
          if (q.empty())
            continue;

          // 若 front 還沒到 ready_at，這一拍不能服務（保持 idle）
          if (cycle < q.front().ready_at)
            continue;

          q.front().remaining -= 1;
          if (q.front().remaining <= 0) {
            q.pop_front();
          }
        }
      }

      // ----------------------------------------------------------
      // Step 1：吸 cmd 進 pending
      // ----------------------------------------------------------
      for (int nid = 0; nid < NUM_NODES; ++nid) {
        while (noc_cmd_in[nid].num_available() > 0) {
          NocCmd<T> c = noc_cmd_in[nid].read();

          if (c.length == 0) {
            SC_REPORT_ERROR(this->name(), "NoC: NocCmd.length must be > 0");
            continue;
          }

          if (c.is_mcast) {
            const int n = static_cast<int>(c.mcast_num.to_uint());
            if (n <= 0 || n > NOC_MCAST_MAX_DESTS) {
              SC_REPORT_ERROR(this->name(),
                              "NoC: multicast cmd has invalid mcast_num");
              continue;
            }
          }

          pending_cmds[nid].push_back(c);
        }
      }

      // ----------------------------------------------------------
      // Step 2：嘗試啟動新的 transfer（只鎖端點，不用等所有 dst FIFO 都 ready）
      // ----------------------------------------------------------
      for (int nid = 0; nid < NUM_NODES; ++nid) {
        if (pending_cmds[nid].empty())
          continue;

        const NocCmd<T> &c = pending_cmds[nid].front();

        int src = coord_to_id(c.src_x, c.src_y);
        if (src != nid) {
          std::ostringstream oss;
          oss << "NoC: drop cmd because src_id(" << src << ") != ingress nid("
              << nid << ") src_xy=(" << c.src_x.to_uint() << ","
              << c.src_y.to_uint() << ")";
          SC_REPORT_ERROR(this->name(), oss.str().c_str());

          pending_cmds[nid].pop_front();
          continue;
        }
        if (src < 0 || src >= NUM_NODES) {
          SC_REPORT_ERROR(this->name(),
                          "NoC: src out of range when starting transfer");
          pending_cmds[nid].pop_front();
          continue;
        }

        // src：同一時間最多一條 send
        if (node_send_busy[src])
          continue;

        // 解析目的地清單並檢查 dst busy
        std::array<int, NOC_MCAST_MAX_DESTS> dst_ids{};
        dst_ids.fill(-1);

        bool is_mcast = c.is_mcast;
        int dst_count = 0;

        if (!is_mcast) {
          int dst = coord_to_id(c.dst_x, c.dst_y);
          if (dst < 0 || dst >= NUM_NODES) {
            SC_REPORT_ERROR(this->name(),
                            "NoC: dst out of range when starting transfer");
            pending_cmds[nid].pop_front();
            continue;
          }
          dst_ids[0] = dst;
          dst_count = 1;
        } else {
          const int n = static_cast<int>(c.mcast_num.to_uint());
          if (n > NOC_MCAST_MAX_DESTS)
            SC_REPORT_ERROR(this->name(),
                            "NoC: multicast cmd has too many destinations");
          dst_count = std::min(n, NOC_MCAST_MAX_DESTS);

          // 檢查同一筆 multicast 內是否有重複 dst_id
          std::array<bool, NUM_NODES> seen_dst{};
          seen_dst.fill(false);

          for (int i = 0; i < dst_count; ++i) {
            const auto &d = c.mcast_dsts[i];
            int dst = coord_to_id(d.dst_x, d.dst_y);

            // dst out-of-range
            if (dst < 0 || dst >= NUM_NODES) {
              std::ostringstream oss;
              oss << "NoC: mcast dst out of range when starting transfer: "
                  << "di=" << i << " dst_xy=(" << d.dst_x.to_uint() << ","
                  << d.dst_y.to_uint() << ") -> dst_id=" << dst;
              SC_REPORT_ERROR(this->name(), oss.str().c_str());
              pending_cmds[nid].pop_front();
              dst_count = 0;
              break;
            }

            // duplicate dst in the same multicast
            if (seen_dst[dst]) {
              std::ostringstream oss;
              oss << "NoC: multicast has duplicated destination: "
                  << "di=" << i << " dst_id=" << dst << " dst_xy=("
                  << d.dst_x.to_uint() << "," << d.dst_y.to_uint() << ")";
              SC_REPORT_ERROR(this->name(), oss.str().c_str());

              pending_cmds[nid].pop_front();
              dst_count = 0;
              break;
            }
            seen_dst[dst] = true;
            dst_ids[i] = dst;
          }
          if (dst_count == 0) {
            SC_REPORT_ERROR(this->name(),
                            "NoC: multicast cmd has no valid destinations");
            continue;
          }
        }

        
        // 任一 dst 已在 recv → 不能啟動（維持 full-duplex 規則）
        bool any_dst_busy = false;
        for (int i = 0; i < dst_count; ++i) {
          int dst = dst_ids[i];
          if (dst < 0)
            continue;
          if (node_recv_busy[dst]) {
            any_dst_busy = true;
            break;
          }
        }
        if (any_dst_busy)
          continue;

        // 建立 TransferState
        TransferState ts{};
        ts.cmd = c;
        ts.src_id = src;
        ts.is_mcast = is_mcast;
        ts.dst_count = dst_count;
        for (auto &d : ts.dsts)
          d = DestState{};

        const int len = static_cast<int>(c.length.to_uint());
        // src side payload fifo must hold full packet
        {
          const int cap_in = fifo_capacity(noc_data_in[src]);
          check_fifo_depth_or_error(cap_in, len, "noc_data_in", src);
        }
        for (int i = 0; i < dst_count; ++i) {
          int dst = dst_ids[i];
          if (dst < 0) continue;
          const int cap_out = fifo_capacity(noc_data_out[dst]);
          check_fifo_depth_or_error(cap_out, len, "noc_data_out", dst);
        }
        // per-dst hop 設定（★加上 congestion_wait）
        if (!is_mcast) {
          DestState d0;
          d0.dst_id = dst_ids[0];

          const int base_hop = hop_latency_cycles_unicast(c, c.dst_x, c.dst_y);
          const int wait =
              estimate_and_reserve_congestion(outq, cycle, src, d0.dst_id, len);

          d0.hop_left = base_hop + wait;
          d0.delivered = false;
          ts.dsts[0] = d0;
        } else {
          // multicast：shared prefix 只 reserve 一次（union of paths）
          // 1) snapshot 計算每個 dst 的等待（不預約資源，避免把自己算進去）
          // 2) 再對真正 outq 做 union reserve（shared hop 只 push 一次）
          OutQ outq_snapshot = outq;

          std::array<int, NOC_MCAST_MAX_DESTS> wait_each{};
          wait_each.fill(0);

          for (int di = 0; di < dst_count; ++di) {
            wait_each[di] = estimate_congestion_wait_only(
                outq_snapshot, cycle, src, dst_ids[di], len);
          }

          reserve_congestion_mcast_union(outq, cycle, src, dst_ids, dst_count,
                                         len);

          for (int di = 0; di < dst_count; ++di) {
            DestState dd;
            dd.dst_id = dst_ids[di];

            const int base_hop = hop_latency_cycles_unicast(
                c, c.mcast_dsts[di].dst_x, c.mcast_dsts[di].dst_y);

            dd.hop_left = base_hop + wait_each[di];
            dd.delivered = false;
            ts.dsts[di] = dd;
          }
        }

        ts.payload_ready = false;
        ts.payload.clear();

        active.push_back(ts);

        // 鎖端點 busy
        node_send_busy[src] = true;
        for (int i = 0; i < dst_count; ++i) {
          int dst = dst_ids[i];
          if (dst >= 0)
            node_recv_busy[dst] = true;
        }

        pending_cmds[nid].pop_front();
      }

      // ----------------------------------------------------------
      // Step 3：推進 active transfers
      //   3-1) 每拍 hop_left--
      //   3-2) 一旦 payload_ready，就可以 per-dst deliver（誰先到先送）
      // ----------------------------------------------------------
      for (auto it = active.begin(); it != active.end();) {
        TransferState &ts = *it;
        const int len = static_cast<int>(ts.cmd.length.to_uint());

        // 3-1: hop 倒數（每個 dst 各自）
        if (ts.payload_ready) {
          for (int di = 0; di < ts.dst_count; ++di) {
            if (!ts.dsts[di].delivered && ts.dsts[di].hop_left > 0) {
              --ts.dsts[di].hop_left;
            }
          }
        }

        // 3-2: 若 payload 尚未 ready，嘗試把 src data 一次讀出來（保持 src
        // 只注入一次）
        if (!ts.payload_ready) {
          if (noc_data_in[ts.src_id].num_available() >= len) {
            ts.payload.clear();
            ts.payload.reserve(len);
            for (int i = 0; i < len; ++i) {
              ts.payload.push_back(noc_data_in[ts.src_id].read());
            }
            ts.payload_ready = true;
          }
        }

        // 3-3: per-dst deliver（只要該 dst hop 到了且 FIFO 有空間且
        // payload_ready）
        if (ts.payload_ready) {
          for (int di = 0; di < ts.dst_count; ++di) {
            DestState &d = ts.dsts[di];
            if (d.dst_id < 0)
              continue;
            if (d.delivered)
              continue;
            if (d.hop_left > 0)
              continue;

            const int dst = d.dst_id;

            // dst FIFO 要能容納：1 cmd + len data
            if (noc_cmd_out[dst].num_free() == 0) {
              continue;
            }
            if (noc_data_out[dst].num_free() < len) {
              continue;
            }

            // 寫 cmd（mcast 轉 unicast）+ 複製 payload
            if (!ts.is_mcast) {
              noc_cmd_out[dst].write(ts.cmd);
            } else {
              NocCmd<T> uc = make_unicast_from_mcast(ts.cmd, di);
              noc_cmd_out[dst].write(uc);
            }
            for (int i = 0; i < len; ++i) {
              noc_data_out[dst].write(ts.payload[i]);
            }

            d.delivered = true;

            // ★此 dst 收到後就釋放 recv_busy（不必等其他 dst）
            node_recv_busy[dst] = false;
          }
        }

        // 3-4: 如果全部 dst 都 delivered → 釋放 src busy，移除 transfer
        bool all_done = true;
        for (int di = 0; di < ts.dst_count; ++di) {
          if (!ts.dsts[di].delivered) {
            all_done = false;
            break;
          }
        }

        if (all_done) {
          node_send_busy[ts.src_id] = false;
          it = active.erase(it);
        } else {
          ++it;
        }
      }

      // ----------------------------------------------------------
      // Step 4：更新 NoC busy flag
      // ----------------------------------------------------------
      noc_busy.write(!active.empty());
    }
  }
};
