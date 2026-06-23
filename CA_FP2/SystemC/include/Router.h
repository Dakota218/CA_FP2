// /home/wilson/SystemC/include/Router.h
#pragma once

#include "hw_config_selector.h"
#include "mem_types.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <sstream>
#include <systemc.h>
#include <utility>
#include <vector>
/*
 * Router (full-duplex: send_fsm + recv_fsm)
 * ----------------------------------------------------------------------------
 * 【本版本重點】
 *  - multicast 改成「真正 multicast」：只送 1 個 is_mcast cmd + 只注入 1 份
 * payload
 *  - 不再在 Router 端展開成多次 unicast（不再複製 num 份 payload 進 NoC）
 *
 *  前提：你的 NoC.h 必須支援 is_mcast + mcast_dsts[]，並且會在 NoC
 * 內部複製投遞。
 */

template <typename T, int MODE = 0> class Router : public sc_core::sc_module {
public:
  sc_core::sc_in<bool> clk;

  // =========================================================================
  // Controller -> Router
  // =========================================================================
  sc_core::sc_in<bool> start;   // one-pulse
  sc_core::sc_in<bool> is_send; // 1=SEND, 0=RECV

  sc_core::sc_in<sc_dt::sc_uint<4>> src_buf;
  sc_core::sc_in<sc_dt::sc_uint<4>> dst_buf;
  sc_core::sc_in<sc_dt::sc_uint<16>> src_offset;
  sc_core::sc_in<sc_dt::sc_uint<16>> dst_offset;
  sc_core::sc_in<sc_dt::sc_uint<16>> length;

  // local coordinate
  sc_core::sc_in<sc_dt::sc_uint<5>> src_x;
  sc_core::sc_in<sc_dt::sc_uint<5>> src_y;

  // unicast dest coordinate
  sc_core::sc_in<sc_dt::sc_uint<5>> dst_x;
  sc_core::sc_in<sc_dt::sc_uint<5>> dst_y;
  sc_core::sc_in<sc_dt::sc_uint<32>> hash;
  // =========================================================================
  // Multicast extension (Controller -> Router)
  // =========================================================================
  static constexpr int MAX_MCAST_DESTS = 64;

  sc_core::sc_in<bool> is_mcast;
  sc_core::sc_in<sc_dt::sc_uint<7>> mcast_num;

  sc_core::sc_in<sc_dt::sc_uint<5>> mcast_dst_x[MAX_MCAST_DESTS];
  sc_core::sc_in<sc_dt::sc_uint<5>> mcast_dst_y[MAX_MCAST_DESTS];
  sc_core::sc_in<sc_dt::sc_uint<4>> mcast_dst_buf[MAX_MCAST_DESTS];
  sc_core::sc_in<sc_dt::sc_uint<16>> mcast_dst_offset[MAX_MCAST_DESTS];

  // =========================================================================
  // Busy/Done
  // =========================================================================
  sc_core::sc_out<bool> Router_send_busy;
  sc_core::sc_out<bool> Router_send_done;
  sc_core::sc_out<bool> Router_recv_busy;
  sc_core::sc_out<bool> Router_recv_done;

  // =========================================================================
  // RECV dst-buf handshake (Router -> Core_Controller)
  // =========================================================================
  sc_core::sc_out<sc_dt::sc_uint<4>> recv_dst_buf;
  sc_core::sc_out<bool> recv_dst_buf_valid;
  sc_core::sc_in<bool> recv_buf_grant;

  // =========================================================================
  // Local buffers
  // =========================================================================
  std::array<sc_core::sc_vector<sc_core::sc_signal<T>> *,
             ActiveHWConfig::NUM_BUFS>
      bufs;

  // shadow write (recv path writes shadow; Tcore commits later)
  std::array<std::vector<T> *, ActiveHWConfig::NUM_BUFS> shadow_bufs;

  // =========================================================================
  // NoC FIFOs
  // =========================================================================
  sc_core::sc_fifo_out<NocCmd<T>> Router_cmd_out;
  sc_core::sc_fifo_in<NocCmd<T>> Router_cmd_in;
  sc_core::sc_fifo_out<T> Router_data_out;
  sc_core::sc_fifo_in<T> Router_data_in;

  SC_HAS_PROCESS(Router);

  Router(sc_core::sc_module_name name) : sc_core::sc_module(name) {
    bufs.fill(nullptr);
    shadow_bufs.fill(nullptr);

    SC_THREAD(send_fsm);
    sensitive << clk.pos();

    SC_THREAD(recv_fsm);
    sensitive << clk.pos();
  }

private:
  static constexpr int NOC_LANES =
      (ActiveHWConfig::SRAM_DMA_WIDTH < ActiveHWConfig::NOC_LINK_WIDTH)
          ? ActiveHWConfig::SRAM_DMA_WIDTH
          : ActiveHWConfig::NOC_LINK_WIDTH;
  static std::size_t cfg_buf_size(std::size_t idx) {
    return static_cast<std::size_t>(ActiveHWConfig::BUF_SIZES[idx]);
  }
  static_assert(ActiveHWConfig::SRAM_DMA_WIDTH > 0,
                "SRAM_DMA_WIDTH must be > 0");
  static_assert(ActiveHWConfig::NOC_LINK_WIDTH > 0,
                "NOC_LINK_WIDTH must be > 0");
  static_assert(NOC_LANES > 0, "Router width must be > 0");

  static constexpr int mesh_x_for_mode_() {
    if constexpr (MODE == 0) {
      return 2;
    } else if constexpr (MODE == 1) {
      return 1;
    } else if constexpr (MODE == 2) {
      return 2;
    } else if constexpr (MODE == 3) {
      return 4;
    } else if constexpr (MODE == 4) {
      return 8;
    } else if constexpr (MODE == 5) {
      return 2;
    } else if constexpr (MODE == 6) {
      return 4;
    } else if constexpr (MODE == 7) {
      return 8; // FP2 8x8 mesh
    } else {
      SC_REPORT_ERROR("Router", "Unsupported MODE");
      return 1;
    }
  }

  static int core_id_from_xy_(const sc_dt::sc_uint<5> &x,
                              const sc_dt::sc_uint<5> &y) {
    return static_cast<int>(y.to_uint()) * mesh_x_for_mode_() +
           static_cast<int>(x.to_uint());
  }

  void append_packet_debug_(std::ostringstream &oss,
                            const NocCmd<T> &cmd) const {
    oss << " packet_hash=" << cmd_hash_u32(cmd)
        << " src_core=" << core_id_from_xy_(cmd.src_x, cmd.src_y)
        << " dst_core=" << core_id_from_xy_(cmd.dst_x, cmd.dst_y) << " src=("
        << cmd.src_x << "," << cmd.src_y << ")"
        << " dst=(" << cmd.dst_x << "," << cmd.dst_y << ")"
        << " src_buf=" << cmd.src_buf << " src_offset=" << cmd.src_offset
        << " dst_buf=" << cmd.dst_buf << " dst_offset=" << cmd.dst_offset
        << " len=" << cmd.length;
  }

  void log_router_send_(const NocCmd<T> &cmd) const {
    std::ostringstream oss;
    oss << sc_core::sc_time_stamp() << " [Router SEND]";
    append_packet_debug_(oss, cmd);
    SC_REPORT_INFO(this->name(), oss.str().c_str());
  }

  void log_recv_wait_(std::uint32_t expected_hash) const {
    std::ostringstream oss;
    oss << sc_core::sc_time_stamp()
        << " [Router RECV wait] expected_hash=" << expected_hash
        << " pending_packets=" << pending_packets_.size();
    SC_REPORT_INFO(this->name(), oss.str().c_str());
  }

  void log_recv_matched_(const NocCmd<T> &cmd, bool from_pending) const {
    std::ostringstream oss;
    oss << sc_core::sc_time_stamp() << " [Router RECV matched]";
    append_packet_debug_(oss, cmd);
    oss << " source=" << (from_pending ? "pending" : "noc_fifo");
    SC_REPORT_INFO(this->name(), oss.str().c_str());
  }

  void log_recv_commit_(const NocCmd<T> &cmd) const {
    std::ostringstream oss;
    oss << sc_core::sc_time_stamp() << " [Router RECV COMMIT]"
        << " actual dst_buf=" << cmd.dst_buf << " dst_offset=" << cmd.dst_offset
        << " len=" << cmd.length << " hash=" << cmd_hash_u32(cmd);
    SC_REPORT_INFO(this->name(), oss.str().c_str());
  }

  // =========================================================================
  // RECV hash matching support
  // =========================================================================
  struct PendingPacket {
    NocCmd<T> cmd;
    std::vector<T> payload;
  };

  // Safety bound for packets that arrived before their NoC,IN,hash instruction.
  // If this fires, the compiler likely forgot to emit a matching receive hash,
  // or emitted receives in an order that keeps too many packets pending.
  static constexpr std::size_t MAX_PENDING_PACKETS = 4096;
  static constexpr std::size_t MAX_PENDING_ELEMS =
      (1u << 20); // elements, not bytes

  std::deque<PendingPacket> pending_packets_;
  std::size_t pending_elems_ = 0;

  static std::uint32_t hash_to_u32(const sc_dt::sc_uint<32> &h) {
    return static_cast<std::uint32_t>(h.to_uint());
  }

  static std::uint32_t hash_to_u32(std::uint32_t h) { return h; }

  std::uint32_t cmd_hash_u32(const NocCmd<T> &cmd) const {
    return hash_to_u32(cmd.hash);
  }

  bool pop_pending_by_hash(std::uint32_t expected_hash, PendingPacket &out) {
    for (auto it = pending_packets_.begin(); it != pending_packets_.end();
         ++it) {
      if (cmd_hash_u32(it->cmd) == expected_hash) {
        const std::size_t elems = it->payload.size();
        out = std::move(*it);
        pending_packets_.erase(it);
        pending_elems_ =
            (pending_elems_ >= elems) ? (pending_elems_ - elems) : 0;
        return true;
      }
    }
    return false;
  }

  void check_pending_capacity_or_fatal(const PendingPacket &pkt) const {
    const std::size_t after_packets = pending_packets_.size() + 1;
    const std::size_t after_elems = pending_elems_ + pkt.payload.size();

    if (after_packets > MAX_PENDING_PACKETS ||
        after_elems > MAX_PENDING_ELEMS) {
      std::ostringstream oss;
      oss << "Router pending packet overflow: packets=" << after_packets
          << " elems=" << after_elems
          << " limit_packets=" << MAX_PENDING_PACKETS
          << " limit_elems=" << MAX_PENDING_ELEMS
          << ". A NoC packet arrived before its NoC,IN,hash instruction, "
          << "but too many packets are now pending. Check compiler receive "
             "hashes/order.";
      SC_REPORT_FATAL(this->name(), oss.str().c_str());
    }
  }

  std::vector<T> drain_payload_to_vector_with_latency(const NocCmd<T> &cmd) {
    const uint16_t len = cmd.length.to_uint();
    std::vector<T> payload;
    payload.reserve(len);

    uint16_t i = 0;
    while (i < len) {
      // Same receive-side timing as do_recv(): one cycle per NOC_LANES burst.
      wait(clk.posedge_event());
      const int chunk = std::min<int>(NOC_LANES, static_cast<int>(len - i));
      for (int k = 0; k < chunk; ++k) {
        payload.push_back(Router_data_in.read());
        ++i;
      }
    }
    return payload;
  }

  void stash_unmatched_packet_from_noc(const NocCmd<T> &cmd,
                                       std::uint32_t expected_hash) {
    PendingPacket pkt;
    pkt.cmd = cmd;
    pkt.payload = drain_payload_to_vector_with_latency(cmd);

    for (const auto &old : pending_packets_) {
      if (cmd_hash_u32(old.cmd) == cmd_hash_u32(pkt.cmd)) {
        std::ostringstream oss;
        oss << "Router pending queue already contains hash="
            << cmd_hash_u32(pkt.cmd)
            << ". Duplicate hashes are legal only if the compiler "
               "intentionally "
            << "receives them in FIFO order on this destination core.";
        SC_REPORT_WARNING(this->name(), oss.str().c_str());
        break;
      }
    }

    check_pending_capacity_or_fatal(pkt);
    pending_elems_ += pkt.payload.size();
    pending_packets_.push_back(std::move(pkt));

    (void)expected_hash; // kept for debugger/watch expressions
  }

  void wait_recv_buf_grant_for(const NocCmd<T> &cmd) {
    recv_dst_buf.write(cmd.dst_buf);
    recv_dst_buf_valid.write(true);

    // Keep valid visible for at least one clock edge, matching the original
    // path.
    wait(clk.posedge_event());

    do {
      wait(clk.posedge_event());
    } while (!recv_buf_grant.read());

    recv_dst_buf_valid.write(false);
    if (!recv_buf_grant.read()) {
      SC_REPORT_FATAL(name(), "RECV: do_recv() called without recv_buf_grant!");
    }
  }

  void write_pending_payload_to_shadow(const PendingPacket &pkt) {
    const NocCmd<T> &cmd = pkt.cmd;
    const uint16_t len = cmd.length.to_uint();
    const uint16_t dstb = cmd.dst_buf.to_uint();
    const uint16_t base = cmd.dst_offset.to_uint();

    if (pkt.payload.size() != static_cast<std::size_t>(len)) {
      SC_REPORT_FATAL(
          this->name(),
          "Router pending payload size mismatch (payload.size != cmd.length)");
      return;
    }

    if (dstb >= bufs.size()) {
      SC_REPORT_ERROR(
          this->name(),
          "Router write_pending_payload_to_shadow(): invalid dst_buf index");
      return;
    }

    auto *dst_vec = bufs[dstb];
    auto *shadow_vec = shadow_bufs[dstb];

    if (!dst_vec) {
      SC_REPORT_ERROR(
          this->name(),
          "Router write_pending_payload_to_shadow(): dst_vec is nullptr");
      return;
    }

    const std::size_t vec_size = dst_vec->size();
    const std::size_t cfg_size = cfg_buf_size(dstb);
    const std::size_t base_z = base;
    const std::size_t len_z = len;

    if (vec_size != cfg_size) {
      SC_REPORT_ERROR(this->name(), "Router write_pending_payload_to_shadow(): "
                                    "dst_vec->size() != config");
      return;
    }

    if (base_z + len_z > cfg_size) {
      SC_REPORT_ERROR(
          this->name(),
          "Router write_pending_payload_to_shadow(): write out of range");
      return;
    }

    // No shadow -> direct write fallback.  No wait here: the receive latency
    // was already paid when the packet was drained into pending_packets_.
    if (!shadow_vec) {
      SC_REPORT_ERROR(this->name(), "Router write_pending_payload_to_shadow(): "
                                    "shadow is nullptr, direct write");
      for (uint16_t i = 0; i < len; ++i) {
        (*dst_vec)[base + i].write(pkt.payload[i]);
      }
      return;
    }

    if (shadow_vec->size() < cfg_size) {
      SC_REPORT_ERROR(this->name(), "Router write_pending_payload_to_shadow(): "
                                    "shadow smaller than config, resizing");
      shadow_vec->resize(cfg_size, T(0));
    }

    for (uint16_t i = 0; i < len; ++i) {
      (*shadow_vec)[base + i] = pkt.payload[i];
    }
  }

  // =========================================================================
  // SEND payload with cycle cost:
  //   - Each cycle inject up to NOC_LANES elements
  //   - If FIFO full, stop for this cycle and retry next cycle
  //   => If always writable: cycles = ceil(len / NOC_LANES)
  // =========================================================================
  void send_payload_chunked(sc_core::sc_vector<sc_core::sc_signal<T>> *src_vec,
                            uint16_t base, uint16_t len) {
    uint16_t i = 0;
    while (i < len) {
      // 1 cycle per burst (up to NOC_LANES flits)
      wait(clk.posedge_event());

      int sent = 0;
      while (sent < NOC_LANES && i < len) {
        T v = (*src_vec)[base + i].read();

        // non-blocking write: if can't write, count as backpressure (stall to
        // next cycle)
        if (Router_data_out.nb_write(v)) {
          ++i;
          ++sent;
        } else {
          break; // FIFO full -> waste remaining lanes this cycle, retry next
                 // cycle
        }
      }
    }
  }

  // =========================================================================
  // SEND FSM
  // =========================================================================
  void send_fsm() {
    Router_send_busy.write(false);
    Router_send_done.write(false);

    wait(); // avoid time0 glitches

    while (true) {
      do {
        wait();
      } while (!start.read() || !is_send.read());

      Router_send_busy.write(true);
      Router_send_done.write(false);

      if (is_mcast.read()) {
        do_send_multicast_native();
      } else {
        do_send_unicast();
      }

      Router_send_busy.write(false);

      wait();
      Router_send_done.write(true);
      wait();
      Router_send_done.write(false);
    }
  }

  // =========================================================================
  // RECV FSM
  // =========================================================================
  void recv_fsm() {
    Router_recv_busy.write(false);
    Router_recv_done.write(false);
    recv_dst_buf_valid.write(false);
    recv_dst_buf.write(0);

    wait(); // avoid time0 glitches

    while (true) {
      do {
        wait();
      } while (!start.read() || is_send.read());

      Router_recv_busy.write(true);
      Router_recv_done.write(false);

      const std::uint32_t expected_hash = hash_to_u32(hash.read());
      log_recv_wait_(expected_hash);

      // Find the packet requested by this NoC,IN,hash instruction.
      // Case A: packet arrived early -> it may already be in pending_packets_.
      // Case B: packet has not arrived yet -> keep reading NoC arrivals until
      //         the expected hash appears. Earlier non-matching arrivals are
      //         drained once into pending_packets_ so they cannot block the dst
      //         NoC FIFO forever.
      PendingPacket pending_pkt;
      NocCmd<T> matched_cmd;
      bool matched_from_pending = false;

      while (true) {
        if (pop_pending_by_hash(expected_hash, pending_pkt)) {
          matched_cmd = pending_pkt.cmd;
          matched_from_pending = true;
          break;
        }

        NocCmd<T> arrived_cmd = Router_cmd_in.read();
        const std::uint32_t arrived_hash = cmd_hash_u32(arrived_cmd);

        if (arrived_hash == expected_hash) {
          matched_cmd = arrived_cmd;
          matched_from_pending = false;
          break;
        }

        // This packet arrived before its corresponding NoC,IN,hash instruction.
        // Drain it from Router_data_in with the normal NOC_LANES timing and
        // store it locally.  When its NoC,IN,hash appears later, committing it
        // to shadow_buf must be zero-wait to avoid double-counting latency.
        stash_unmatched_packet_from_noc(arrived_cmd, expected_hash);
      }

      log_recv_matched_(matched_cmd, matched_from_pending);

      // Tell Core_Controller which dst_buf this matched packet will write.
      // The scoreboard/grant latency is kept identical to the old path.
      wait_recv_buf_grant_for(matched_cmd);

      if (matched_from_pending) {
        // The network/receive drain latency was already paid when the packet
        // was stashed.  Commit to shadow immediately after the grant.
        write_pending_payload_to_shadow(pending_pkt);
      } else {
        // Normal path: the expected packet arrives after/while this RECV is
        // active, so receive it with the original NOC_LANES timing.
        do_recv(matched_cmd);
      }
      log_recv_commit_(matched_cmd);

      Router_recv_busy.write(false);

      wait();
      Router_recv_done.write(true);
      wait();
      Router_recv_done.write(false);

      recv_dst_buf_valid.write(false);
    }
  }

  // =========================================================================
  // SEND unicast: local SRAM -> NoC
  // =========================================================================
  void do_send_unicast() {
    NocCmd<T> cmd;
    cmd.src_x = src_x.read();
    cmd.src_y = src_y.read();
    cmd.dst_x = dst_x.read();
    cmd.dst_y = dst_y.read();
    cmd.src_buf = src_buf.read();
    cmd.dst_buf = dst_buf.read();
    cmd.src_offset = src_offset.read();
    cmd.dst_offset = dst_offset.read();
    cmd.length = length.read();

    cmd.is_mcast = false;
    cmd.mcast_num = 0;
    cmd.hash = hash.read();

    const uint16_t srcb = cmd.src_buf.to_uint();
    const uint16_t dstb = cmd.dst_buf.to_uint();
    const uint16_t len = cmd.length.to_uint();
    const uint16_t base = cmd.src_offset.to_uint();     // idx -> offset
    const uint16_t dst_base = cmd.dst_offset.to_uint(); // idx -> offset

    if (srcb >= bufs.size()) {
      SC_REPORT_ERROR(this->name(),
                      "Router do_send_unicast(): invalid src_buf index");
      return;
    }
    if (dstb >= bufs.size()) {
      SC_REPORT_ERROR(this->name(),
                      "Router do_send_unicast(): invalid dst_buf index");
      return;
    }
    auto *src_vec = bufs[srcb];
    if (!src_vec) {
      SC_REPORT_ERROR(
          this->name(),
          "Router do_send_unicast(): src_vec is nullptr (buf not bound)");
      return;
    }

    {
      const std::size_t vec_size = src_vec->size();
      const std::size_t cfg_size = cfg_buf_size(srcb);
      const std::size_t base_z = base;
      const std::size_t len_z = len;

      if (vec_size != cfg_size) {
        SC_REPORT_ERROR(this->name(),
                        "Router do_send_unicast(): src_vec->size() != "
                        "ActiveHWConfig::BUF_SIZES[src_buf]");
        return;
      }

      if (base_z + len_z > cfg_size) {
        SC_REPORT_ERROR(this->name(),
                        "Router do_send_unicast(): read out of range "
                        "(src_offset+len > configured src buf size)");
        return;
      }
    }
    {
      const std::size_t cfg_size = cfg_buf_size(dstb);
      const std::size_t dst_base_z = dst_base;
      const std::size_t len_z = len;

      if (dst_base_z + len_z > cfg_size) {
        SC_REPORT_ERROR(this->name(),
                        "Router do_send_unicast(): write out of range "
                        "(dst_offset+len > configured dst buf size)");
        return;
      }
    }

    log_router_send_(cmd);
    Router_cmd_out.write(cmd);
    send_payload_chunked(src_vec, base, len);
  }

  // =========================================================================
  // SEND multicast (native): local SRAM -> NoC
  //   - 只送 1 個 is_mcast cmd（含 dest list）
  //   - payload 只注入 1 份 len 元素
  //   - NoC 內部負責複製/投遞到各 dst
  // =========================================================================
  void do_send_multicast_native() {
    const uint16_t srcb = src_buf.read().to_uint();
    const uint16_t len = length.read().to_uint();
    const uint16_t base = src_offset.read().to_uint(); //  idx -> offset

    if (srcb >= bufs.size()) {
      SC_REPORT_ERROR(
          this->name(),
          "Router do_send_multicast_native(): invalid src_buf index");
      return;
    }
    auto *src_vec = bufs[srcb];
    if (!src_vec) {
      SC_REPORT_ERROR(this->name(), "Router do_send_multicast_native(): "
                                    "src_vec is nullptr (buf not bound)");
      return;
    }

    {
      const std::size_t vec_size = src_vec->size();
      const std::size_t cfg_size = cfg_buf_size(srcb);
      const std::size_t base_z = base;
      const std::size_t len_z = len;

      if (vec_size != cfg_size) {
        SC_REPORT_ERROR(this->name(),
                        "Router do_send_multicast_native(): src_vec->size() != "
                        "ActiveHWConfig::BUF_SIZES[src_buf]");
        return;
      }

      if (base_z + len_z > cfg_size) {
        SC_REPORT_ERROR(this->name(),
                        "Router do_send_multicast_native(): read out of range "
                        "(src_offset+len > configured src buf size)");
        return;
      }
    }

    const int num = static_cast<int>(mcast_num.read().to_uint());
    if (num <= 0 || num > MAX_MCAST_DESTS) {
      SC_REPORT_ERROR(this->name(),
                      "Router do_send_multicast_native(): invalid mcast_num");
      return;
    }

    // build one multicast cmd
    NocCmd<T> mc;
    mc.src_x = src_x.read();
    mc.src_y = src_y.read();
    mc.src_buf = src_buf.read();
    mc.src_offset = src_offset.read(); //  idx -> offset
    mc.length = length.read();

    // optional: keep legacy dst_* fields meaningful (use dest[0]) for debug
    mc.dst_x = mcast_dst_x[0].read();
    mc.dst_y = mcast_dst_y[0].read();
    mc.dst_buf = mcast_dst_buf[0].read();
    mc.dst_offset = mcast_dst_offset[0].read(); //  idx -> offset
    mc.hash = hash.read();

    mc.is_mcast = true;
    mc.mcast_num = static_cast<sc_dt::sc_uint<7>>(num);

    // fill dest list into cmd
    for (int di = 0; di < num; ++di) {
      const uint16_t dstb_di = mcast_dst_buf[di].read().to_uint();
      const uint16_t dst_base_di = mcast_dst_offset[di].read().to_uint();

      if (dstb_di >= bufs.size()) {
        SC_REPORT_ERROR(this->name(), "Router do_send_multicast_native(): "
                                      "invalid multicast dst_buf index");
        return;
      }

      if (static_cast<std::size_t>(dst_base_di) +
              static_cast<std::size_t>(len) >
          cfg_buf_size(dstb_di)) {
        SC_REPORT_ERROR(this->name(),
                        "Router do_send_multicast_native(): write out of range "
                        "(mcast dst_offset+len > configured dst buf size)");
        return;
      }

      mc.mcast_dsts[di].dst_x = mcast_dst_x[di].read();
      mc.mcast_dsts[di].dst_y = mcast_dst_y[di].read();
      mc.mcast_dsts[di].dst_buf = mcast_dst_buf[di].read();
      mc.mcast_dsts[di].dst_offset = mcast_dst_offset[di].read();
    }

    // inject multicast cmd once
    Router_cmd_out.write(mc);

    // inject payload once
    send_payload_chunked(src_vec, base, len);
  }
  // =========================================================================
  // RECV: NoC -> local SRAM (write shadow, Tcore commits)
  // =========================================================================
  void do_recv(const NocCmd<T> &cmd) {
    const uint16_t len = cmd.length.to_uint();
    const uint16_t dstb = cmd.dst_buf.to_uint();
    const uint16_t base = cmd.dst_offset.to_uint();

    if (dstb >= bufs.size()) {
      SC_REPORT_ERROR(this->name(), "Router do_recv(): invalid dst_buf index");
      for (uint16_t i = 0; i < len; ++i)
        (void)Router_data_in.read();
      return;
    }

    auto *dst_vec = bufs[dstb];
    auto *shadow_vec = shadow_bufs[dstb];

    if (!dst_vec) {
      SC_REPORT_ERROR(this->name(),
                      "Router do_recv(): dst_vec is nullptr (buf not bound)");
      for (uint16_t i = 0; i < len; ++i)
        (void)Router_data_in.read();
      return;
    }

    const std::size_t vec_size = dst_vec->size();
    const std::size_t cfg_size = cfg_buf_size(dstb);
    const std::size_t base_z = base;
    const std::size_t len_z = len;

    if (vec_size != cfg_size) {
      SC_REPORT_ERROR(this->name(), "Router do_recv(): dst_vec->size() != "
                                    "ActiveHWConfig::BUF_SIZES[dst_buf]");
      for (uint16_t i = 0; i < len; ++i)
        (void)Router_data_in.read();
      return;
    }

    if (base_z + len_z > cfg_size) {
      SC_REPORT_ERROR(this->name(),
                      "Router do_recv(): write out of range "
                      "(dst_offset+len > configured dst buf size)");
      for (uint16_t i = 0; i < len; ++i)
        (void)Router_data_in.read();
      return;
    }

    // no shadow -> direct write (fallback)
    if (!shadow_vec) {
      SC_REPORT_ERROR(this->name(), "Router do_recv(): shadow_bufs[dstb] is "
                                    "nullptr, write direct to sc_signal");
      uint16_t i = 0;
      while (i < len) {
        wait(clk.posedge_event());
        const int chunk = std::min<int>(NOC_LANES, (int)len - (int)i);
        for (int k = 0; k < chunk; ++k) {
          T v = Router_data_in.read();
          (*dst_vec)[base + i].write(v);
          ++i;
        }
      }
      return;
    }

    if (shadow_vec->size() < cfg_size) {
      SC_REPORT_ERROR(
          this->name(),
          "Router do_recv(): shadow smaller than configured buffer, resizing");
      shadow_vec->resize(cfg_size, T(0));
    }

    uint16_t i = 0;
    while (i < len) {
      // flit delay starts here (per cycle up to NOC_LANES)
      wait(clk.posedge_event());
      const int chunk = std::min<int>(NOC_LANES, (int)len - (int)i);
      for (int k = 0; k < chunk; ++k) {
        T v = Router_data_in.read();
        (*shadow_vec)[base + i] = v;
        ++i;
      }
    }
  }
};
