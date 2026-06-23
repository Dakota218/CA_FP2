#!/usr/bin/env python3
"""
TA-side evaluator for hardware submissions.

This version is designed to encourage thoughtful trade-offs:
- Reward real performance improvement
- Penalize blind over-building of hardware
- Give credit for saving cost, but only if performance is not hurt too much
- Keep NoC as a special cost term, but avoid overweighting it too much

Default policy in this version:
- total_hardware_cost_multiplier = 50% area + 5% SRAM BW + 5% NoC BW
  + 25% DRAM BW + 15% power
- noc_cost_multiplier = noc_bw_scale ** 1.35
- cost-saving bonus is gated by revenue_scale, so students cannot win
  by blindly shrinking hardware

Example:
python /home/wilson/SystemC/TA/FP1/TA_eval_profit.py \
  --baseline-config /home/wilson/SystemC/include/baseline_hw.h \
  --student-config /home/wilson/SystemC/include/student_hw.h \
  --baseline-log <baseline_log> \
  --candidate-log <candidate_log> \
  --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class HWConfig:
    array_macs_per_cycle: int
    dram_lanes: int
    sram_dma_width: int
    noc_link_width: int
    vector_lanes: int
    num_bufs: int
    buf_sizes: list[int]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def extract_int(config_block: str, name: str) -> int:
    pattern = rf"\b{re.escape(name)}\b\s*=\s*([0-9]+)\s*;"
    m = re.search(pattern, config_block)
    if not m:
        raise ValueError(f"Cannot find {name} in struct Config.")
    return int(m.group(1))


def extract_buf_sizes(config_block: str) -> list[int]:
    names = [
        "BUF_ARRAY_IN_SIZE",
        "BUF_ARRAY_OUT_SIZE",
        "BUF_VEC_IN_SIZE",
        "BUF_VEC_W_SIZE",
        "BUF_VEC_OUT_SIZE",
        "BUF_OTHER_IN_SIZE",
        "BUF_OTHER_OUT_SIZE",
        "BUF_K_SIZE",
        "BUF_V_SIZE",
        "BUF_Q_SV_SIZE",
        "BUF_RMS_SIZE",
        "BUF_CORE_CONTROLLER_SIZE",
    ]
    return [extract_int(config_block, name) for name in names]

def parse_hw_header(path: str | Path) -> HWConfig:
    text = Path(path).read_text(encoding="utf-8")
    config_block = extract_config_block(text, path)

    num_bufs = extract_int(config_block, "NUM_BUFS")
    buf_sizes = extract_buf_sizes(config_block)

    if len(buf_sizes) != num_bufs:
        raise ValueError(
            f"{path}: NUM_BUFS = {num_bufs}, but BUF_SIZES contains {len(buf_sizes)} elements."
        )

    return HWConfig(
        array_macs_per_cycle=extract_int(config_block, "ARRAY_MACS_PER_CYCLE"),
        dram_lanes=extract_int(config_block, "DRAM_LANES"),
        sram_dma_width=extract_int(config_block, "SRAM_DMA_WIDTH"),
        noc_link_width=extract_int(config_block, "NOC_LINK_WIDTH"),
        vector_lanes=extract_int(config_block, "VECTOR_LANES"),
        num_bufs=num_bufs,
        buf_sizes=buf_sizes,
    )

def extract_config_block(text: str, path: str | Path) -> str:
    """
    Extract only the body of `struct Config { ... };`
    so we do not accidentally parse values from comments,
    examples, or other unrelated text.
    """
    m = re.search(r"struct\s+Config\s*\{(.*?)\};", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Cannot find `struct Config {{ ... }};` in header: {path}")
    return m.group(1)

# ------------------------------------------------------------
# Normalized scales: student relative to baseline
# ------------------------------------------------------------
def array_scale(student: HWConfig, baseline: HWConfig) -> float:
    return safe_div(student.array_macs_per_cycle, baseline.array_macs_per_cycle)


def dram_bw_scale(student: HWConfig, baseline: HWConfig) -> float:
    return safe_div(student.dram_lanes, baseline.dram_lanes)


def sram_bw_scale(student: HWConfig, baseline: HWConfig) -> float:
    return safe_div(student.sram_dma_width, baseline.sram_dma_width)


def noc_bw_scale(student: HWConfig, baseline: HWConfig) -> float:
    return safe_div(student.noc_link_width, baseline.noc_link_width)


def vector_scale(student: HWConfig, baseline: HWConfig) -> float:
    return safe_div(student.vector_lanes, baseline.vector_lanes)


def sram_capacity_scale(student: HWConfig, baseline: HWConfig) -> float:
    return safe_div(sum(student.buf_sizes), sum(baseline.buf_sizes))


# ------------------------------------------------------------
# Hardware cost model
# ------------------------------------------------------------
def silicon_area_multiplier(student: HWConfig, baseline: HWConfig) -> float:
    # Slide mix:
    # 38% SRAM Capacity, 3% SRAM BW, 25% Array, 26% Vector, 8% NoC BW
    return (
        0.38 * sram_capacity_scale(student, baseline)
        + 0.03 * sram_bw_scale(student, baseline)
        + 0.25 * array_scale(student, baseline)
        + 0.26 * vector_scale(student, baseline)
        + 0.08 * noc_bw_scale(student, baseline)
    )


def sram_bw_cost_multiplier(student: HWConfig, baseline: HWConfig) -> float:
    return sram_bw_scale(student, baseline) ** 1.15


def noc_cost_multiplier(
    student: HWConfig,
    baseline: HWConfig,
    noc_cost_exponent: float = 1.35,
) -> float:
    return noc_bw_scale(student, baseline) ** noc_cost_exponent


def dram_cost_multiplier(student: HWConfig, baseline: HWConfig) -> float:
    return dram_bw_scale(student, baseline) ** 1.50


def system_power_multiplier(student: HWConfig, baseline: HWConfig) -> float:
    # Slide mix:
    # 30% DRAM BW, 5% NoC BW, 27% SRAM Capacity, 7% SRAM BW, 15% Array, 16% Vector
    return (
        0.30 * dram_bw_scale(student, baseline)
        + 0.05 * noc_bw_scale(student, baseline)
        + 0.27 * sram_capacity_scale(student, baseline)
        + 0.07 * sram_bw_scale(student, baseline)
        + 0.15 * array_scale(student, baseline)
        + 0.16 * vector_scale(student, baseline)
    )


def power_delivery_cost_multiplier(student: HWConfig, baseline: HWConfig) -> float:
    return system_power_multiplier(student, baseline) ** 1.20


def total_hardware_cost_multiplier(
    student: HWConfig,
    baseline: HWConfig,
    area_weight: float = 0.50,
    sram_bw_weight: float = 0.05,
    noc_weight: float = 0.05,
    dram_weight: float = 0.25,
    power_weight: float = 0.15,
    noc_cost_exponent: float = 1.35,
) -> float:
    return (
        area_weight * silicon_area_multiplier(student, baseline)
        + sram_bw_weight * sram_bw_cost_multiplier(student, baseline)
        + noc_weight * noc_cost_multiplier(student, baseline, noc_cost_exponent)
        + dram_weight * dram_cost_multiplier(student, baseline)
        + power_weight * power_delivery_cost_multiplier(student, baseline)
    )


def extra_cost_ratio(
    student: HWConfig,
    baseline: HWConfig,
    area_weight: float = 0.50,
    sram_bw_weight: float = 0.05,
    noc_weight: float = 0.05,
    dram_weight: float = 0.25,
    power_weight: float = 0.15,
    noc_cost_exponent: float = 1.35,
) -> float:
    return (
        total_hardware_cost_multiplier(
            student,
            baseline,
            area_weight=area_weight,
            sram_bw_weight=sram_bw_weight,
            noc_weight=noc_weight,
            dram_weight=dram_weight,
            power_weight=power_weight,
            noc_cost_exponent=noc_cost_exponent,
        )
        - 1.0
    )


# ------------------------------------------------------------
# Grading-side cost adjustment
# ------------------------------------------------------------
def grading_cost_penalty_ratio(
    student: HWConfig,
    baseline: HWConfig,
    area_weight: float = 0.50,
    sram_bw_weight: float = 0.05,
    noc_weight: float = 0.05,
    dram_weight: float = 0.25,
    power_weight: float = 0.15,
    noc_cost_exponent: float = 1.35,
) -> float:
    """Penalty for overspending hardware cost.

    If extra_cost_ratio > 0, apply a soft-but-nonlinear penalty.
    """
    extra = max(
        0.0,
        extra_cost_ratio(
            student,
            baseline,
            area_weight=area_weight,
            sram_bw_weight=sram_bw_weight,
            noc_weight=noc_weight,
            dram_weight=dram_weight,
            power_weight=power_weight,
            noc_cost_exponent=noc_cost_exponent,
        ),
    )
    return 0.35 * extra + 0.65 * extra * extra
    # return 0.35 * math.log1p(extra) + 0.25 * extra
    # return 0.45 * extra + 0.12 * (extra ** 1.5)


def cost_saving_gate_from_revenue_scale(
    revenue_scale: float,
    gate_floor: float = 0.90,
    gate_full: float = 1.00,
) -> float:
    """Gate cost-saving bonus by performance.

    revenue_scale <= gate_floor -> 0 bonus
    revenue_scale >= gate_full  -> full bonus
    linear interpolation in between
    """
    if gate_full <= gate_floor:
        raise ValueError("gate_full must be greater than gate_floor")
    return clamp((revenue_scale - gate_floor) / (gate_full - gate_floor), 0.0, 1.0)


def grading_cost_saving_bonus_ratio(
    revenue_scale: float,
    student: HWConfig,
    baseline: HWConfig,
    saving_bonus_weight: float = 0.40,
    gate_floor: float = 0.90,
    gate_full: float = 1.00,
    area_weight: float = 0.50,
    sram_bw_weight: float = 0.05,
    noc_weight: float = 0.05,
    dram_weight: float = 0.25,
    power_weight: float = 0.15,
    noc_cost_exponent: float = 1.35,
) -> float:
    """Bonus for saving cost, but only when performance is not hurt too much."""
    saving = max(
        0.0,
        -extra_cost_ratio(
            student,
            baseline,
            area_weight=area_weight,
            sram_bw_weight=sram_bw_weight,
            noc_weight=noc_weight,
            dram_weight=dram_weight,
            power_weight=power_weight,
            noc_cost_exponent=noc_cost_exponent,
        ),
    )
    gate = cost_saving_gate_from_revenue_scale(
        revenue_scale,
        gate_floor=gate_floor,
        gate_full=gate_full,
    )
    return saving_bonus_weight * saving * gate


def grading_cost_adjustment_ratio(
    revenue_scale: float,
    student: HWConfig,
    baseline: HWConfig,
    saving_bonus_weight: float = 0.40,
    gate_floor: float = 0.90,
    gate_full: float = 1.00,
    area_weight: float = 0.50,
    sram_bw_weight: float = 0.05,
    noc_weight: float = 0.05,
    dram_weight: float = 0.25,
    power_weight: float = 0.15,
    noc_cost_exponent: float = 1.35,
) -> float:
    penalty = grading_cost_penalty_ratio(
        student,
        baseline,
        area_weight=area_weight,
        sram_bw_weight=sram_bw_weight,
        noc_weight=noc_weight,
        dram_weight=dram_weight,
        power_weight=power_weight,
        noc_cost_exponent=noc_cost_exponent,
    )
    bonus = grading_cost_saving_bonus_ratio(
        revenue_scale,
        student,
        baseline,
        saving_bonus_weight=saving_bonus_weight,
        gate_floor=gate_floor,
        gate_full=gate_full,
        area_weight=area_weight,
        sram_bw_weight=sram_bw_weight,
        noc_weight=noc_weight,
        dram_weight=dram_weight,
        power_weight=power_weight,
        noc_cost_exponent=noc_cost_exponent,
    )
    return penalty - bonus


# ------------------------------------------------------------
# Gentle power throttling
# ------------------------------------------------------------
def clock_scale_after_throttle(
    student: HWConfig,
    baseline: HWConfig,
    power_budget: float = 1.50,
) -> float:
    p = system_power_multiplier(student, baseline)
    if p <= power_budget:
        return 1.0
    return math.sqrt(power_budget / p)


def effective_perf(
    student: HWConfig,
    baseline: HWConfig,
    raw_perf: float,
    power_budget: float = 1.50,
    perf_already_throttled: bool = False,
) -> float:
    if perf_already_throttled:
        return raw_perf
    return raw_perf * clock_scale_after_throttle(student, baseline, power_budget)


def revenue_scale_from_perf(
    student: HWConfig,
    baseline_hw: HWConfig,
    baseline_perf: float,
    candidate_perf: float,
    power_budget: float = 1.50,
    perf_already_throttled: bool = False,
) -> float:
    effective_candidate = effective_perf(
        student,
        baseline_hw,
        candidate_perf,
        power_budget=power_budget,
        perf_already_throttled=perf_already_throttled,
    )
    return safe_div(effective_candidate, baseline_perf)


def profit_score(
    revenue_scale: float,
    student: HWConfig,
    baseline: HWConfig,
    perf_weight: float = 100.0,
    cost_weight: float = 45.0,
    saving_bonus_weight: float = 0.40,
    gate_floor: float = 0.90,
    gate_full: float = 1.00,
    area_weight: float = 0.50,
    sram_bw_weight: float = 0.05,
    noc_weight: float = 0.05,
    dram_weight: float = 0.25,
    power_weight: float = 0.15,
    noc_cost_exponent: float = 1.35,
) -> float:
    perf_gain = revenue_scale - 1.0
    reward = perf_weight * perf_gain
    cost_adjustment = grading_cost_adjustment_ratio(
        revenue_scale,
        student,
        baseline,
        saving_bonus_weight=saving_bonus_weight,
        gate_floor=gate_floor,
        gate_full=gate_full,
        area_weight=area_weight,
        sram_bw_weight=sram_bw_weight,
        noc_weight=noc_weight,
        dram_weight=dram_weight,
        power_weight=power_weight,
        noc_cost_exponent=noc_cost_exponent,
    )
    return reward - cost_weight * cost_adjustment


# ------------------------------------------------------------
# Performance extraction from simulation summary
# We now use Total cycles from NocBusyMeter.
# Smaller cycles = better performance.
# Internally we convert cycles to pseudo-throughput = 1 / cycles
# so the rest of the grading pipeline can stay almost unchanged.
# ------------------------------------------------------------
TOTAL_CYCLES_PATTERNS = [
    r"Info:\s*NocBusyMeter:\s*[\s\S]*?NoC\s+Busy\s+Utilization\s*\(up\s+to\s+all_done\)\s*[\s\S]*?Total\s+cycles:\s*([0-9]+(?:\.[0-9]+)?)",
    r"NoC\s+Busy\s+Utilization\s*\(up\s+to\s+all_done\)\s*[\s\S]*?Total\s+cycles:\s*([0-9]+(?:\.[0-9]+)?)",
]


def _find_last_match(text: str, pattern: str) -> Optional[float]:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    last = matches[-1]
    if isinstance(last, tuple):
        last = last[0]
    return float(last)


def extract_total_cycles_from_log(log_path: str | Path, regex: Optional[str] = None) -> float:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")

    if regex is not None:
        value = _find_last_match(text, regex)
        if value is None:
            raise ValueError(
                f"Cannot find total cycles in {log_path} using custom regex: {regex}"
            )
        return value

    for pattern in TOTAL_CYCLES_PATTERNS:
        value = _find_last_match(text, pattern)
        if value is not None:
            return value

    raise ValueError(
        "Cannot find 'Total cycles' in log automatically. "
        "Please pass --cycles-regex with a regex containing one capture group."
    )


def resolve_total_cycles(
    value: Optional[float], log_path: Optional[str], regex: Optional[str]
) -> float:
    if value is not None:
        return value
    if log_path is not None:
        return extract_total_cycles_from_log(log_path, regex)
    raise ValueError("Total cycles must be provided either by a number or by a log file.")


def cycles_to_perf(total_cycles: float) -> float:
    if total_cycles <= 0:
        raise ValueError(f"Total cycles must be > 0, but got {total_cycles}")
    return 1.0 / total_cycles
# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TA-side evaluator: compare baseline_hw.h and student_hw.h, then compute "
            "cost, power, throttling, and profit with trade-off-aware grading."
        )
    )
    parser.add_argument("--baseline-config", required=True, help="Path to baseline_hw.h")
    parser.add_argument("--student-config", required=True, help="Path to student_hw.h")

    parser.add_argument("--baseline-cycles", type=float, default=None)
    parser.add_argument("--candidate-cycles", type=float, default=None)
    parser.add_argument("--baseline-log", default=None)
    parser.add_argument("--candidate-log", default=None)
    parser.add_argument(
        "--cycles-regex",
        default=None,
        help="Optional regex with one capture group for total cycle extraction",
    )

    parser.add_argument("--power-budget", type=float, default=1.50)
    parser.add_argument("--perf-weight", type=float, default=100.0)
    parser.add_argument("--cost-weight", type=float, default=45.0)

    # Cost model knobs
    parser.add_argument("--area-weight", type=float, default=0.50)
    parser.add_argument("--sram-bw-weight", type=float, default=0.05)
    parser.add_argument("--noc-weight", type=float, default=0.05)
    parser.add_argument("--dram-weight", type=float, default=0.25)
    parser.add_argument("--power-weight", type=float, default=0.15)
    parser.add_argument("--noc-cost-exponent", type=float, default=1.35)

    # Cost-saving bonus knobs
    parser.add_argument("--saving-bonus-weight", type=float, default=0.40)
    parser.add_argument("--saving-gate-floor", type=float, default=0.90)
    parser.add_argument("--saving-gate-full", type=float, default=1.00)

    parser.add_argument(
        "--perf-already-throttled",
        action="store_true",
        help="Use this only if the input performance metric (derived from total cycles) already reflects the TA-side throttling rule.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text summary.",
    )
    return parser


def validate_weights(args: argparse.Namespace) -> None:
    total = (
        args.area_weight
        + args.sram_bw_weight
        + args.noc_weight
        + args.dram_weight
        + args.power_weight
    )
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Cost weights must sum to 1.0, but got {total:.12f}.")
    if args.saving_gate_full <= args.saving_gate_floor:
        raise ValueError("--saving-gate-full must be greater than --saving-gate-floor")


def main() -> None:
    args = build_argparser().parse_args()
    validate_weights(args)

    baseline_hw = parse_hw_header(args.baseline_config)
    student_hw = parse_hw_header(args.student_config)

    baseline_cycles = resolve_total_cycles(
        args.baseline_cycles, args.baseline_log, args.cycles_regex
    )
    candidate_cycles = resolve_total_cycles(
        args.candidate_cycles, args.candidate_log, args.cycles_regex
    )

    # Convert cycles to a monotonic performance metric so the rest of
    # the pipeline can still use the existing throughput-style formulas.
    baseline_perf = cycles_to_perf(baseline_cycles)
    candidate_perf = cycles_to_perf(candidate_cycles)
    clk_scale = clock_scale_after_throttle(student_hw, baseline_hw, args.power_budget)
    effective_candidate_perf = effective_perf(
        student_hw,
        baseline_hw,
        candidate_perf,
        power_budget=args.power_budget,
        perf_already_throttled=args.perf_already_throttled,
    )

    raw_perf_ratio = safe_div(candidate_perf, baseline_perf)

    revenue_scale = revenue_scale_from_perf(
        student_hw,
        baseline_hw,
        baseline_perf,
        candidate_perf,
        power_budget=args.power_budget,
        perf_already_throttled=args.perf_already_throttled,
    )

    total_cost = total_hardware_cost_multiplier(
        student_hw,
        baseline_hw,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    extra_cost = extra_cost_ratio(
        student_hw,
        baseline_hw,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    grading_penalty = grading_cost_penalty_ratio(
        student_hw,
        baseline_hw,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    saving_gate = cost_saving_gate_from_revenue_scale(
        revenue_scale,
        gate_floor=args.saving_gate_floor,
        gate_full=args.saving_gate_full,
    )
    saving_bonus = grading_cost_saving_bonus_ratio(
        revenue_scale,
        student_hw,
        baseline_hw,
        saving_bonus_weight=args.saving_bonus_weight,
        gate_floor=args.saving_gate_floor,
        gate_full=args.saving_gate_full,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    cost_adjustment = grading_cost_adjustment_ratio(
        revenue_scale,
        student_hw,
        baseline_hw,
        saving_bonus_weight=args.saving_bonus_weight,
        gate_floor=args.saving_gate_floor,
        gate_full=args.saving_gate_full,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    profit = profit_score(
        revenue_scale,
        student_hw,
        baseline_hw,
        perf_weight=args.perf_weight,
        cost_weight=args.cost_weight,
        saving_bonus_weight=args.saving_bonus_weight,
        gate_floor=args.saving_gate_floor,
        gate_full=args.saving_gate_full,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )

    result = {
        "baseline_config_path": str(args.baseline_config),
        "student_config_path": str(args.student_config),
        "baseline_config": asdict(baseline_hw),
        "student_config": asdict(student_hw),
        "baseline_total_cycles": baseline_cycles,
        "candidate_total_cycles": candidate_cycles,
        "baseline_perf_from_cycles": baseline_perf,
        "candidate_perf_from_cycles_raw": candidate_perf,
        "candidate_perf_from_cycles_effective": effective_candidate_perf,
        "raw_perf_ratio": raw_perf_ratio,
        "Performance_scale": revenue_scale,
        "array_scale": array_scale(student_hw, baseline_hw),
        "dram_bw_scale": dram_bw_scale(student_hw, baseline_hw),
        "sram_bw_scale": sram_bw_scale(student_hw, baseline_hw),
        "noc_bw_scale": noc_bw_scale(student_hw, baseline_hw),
        "vector_scale": vector_scale(student_hw, baseline_hw),
        "sram_capacity_scale": sram_capacity_scale(student_hw, baseline_hw),
        "silicon_area_multiplier": silicon_area_multiplier(student_hw, baseline_hw),
        "sram_bw_cost_multiplier": sram_bw_cost_multiplier(student_hw, baseline_hw),
        "noc_cost_multiplier": noc_cost_multiplier(
            student_hw, baseline_hw, args.noc_cost_exponent
        ),
        "dram_cost_multiplier": dram_cost_multiplier(student_hw, baseline_hw),
        "system_power_multiplier": system_power_multiplier(student_hw, baseline_hw),
        "power_delivery_cost_multiplier": power_delivery_cost_multiplier(student_hw, baseline_hw),
        "power_budget": args.power_budget,
        "clock_scale_after_throttle": 1.0 if args.perf_already_throttled  else clk_scale,
        "cost_weights": {
            "area_weight": args.area_weight,
            "sram_bw_weight": args.sram_bw_weight,
            "noc_weight": args.noc_weight,
            "dram_weight": args.dram_weight,
            "power_weight": args.power_weight,
            "sum": args.area_weight
            + args.sram_bw_weight
            + args.noc_weight
            + args.dram_weight
            + args.power_weight,
        },
        "noc_cost_exponent": args.noc_cost_exponent,
        "total_hardware_cost_multiplier": total_cost,
        "extra_cost_ratio": extra_cost,
        "grading_cost_penalty_ratio": grading_penalty,
        "saving_bonus_weight": args.saving_bonus_weight,
        "saving_gate_floor": args.saving_gate_floor,
        "saving_gate_full": args.saving_gate_full,
        "cost_saving_gate": saving_gate,
        "grading_cost_saving_bonus_ratio": saving_bonus,
        "grading_cost_adjustment_ratio": cost_adjustment,
        "perf_weight": args.perf_weight,
        "cost_weight": args.cost_weight,
        "profit_score": profit,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("=== TA Evaluation Summary ===")
    print(f"baseline_config                       : {args.baseline_config}")
    print(f"student_config                        : {args.student_config}")
    print(f"baseline_total_cycles                 : {baseline_cycles:.6f}")
    print(f"candidate_total_cycles                : {candidate_cycles:.6f}")
    print(f"baseline_perf_from_cycles             : {baseline_perf:.12f}")
    print(f"candidate_perf_from_cycles_raw        : {candidate_perf:.12f}")
    print(f"candidate_perf_from_cycles_effective  : {effective_candidate_perf:.12f}")
    print(f"raw_perf_ratio                        : {raw_perf_ratio:.6f}")
    print(f"Performance_scale                     : {revenue_scale:.6f}")
    print(f"clock_scale_after_throttle            : {result['clock_scale_after_throttle']:.6f}")
    print(f"total_hardware_cost_multiplier        : {total_cost:.6f}")
    print(f"extra_cost_ratio                      : {extra_cost:.6f}")
    print(f"grading_cost_penalty_ratio            : {grading_penalty:.6f}")
    print(f"cost_saving_gate                      : {saving_gate:.6f}")
    print(f"grading_cost_saving_bonus_ratio       : {saving_bonus:.6f}")
    print(f"grading_cost_adjustment_ratio         : {cost_adjustment:.6f}")
    print(f"system_power_multiplier               : {result['system_power_multiplier']:.6f}")
    print(f"profit_score                          : {profit:.6f}")


if __name__ == "__main__":
    main()