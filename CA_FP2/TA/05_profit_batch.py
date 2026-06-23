#!/usr/bin/env python3
"""
Batch FP2 profit evaluation.

This script merges:
  - submission/naming status from fp2_submission_check.csv
  - correctness status from fp2_verify_correctness.csv
  - profit score computed by TA_eval_profit.py formulas

Only groups with:
  submitted=YES, status=OK, naming_error_count=0, correctness PASS
will receive a profit score.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SYSTEMC_DIR = Path(os.environ.get("SYSTEMC_DIR", str(REPO_ROOT / "SystemC")))
DEFAULT_FP2_ROOT = Path(os.environ.get("FP2_ROOT", str(REPO_ROOT / "FP2")))
DEFAULT_TA_DIR = Path(os.environ.get("TA_DIR", str(SCRIPT_DIR)))
DEFAULT_SUBMISSION_DIR = DEFAULT_FP2_ROOT / "submission"
if not DEFAULT_SUBMISSION_DIR.exists() and (DEFAULT_FP2_ROOT / "submission_example").exists():
    DEFAULT_SUBMISSION_DIR = DEFAULT_FP2_ROOT / "submission_example"
DEFAULT_SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", str(DEFAULT_SUBMISSION_DIR)))
DEFAULT_OUTCOME_DIR = Path(os.environ.get("OUTCOME_DIR", str(DEFAULT_FP2_ROOT / "outcome")))
DEFAULT_NAMING_CSV = DEFAULT_OUTCOME_DIR / "fp2_submission_check.csv"
DEFAULT_CORRECTNESS_CSV = DEFAULT_OUTCOME_DIR / "fp2_verify_correctness.csv"
DEFAULT_FINAL_CSV = DEFAULT_OUTCOME_DIR / "fp2_final_report.csv"
DEFAULT_BASELINE_CONFIG = DEFAULT_SYSTEMC_DIR / "include" / "baseline_hw.h"
DEFAULT_BASELINE_CYCLES = 315259.0


def load_profit_module(ta_dir: Path):
    profit_py = ta_dir / "TA_eval_profit.py"
    if not profit_py.is_file():
        raise FileNotFoundError(
            f"missing required profit evaluator: {profit_py}. "
            "05_profit_batch.py must be distributed together with TA_eval_profit.py."
        )

    sys.path.insert(0, str(ta_dir))
    import TA_eval_profit as profit  # noqa: PLC0415

    return profit


def read_csv_by_group(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing CSV: {path}")

    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group = row.get("group", "").strip()
            if group:
                rows[group] = row
    return rows


def group_sort_key(group: str) -> tuple[int, str]:
    if len(group) == 3 and group[0] == "G" and group[1:].isdigit():
        return int(group[1:]), group
    return 999, group


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def compute_profit_for_group(args, profit, group: str) -> dict[str, str]:
    group_dir = Path(args.submission_dir) / f"CA_FP2_{group}"
    student_config = group_dir / f"student_hw_{group}.h"
    sim_log = Path(args.outcome_dir) / group / "sim.log"

    baseline_hw = profit.parse_hw_header(args.baseline_config)
    student_hw = profit.parse_hw_header(student_config)

    baseline_cycles = float(args.baseline_cycles)
    candidate_cycles = profit.extract_total_cycles_from_log(sim_log, args.cycles_regex)

    baseline_perf = profit.cycles_to_perf(baseline_cycles)
    candidate_perf = profit.cycles_to_perf(candidate_cycles)
    effective_candidate_perf = profit.effective_perf(
        student_hw,
        baseline_hw,
        candidate_perf,
        power_budget=args.power_budget,
        perf_already_throttled=args.perf_already_throttled,
    )
    raw_perf_ratio = profit.safe_div(candidate_perf, baseline_perf)
    revenue_scale = profit.revenue_scale_from_perf(
        student_hw,
        baseline_hw,
        baseline_perf,
        candidate_perf,
        power_budget=args.power_budget,
        perf_already_throttled=args.perf_already_throttled,
    )

    total_cost = profit.total_hardware_cost_multiplier(
        student_hw,
        baseline_hw,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    extra_cost = profit.extra_cost_ratio(
        student_hw,
        baseline_hw,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    grading_penalty = profit.grading_cost_penalty_ratio(
        student_hw,
        baseline_hw,
        area_weight=args.area_weight,
        sram_bw_weight=args.sram_bw_weight,
        noc_weight=args.noc_weight,
        dram_weight=args.dram_weight,
        power_weight=args.power_weight,
        noc_cost_exponent=args.noc_cost_exponent,
    )
    saving_gate = profit.cost_saving_gate_from_revenue_scale(
        revenue_scale,
        gate_floor=args.saving_gate_floor,
        gate_full=args.saving_gate_full,
    )
    saving_bonus = profit.grading_cost_saving_bonus_ratio(
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
    cost_adjustment = profit.grading_cost_adjustment_ratio(
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
    profit_score = profit.profit_score(
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

    return {
        "student_config": str(student_config),
        "sim_log": str(sim_log),
        "baseline_total_cycles": fmt(baseline_cycles),
        "candidate_total_cycles": fmt(candidate_cycles),
        "raw_perf_ratio": fmt(raw_perf_ratio),
        "Performance_scale": fmt(revenue_scale),
        "candidate_perf_from_cycles_effective": fmt(effective_candidate_perf),
        "clock_scale_after_throttle": fmt(
            1.0
            if args.perf_already_throttled
            else profit.clock_scale_after_throttle(student_hw, baseline_hw, args.power_budget)
        ),
        "array_scale": fmt(profit.array_scale(student_hw, baseline_hw)),
        "dram_bw_scale": fmt(profit.dram_bw_scale(student_hw, baseline_hw)),
        "sram_bw_scale": fmt(profit.sram_bw_scale(student_hw, baseline_hw)),
        "noc_bw_scale": fmt(profit.noc_bw_scale(student_hw, baseline_hw)),
        "vector_scale": fmt(profit.vector_scale(student_hw, baseline_hw)),
        "sram_capacity_scale": fmt(profit.sram_capacity_scale(student_hw, baseline_hw)),
        "total_hardware_cost_multiplier": fmt(total_cost),
        "extra_cost_ratio": fmt(extra_cost),
        "grading_cost_penalty_ratio": fmt(grading_penalty),
        "cost_saving_gate": fmt(saving_gate),
        "grading_cost_saving_bonus_ratio": fmt(saving_bonus),
        "grading_cost_adjustment_ratio": fmt(cost_adjustment),
        "system_power_multiplier": fmt(profit.system_power_multiplier(student_hw, baseline_hw)),
        "profit_score": fmt(profit_score),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch compute FP2 profit score for correctness-passing groups."
    )
    parser.add_argument("--ta-dir", type=Path, default=DEFAULT_TA_DIR)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--outcome-dir", type=Path, default=DEFAULT_OUTCOME_DIR)
    parser.add_argument("--naming-csv", type=Path, default=DEFAULT_NAMING_CSV)
    parser.add_argument("--correctness-csv", type=Path, default=DEFAULT_CORRECTNESS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FINAL_CSV)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--baseline-cycles", type=float, default=DEFAULT_BASELINE_CYCLES)
    parser.add_argument("--cycles-regex", default=None)

    parser.add_argument("--power-budget", type=float, default=1.50)
    parser.add_argument("--perf-weight", type=float, default=100.0)
    parser.add_argument("--cost-weight", type=float, default=45.0)
    parser.add_argument("--area-weight", type=float, default=0.50)
    parser.add_argument("--sram-bw-weight", type=float, default=0.05)
    parser.add_argument("--noc-weight", type=float, default=0.05)
    parser.add_argument("--dram-weight", type=float, default=0.25)
    parser.add_argument("--power-weight", type=float, default=0.15)
    parser.add_argument("--noc-cost-exponent", type=float, default=1.35)
    parser.add_argument("--saving-bonus-weight", type=float, default=0.40)
    parser.add_argument("--saving-gate-floor", type=float, default=0.90)
    parser.add_argument("--saving-gate-full", type=float, default=1.00)
    parser.add_argument("--perf-already-throttled", action="store_true")
    return parser


def validate_args(args) -> None:
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


def main() -> int:
    args = build_argparser().parse_args()
    validate_args(args)
    profit = load_profit_module(args.ta_dir)

    naming_rows = read_csv_by_group(args.naming_csv)
    correctness_rows = read_csv_by_group(args.correctness_csv)

    fieldnames = [
        "group",
        "submitted",
        "naming_status",
        "naming_error_count",
        "weights_dir_ok",
        "memory_map_json_ok",
        "student_hw_ok",
        "core_csv_count",
        "core_csv_ok",
        "bin_count",
        "bin_suffix_ok",
        "correctness_status",
        "correctness_fail_count",
        "correctness_total",
        "correctness_max_abs",
        "correctness_rmse",
        "eligible_for_profit",
        "profit_status",
        "baseline_total_cycles",
        "candidate_total_cycles",
        "raw_perf_ratio",
        "Performance_scale",
        "clock_scale_after_throttle",
        "total_hardware_cost_multiplier",
        "extra_cost_ratio",
        "grading_cost_adjustment_ratio",
        "system_power_multiplier",
        "student_config",
        "sim_log",
        "message",
        "profit_score",
    ]

    rows = []
    pass_profit = 0
    groups = sorted(naming_rows, key=group_sort_key)
    for group in groups:
        naming = naming_rows[group]
        correctness = correctness_rows.get(group, {})

        naming_ok = (
            naming.get("submitted") == "YES"
            and naming.get("status") == "OK"
            and naming.get("naming_error_count") == "0"
        )
        correctness_pass = correctness.get("status") == "PASS"
        eligible = naming_ok and correctness_pass

        row = {
            "group": group,
            "submitted": naming.get("submitted", ""),
            "naming_status": naming.get("status", ""),
            "naming_error_count": naming.get("naming_error_count", ""),
            "weights_dir_ok": naming.get("weights_dir_ok", ""),
            "memory_map_json_ok": naming.get("memory_map_json_ok", ""),
            "student_hw_ok": naming.get("student_hw_ok", ""),
            "core_csv_count": naming.get("core_csv_count", ""),
            "core_csv_ok": naming.get("core_csv_ok", ""),
            "bin_count": naming.get("bin_count", ""),
            "bin_suffix_ok": naming.get("bin_suffix_ok", ""),
            "correctness_status": correctness.get("status", "NOT_CHECKED"),
            "correctness_fail_count": correctness.get("fail_count", ""),
            "correctness_total": correctness.get("total", ""),
            "correctness_max_abs": correctness.get("max_abs", ""),
            "correctness_rmse": correctness.get("rmse", ""),
            "eligible_for_profit": "YES" if eligible else "NO",
            "profit_status": "SKIPPED",
            "profit_score": "",
            "baseline_total_cycles": "",
            "candidate_total_cycles": "",
            "raw_perf_ratio": "",
            "Performance_scale": "",
            "clock_scale_after_throttle": "",
            "total_hardware_cost_multiplier": "",
            "extra_cost_ratio": "",
            "grading_cost_adjustment_ratio": "",
            "system_power_multiplier": "",
            "student_config": "",
            "sim_log": "",
            "message": "",
        }

        if not naming_ok:
            row["message"] = "skip profit: submission/naming not OK"
        elif not correctness_pass:
            row["message"] = "skip profit: correctness not PASS"
        else:
            try:
                profit_row = compute_profit_for_group(args, profit, group)
                row.update({k: profit_row.get(k, row.get(k, "")) for k in row})
                row["profit_status"] = "OK"
                row["message"] = "OK"
                pass_profit += 1
            except Exception as e:
                row["profit_status"] = "ERROR"
                row["message"] = str(e)

        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("========== FP2 Profit Batch ==========")
    print(f"naming_csv        : {args.naming_csv}")
    print(f"correctness_csv   : {args.correctness_csv}")
    print(f"baseline_cycles   : {args.baseline_cycles:g}")
    print(f"groups            : {len(rows)}")
    print(f"profit computed   : {pass_profit}")
    print(f"output_csv        : {args.output_csv}")

    return 0 if all(r["profit_status"] != "ERROR" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
