#!/usr/bin/env python3
"""
FP2 verifier.

By default this script reads fp2_submission_check.csv, selects groups with
valid submissions, and compares each outcome/GXX/output_tensor.bin against
the golden concatenated output tensor.

  Golden:
    <repo>/TA/pattern/output_tensor.bin

  Per-group simulator output:
    <repo>/FP2/outcome/GXX/output_tensor.bin

Important:
  output_tensor.bin is a raw binary file, so it does not store shape metadata.
  This script compares it as a flat FP32 array by default.

"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SYSTEMC_DIR = Path(os.environ.get("SYSTEMC_DIR", str(REPO_ROOT / "SystemC")))
DEFAULT_FP2_ROOT = Path(os.environ.get("FP2_ROOT", str(REPO_ROOT / "FP2")))
DEFAULT_TA_DIR = Path(os.environ.get("TA_DIR", str(SCRIPT_DIR)))
DEFAULT_OUTCOME_DIR = Path(os.environ.get("OUTCOME_DIR", str(DEFAULT_FP2_ROOT / "outcome")))
DEFAULT_DRAM_JSON_DIR = Path(os.environ.get("DRAM_JSON_DIR", str(DEFAULT_SYSTEMC_DIR / "DRAM_json")))
DEFAULT_PATTERN_INPUT_TENSOR = str(DEFAULT_TA_DIR / "pattern" / "input_tensor.bin")
DEFAULT_SIM_INPUT_TENSOR = str(DEFAULT_DRAM_JSON_DIR / "input_tensor.bin")
DEFAULT_GOLDEN_TENSOR = str(DEFAULT_TA_DIR / "pattern" / "output_tensor.bin")
DEFAULT_OUTPUT_TENSOR = str(DEFAULT_DRAM_JSON_DIR / "output_tensor.bin")
DEFAULT_CHECK_CSV = str(DEFAULT_OUTCOME_DIR / "fp2_submission_check.csv")
DEFAULT_VERIFY_CSV = "fp2_verify_correctness.csv"


def parse_shape(shape_str):
    """
    Parse shape strings like:
      10,10
      10x10
      1,1000

    Return None if shape is omitted.
    """
    if shape_str is None:
        return None

    s = str(shape_str).strip()

    if not s:
        return None

    s = s.lower().replace("x", ",")

    dims = []

    for part in s.split(","):
        part = part.strip()

        if not part:
            continue

        try:
            dim = int(part)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"invalid --shape={shape_str!r}; use format like 10,10 or 10x10"
            ) from e

        if dim <= 0:
            raise argparse.ArgumentTypeError(
                f"invalid --shape={shape_str!r}; every dimension must be > 0"
            )

        dims.append(dim)

    if not dims:
        raise argparse.ArgumentTypeError(
            f"invalid --shape={shape_str!r}; use format like 10,10 or 10x10"
        )

    return tuple(dims)


def check_fp32_file(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    size_bytes = path.stat().st_size

    if size_bytes == 0:
        raise ValueError(f"{name} is empty: {path}")

    if size_bytes % 4 != 0:
        raise ValueError(
            f"{name} size is not a multiple of 4 bytes: {path}, size={size_bytes}"
        )

    return size_bytes // 4


def load_fp32(path, name):
    check_fp32_file(path, name)
    return np.fromfile(path, dtype=np.float32)


def prepare_simulator_io(pattern_input, sim_input, sim_output):
    pattern_input = Path(pattern_input)
    sim_input = Path(sim_input)
    sim_output = Path(sim_output)

    check_fp32_file(pattern_input, "pattern input_tensor")

    sim_input.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pattern_input, sim_input)

    removed_output = False
    if sim_output.exists():
        sim_output.unlink()
        removed_output = True

    print("========== FP2 Prepare Simulator I/O ==========")
    print(f"pattern_input : {pattern_input}")
    print(f"sim_input     : {sim_input}")
    print(f"sim_output    : {sim_output}")
    print(f"copied input  : {pattern_input.stat().st_size} bytes")
    print(f"removed output: {'yes' if removed_output else 'no (did not exist)'}")
    print("")
    print("Next:")
    print(f"  1. cd {DEFAULT_SYSTEMC_DIR / 'test_tool'}")
    print("  2. ./build/main")
    print(f"  3. cd {DEFAULT_TA_DIR} && ./04_verify.py")


def validate_shape(arr, shape, name):
    if shape is None:
        return arr

    expected = int(np.prod(shape))

    if arr.size != expected:
        raise ValueError(
            f"{name} has {arr.size} FP32 elems, but --shape {shape} needs "
            f"{expected} FP32 elems"
        )

    return arr.reshape(shape)


def compare_arrays(student, golden, atol, rtol):
    student64 = student.astype(np.float64)
    golden64 = golden.astype(np.float64)

    diff = student64 - golden64
    abs_diff = np.abs(diff)

    denom = np.maximum(np.abs(golden64), np.finfo(np.float32).tiny)
    rel_diff = abs_diff / denom

    close = np.isclose(
        student,
        golden,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )

    fail_count = int(close.size - np.count_nonzero(close))

    if abs_diff.size:
        worst_flat = int(np.argmax(abs_diff))
        max_abs = float(abs_diff.flat[worst_flat])
        mean_abs = float(abs_diff.mean())
        rmse = float(np.sqrt(np.mean(diff * diff)))
        max_rel = float(rel_diff.flat[int(np.argmax(rel_diff))])
    else:
        worst_flat = -1
        max_abs = 0.0
        mean_abs = 0.0
        rmse = 0.0
        max_rel = 0.0

    return {
        "passed": bool(np.all(close)),
        "fail_count": fail_count,
        "total": int(close.size),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_rel": max_rel,
        "worst_flat": worst_flat,
        "close": close,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
    }


def format_index(flat_idx, shape):
    if flat_idx < 0:
        return "N/A"

    if shape is None:
        return str(flat_idx)

    return str(np.unravel_index(flat_idx, shape))


def print_mismatches(result, student, golden, show_mismatch):
    if result["passed"] or show_mismatch <= 0:
        return

    bad_flat = np.flatnonzero(~result["close"].ravel())
    shape = student.shape if student.ndim > 1 else None

    print(f"first {min(show_mismatch, bad_flat.size)} mismatch(es):")

    student_flat = student.ravel()
    golden_flat = golden.ravel()
    abs_flat = result["abs_diff"].ravel()
    rel_flat = result["rel_diff"].ravel()

    for flat_idx in bad_flat[:show_mismatch]:
        flat_idx = int(flat_idx)
        idx_str = format_index(flat_idx, shape)

        print(
            f"  idx={idx_str:>14s}, flat={flat_idx:8d}, "
            f"student={student_flat[flat_idx]:.9g}, "
            f"golden={golden_flat[flat_idx]:.9g}, "
            f"abs={abs_flat[flat_idx]:.9g}, "
            f"rel={rel_flat[flat_idx]:.9g}"
        )


VERIFY_FIELDS = [
    "group",
    "status",
    "output_tensor",
    "golden_elems",
    "student_elems",
    "fail_count",
    "total",
    "max_abs",
    "mean_abs",
    "rmse",
    "max_rel",
    "worst_flat",
    "message",
]


def groups_from_check_csv(check_csv):
    check_csv = Path(check_csv)
    if not check_csv.is_file():
        raise FileNotFoundError(f"submission check CSV not found: {check_csv}")

    groups = []
    with check_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("submitted") == "YES"
                and row.get("status") == "OK"
                and row.get("naming_error_count") == "0"
            ):
                groups.append(row["group"])

    return groups


def verify_one_group(group, outcome_dir, golden, golden_path, shape, atol, rtol, show_mismatch):
    output_path = Path(outcome_dir) / group / "output_tensor.bin"
    row = {
        "group": group,
        "status": "ERROR",
        "output_tensor": str(output_path),
        "golden_elems": int(golden.size),
        "student_elems": "",
        "fail_count": "",
        "total": "",
        "max_abs": "",
        "mean_abs": "",
        "rmse": "",
        "max_rel": "",
        "worst_flat": "",
        "message": "",
    }

    print(f"========== {group} ==========")
    print(f"golden_tensor   : {golden_path}")
    print(f"simulator_output: {output_path}")

    try:
        student = load_fp32(output_path, f"{group} output_tensor")
        row["student_elems"] = int(student.size)

        if golden.size != student.size:
            row["status"] = "FAIL"
            row["message"] = (
                f"element count mismatch: golden={golden.size}, student={student.size}"
            )
            print(f"[FAIL] {row['message']}")
            print("")
            return row, False

        golden_cmp = golden
        student_cmp = student
        if shape is not None:
            golden_cmp = validate_shape(golden_cmp, shape, "golden output_tensor")
            student_cmp = validate_shape(student_cmp, shape, f"{group} output_tensor")

        if not np.isfinite(student_cmp).all():
            row["status"] = "FAIL"
            row["message"] = "student output contains NaN or Inf"
            print(f"[FAIL] {row['message']}")
            print("")
            return row, False

        result = compare_arrays(
            student=student_cmp,
            golden=golden_cmp,
            atol=atol,
            rtol=rtol,
        )

        row.update(
            {
                "status": "PASS" if result["passed"] else "FAIL",
                "fail_count": result["fail_count"],
                "total": result["total"],
                "max_abs": f"{result['max_abs']:.9g}",
                "mean_abs": f"{result['mean_abs']:.9g}",
                "rmse": f"{result['rmse']:.9g}",
                "max_rel": f"{result['max_rel']:.9g}",
                "worst_flat": result["worst_flat"],
                "message": "OK" if result["passed"] else "values mismatch",
            }
        )

        print(f"status    : {row['status']}")
        print(f"fail      : {result['fail_count']}/{result['total']}")
        print(f"max_abs   : {result['max_abs']:.9g}")
        print(f"mean_abs  : {result['mean_abs']:.9g}")
        print(f"rmse      : {result['rmse']:.9g}")
        print(f"max_rel   : {result['max_rel']:.9g}")
        print(f"worst_idx : flat={result['worst_flat']}")

        print_mismatches(
            result=result,
            student=student_cmp,
            golden=golden_cmp,
            show_mismatch=show_mismatch,
        )
        print("")
        return row, bool(result["passed"])

    except Exception as e:
        row["status"] = "ERROR"
        row["message"] = str(e)
        print(f"[ERROR] {e}")
        print("")
        return row, False


def write_verify_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VERIFY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Verify FP2 output_tensor.bin against golden output_tensor.bin."
    )

    parser.add_argument(
        "--golden-tensor",
        default=DEFAULT_GOLDEN_TENSOR,
        help="Path to golden output_tensor.bin.",
    )

    parser.add_argument(
        "--output-tensor",
        default=DEFAULT_OUTPUT_TENSOR,
        help="Path to simulator output_tensor.bin used with --single.",
    )

    parser.add_argument(
        "--single",
        action="store_true",
        help="Verify only --output-tensor instead of all OK groups in the outcome directory.",
    )

    parser.add_argument(
        "--check-csv",
        default=DEFAULT_CHECK_CSV,
        help="Path to fp2_submission_check.csv used to find valid submitted groups.",
    )

    parser.add_argument(
        "--outcome-dir",
        default=DEFAULT_OUTCOME_DIR,
        help="Directory containing per-group outcome/GXX/output_tensor.bin files.",
    )

    parser.add_argument(
        "--verify-csv",
        default=None,
        help=(
            "Path to write verification summary CSV. Default: "
            "<outcome-dir>/fp2_verify_correctness.csv"
        ),
    )

    parser.add_argument(
        "--prepare-sim",
        action="store_true",
        help=(
            "Prepare simulator I/O: copy pattern input_tensor.bin to the "
            "simulator input path and delete the old simulator output file."
        ),
    )

    parser.add_argument(
        "--pattern-input-tensor",
        default=DEFAULT_PATTERN_INPUT_TENSOR,
        help="Path to pattern input_tensor.bin used by --prepare-sim.",
    )

    parser.add_argument(
        "--sim-input-tensor",
        default=DEFAULT_SIM_INPUT_TENSOR,
        help="Path to simulator input_tensor.bin used by --prepare-sim.",
    )

    parser.add_argument(
        "--shape",
        type=parse_shape,
        default=None,
        help=(
            "Optional logical shape, e.g. 10,10 or 10x10. "
            "The binary file itself is still read as FP32."
        ),
    )

    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)

    parser.add_argument(
        "--show-mismatch",
        type=int,
        default=10,
        help="Show first N mismatches.",
    )

    args = parser.parse_args()

    golden_path = Path(args.golden_tensor)
    output_path = Path(args.output_tensor)

    try:
        if args.prepare_sim:
            prepare_simulator_io(
                pattern_input=args.pattern_input_tensor,
                sim_input=args.sim_input_tensor,
                sim_output=args.output_tensor,
            )
            return 0

        print("========== FP2 Verify ==========")
        print(f"golden_tensor   : {golden_path}")
        print(f"tolerance       : atol={args.atol}, rtol={args.rtol}")

        golden = load_fp32(golden_path, "golden output_tensor")
        if not np.isfinite(golden).all():
            print("[FAIL] FP2 verify FAIL")
            print("Reason: golden output contains NaN or Inf")
            return 1

        if not args.single:
            outcome_dir = Path(args.outcome_dir)
            verify_csv = Path(args.verify_csv) if args.verify_csv else outcome_dir / DEFAULT_VERIFY_CSV
            groups = groups_from_check_csv(args.check_csv)

            print(f"check_csv       : {args.check_csv}")
            print(f"outcome_dir     : {outcome_dir}")
            print(f"verify_csv      : {verify_csv}")
            print(f"groups          : {' '.join(groups) if groups else '(none)'}")
            print("")
            print("Raw FP32 size:")
            print(f"  golden concat : {golden.size} elems ({golden.size * 4} bytes)")
            print("")

            rows = []
            pass_count = 0
            for group in groups:
                row, passed = verify_one_group(
                    group=group,
                    outcome_dir=outcome_dir,
                    golden=golden,
                    golden_path=golden_path,
                    shape=args.shape,
                    atol=args.atol,
                    rtol=args.rtol,
                    show_mismatch=args.show_mismatch,
                )
                rows.append(row)
                if passed:
                    pass_count += 1

            write_verify_csv(verify_csv, rows)

            fail_count = len(rows) - pass_count
            print("========== FP2 Verify Summary ==========")
            print(f"checked groups : {len(rows)}")
            print(f"PASS           : {pass_count}")
            print(f"FAIL/ERROR     : {fail_count}")
            print(f"verify_csv     : {verify_csv}")

            return 0 if fail_count == 0 else 1

        print(f"simulator_output: {output_path}")

        student = load_fp32(output_path, "simulator output_tensor")

        print("")
        print("Raw FP32 size:")
        print(f"  golden   : {golden.size} elems ({golden.size * 4} bytes)")
        print(f"  simulator: {student.size} elems ({student.size * 4} bytes)")

        if golden.size != student.size:
            print("")
            print("[FAIL] FP2 verify FAIL")
            print(
                f"Reason: element count mismatch: "
                f"golden={golden.size}, simulator={student.size}"
            )
            print("")
            print("Hint: run './04_verify.py --prepare-sim' before ./build/main.")
            print("      This copies the FP2 pattern input into DRAM_json/input_tensor.bin")
            print("      and removes the stale DRAM_json/output_tensor.bin so old tail data")
            print("      cannot remain after simulator writes the new output.")
            return 1

        if args.shape is not None:
            golden = validate_shape(golden, args.shape, "golden output_tensor")
            student = validate_shape(student, args.shape, "simulator output_tensor")

        print("")
        if args.shape is None:
            print(f"Compare shape    : flat ({golden.size},)")
        else:
            print(f"Compare shape    : {args.shape}")

        if not np.isfinite(golden).all():
            print("")
            print("[FAIL] FP2 verify FAIL")
            print("Reason: golden output contains NaN or Inf")
            return 1

        if not np.isfinite(student).all():
            print("")
            print("[FAIL] FP2 verify FAIL")
            print("Reason: simulator output contains NaN or Inf")
            return 1

        result = compare_arrays(
            student=student,
            golden=golden,
            atol=args.atol,
            rtol=args.rtol,
        )

        print("")
        print("Compare result:")
        print(f"  status    : {'PASS' if result['passed'] else 'FAIL'}")
        print(f"  fail      : {result['fail_count']}/{result['total']}")
        print(f"  max_abs   : {result['max_abs']:.9g}")
        print(f"  mean_abs  : {result['mean_abs']:.9g}")
        print(f"  rmse      : {result['rmse']:.9g}")
        print(f"  max_rel   : {result['max_rel']:.9g}")
        print(
            f"  worst_idx : {format_index(result['worst_flat'], student.shape if student.ndim > 1 else None)} "
            f"(flat={result['worst_flat']})"
        )

        print_mismatches(
            result=result,
            student=student,
            golden=golden,
            show_mismatch=args.show_mismatch,
        )

        print("")

        if result["passed"]:
            print("[OK] FP2 verify PASS")
            return 0

        print("[FAIL] FP2 verify FAIL")
        return 1

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
