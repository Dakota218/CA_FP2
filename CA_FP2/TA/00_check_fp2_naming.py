#!/usr/bin/env python3
"""
Check FP2 submission folder/file naming.

Default check target:
  <repo>/FP2/submission

Default output:
  <repo>/FP2/outcome/fp2_submission_check.csv
  <repo>/FP2/outcome/fp2_submission_check_errors.csv


Rules:
  - There are 26 groups: G01 to G26.
  - Group directory must be named CA_FP2_GXX.
  - Weights directory must be named weights_GXX.
  - Student hardware header must be named student_hw_GXX.h.
  - Memory map file must be named memory_map_sim_GXX.json.
  - Core CSV files must be named core_<0..63>_GXX.csv.
  - .bin files under weights_GXX may use any name, but the filename stem
    must end with _GXX.
"""

import argparse
import csv
import os
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_FP2_ROOT = Path(os.environ.get("FP2_ROOT", str(REPO_ROOT / "FP2")))
DEFAULT_SUBMISSION_DIR = DEFAULT_FP2_ROOT / "submission"
if not DEFAULT_SUBMISSION_DIR.exists() and (DEFAULT_FP2_ROOT / "submission_example").exists():
    DEFAULT_SUBMISSION_DIR = DEFAULT_FP2_ROOT / "submission_example"
DEFAULT_SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", str(DEFAULT_SUBMISSION_DIR)))
DEFAULT_OUTCOME_DIR = Path(os.environ.get("OUTCOME_DIR", str(DEFAULT_FP2_ROOT / "outcome")))
DEFAULT_GROUPS = 26
DEFAULT_CORES = 64


SUMMARY_FIELDS = [
    "group",
    "expected_group_dir",
    "submitted",
    "weights_dir_ok",
    "memory_map_json_ok",
    "student_hw_ok",
    "core_csv_count",
    "core_csv_ok",
    "missing_core_csv",
    "unexpected_core_csv",
    "duplicate_core_csv",
    "bin_count",
    "bin_suffix_ok",
    "naming_error_count",
    "status",
]

ERROR_FIELDS = ["group", "kind", "path", "message"]


def yes_no(value):
    return "YES" if value else "NO"


def group_tag(group_id):
    return f"G{group_id:02d}"


def rel(path, base):
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def add_error(errors, group, kind, path, message):
    errors.append(
        {
            "group": group,
            "kind": kind,
            "path": path,
            "message": message,
        }
    )


def inspect_unexpected_group_dirs(submission_dir, expected_names):
    errors = []

    if not submission_dir.exists():
        add_error(
            errors,
            "ALL",
            "missing_submission_dir",
            str(submission_dir),
            "submission directory does not exist",
        )
        return errors

    for child in sorted(submission_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in expected_names:
            continue

        add_error(
            errors,
            "UNKNOWN",
            "unexpected_group_dir",
            rel(child, submission_dir),
            "unexpected top-level directory; expected CA_FP2_GXX",
        )

    return errors


def check_group(submission_dir, group_id, core_count):
    tag = group_tag(group_id)
    expected_group_name = f"CA_FP2_{tag}"
    group_dir = submission_dir / expected_group_name
    weights_dir = group_dir / f"weights_{tag}"
    memory_map = weights_dir / f"memory_map_sim_{tag}.json"
    student_hw = group_dir / f"student_hw_{tag}.h"

    errors = []
    submitted = group_dir.is_dir()

    if not submitted:
        add_error(
            errors,
            tag,
            "missing_group_dir",
            expected_group_name,
            f"missing group directory {expected_group_name}",
        )
        return {
            "group": tag,
            "expected_group_dir": expected_group_name,
            "submitted": "NO",
            "weights_dir_ok": "NO",
            "memory_map_json_ok": "NO",
            "student_hw_ok": "NO",
            "core_csv_count": 0,
            "core_csv_ok": "NO",
            "missing_core_csv": ",".join(str(i) for i in range(core_count)),
            "unexpected_core_csv": "",
            "duplicate_core_csv": "",
            "bin_count": 0,
            "bin_suffix_ok": "NO",
            "naming_error_count": len(errors),
            "status": "NOT_SUBMITTED",
        }, errors

    if not weights_dir.is_dir():
        add_error(
            errors,
            tag,
            "missing_weights_dir",
            rel(weights_dir, submission_dir),
            f"missing weights directory weights_{tag}",
        )

    if not memory_map.is_file():
        add_error(
            errors,
            tag,
            "missing_memory_map_json",
            rel(memory_map, submission_dir),
            f"missing memory_map_sim_{tag}.json under weights_{tag}",
        )

    if not student_hw.is_file():
        add_error(
            errors,
            tag,
            "missing_student_hw",
            rel(student_hw, submission_dir),
            f"missing student_hw_{tag}.h under {expected_group_name}",
        )

    for header_path in sorted(group_dir.glob("*.h")):
        if header_path.name == student_hw.name:
            continue
        add_error(
            errors,
            tag,
            "bad_student_hw_name",
            rel(header_path, submission_dir),
            f"unexpected header filename; expected student_hw_{tag}.h",
        )

    core_pattern = re.compile(rf"^core_(\d+)_{tag}\.csv$")
    wrong_suffix_csv_pattern = re.compile(r"^core_(\d+)_G\d{2}\.csv$")
    core_seen = {}
    unexpected_core_csv = []
    duplicate_core_csv = []

    for csv_path in sorted(group_dir.glob("*.csv")):
        match = core_pattern.fullmatch(csv_path.name)
        if match:
            core_id = int(match.group(1))
            core_seen.setdefault(core_id, []).append(csv_path)
            continue

        unexpected_core_csv.append(csv_path.name)
        if wrong_suffix_csv_pattern.fullmatch(csv_path.name):
            message = f"core CSV has wrong group suffix; expected _{tag}.csv"
        else:
            message = f"unexpected CSV filename; expected core_<0..{core_count - 1}>_{tag}.csv"
        add_error(errors, tag, "bad_core_csv_name", rel(csv_path, submission_dir), message)

    for core_id, paths in sorted(core_seen.items()):
        if core_id < 0 or core_id >= core_count:
            unexpected_core_csv.extend(path.name for path in paths)
            for path in paths:
                add_error(
                    errors,
                    tag,
                    "bad_core_index",
                    rel(path, submission_dir),
                    f"core index {core_id} is outside expected range 0..{core_count - 1}",
                )
        elif len(paths) > 1:
            duplicate_core_csv.append(str(core_id))
            for path in paths[1:]:
                add_error(
                    errors,
                    tag,
                    "duplicate_core_csv",
                    rel(path, submission_dir),
                    f"duplicate core_{core_id}_{tag}.csv",
                )

    missing_core_csv = [str(i) for i in range(core_count) if i not in core_seen]
    if missing_core_csv:
        add_error(
            errors,
            tag,
            "missing_core_csv",
            rel(group_dir, submission_dir),
            "missing core CSV index(es): " + ",".join(missing_core_csv),
        )

    bin_count = 0
    bin_suffix_ok = True
    if weights_dir.is_dir():
        for bin_path in sorted(weights_dir.glob("*.bin")):
            bin_count += 1
            if not bin_path.stem.endswith(f"_{tag}"):
                bin_suffix_ok = False
                add_error(
                    errors,
                    tag,
                    "bad_bin_suffix",
                    rel(bin_path, submission_dir),
                    f".bin filename stem must end with _{tag}",
                )

        for child in sorted(weights_dir.iterdir()):
            if child.name == memory_map.name or child.suffix == ".bin":
                continue
            add_error(
                errors,
                tag,
                "unexpected_weights_file",
                rel(child, submission_dir),
                f"unexpected file under weights_{tag}; expected memory_map_sim_{tag}.json or .bin",
            )
    else:
        bin_suffix_ok = False

    weights_ok = weights_dir.is_dir()
    json_ok = memory_map.is_file()
    student_hw_ok = student_hw.is_file()
    core_ok = not missing_core_csv and not unexpected_core_csv and not duplicate_core_csv

    if not errors:
        status = "OK"
    elif submitted:
        status = "HAS_ERRORS"
    else:
        status = "NOT_SUBMITTED"

    summary = {
        "group": tag,
        "expected_group_dir": expected_group_name,
        "submitted": yes_no(submitted),
        "weights_dir_ok": yes_no(weights_ok),
        "memory_map_json_ok": yes_no(json_ok),
        "student_hw_ok": yes_no(student_hw_ok),
        "core_csv_count": sum(1 for i in range(core_count) if i in core_seen),
        "core_csv_ok": yes_no(core_ok),
        "missing_core_csv": ",".join(missing_core_csv),
        "unexpected_core_csv": ",".join(unexpected_core_csv),
        "duplicate_core_csv": ",".join(duplicate_core_csv),
        "bin_count": bin_count,
        "bin_suffix_ok": yes_no(bin_suffix_ok),
        "naming_error_count": len(errors),
        "status": status,
    }

    return summary, errors


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check FP2 group submission completeness and naming."
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=DEFAULT_SUBMISSION_DIR,
        help=f"default: {DEFAULT_SUBMISSION_DIR}",
    )
    parser.add_argument(
        "--outcome-dir",
        type=Path,
        default=DEFAULT_OUTCOME_DIR,
        help=f"default: {DEFAULT_OUTCOME_DIR}",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=DEFAULT_GROUPS,
        help=f"number of groups to check, default: {DEFAULT_GROUPS}",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=DEFAULT_CORES,
        help=f"number of core CSV files expected per group, default: {DEFAULT_CORES}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    expected_names = {f"CA_FP2_{group_tag(i)}" for i in range(1, args.groups + 1)}
    summary_rows = []
    error_rows = inspect_unexpected_group_dirs(args.submission_dir, expected_names)

    for group_id in range(1, args.groups + 1):
        summary, errors = check_group(args.submission_dir, group_id, args.cores)
        summary_rows.append(summary)
        error_rows.extend(errors)

    summary_csv = args.outcome_dir / "fp2_submission_check.csv"
    errors_csv = args.outcome_dir / "fp2_submission_check_errors.csv"

    write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    write_csv(errors_csv, ERROR_FIELDS, error_rows)

    ok_count = sum(1 for row in summary_rows if row["status"] == "OK")
    submitted_count = sum(1 for row in summary_rows if row["submitted"] == "YES")
    missing_count = args.groups - submitted_count
    error_count = len(error_rows)

    print("========== FP2 Submission Naming Check ==========")
    print(f"submission_dir : {args.submission_dir}")
    print(f"outcome_dir    : {args.outcome_dir}")
    print(f"groups checked : {args.groups}")
    print(f"submitted      : {submitted_count}")
    print(f"not submitted  : {missing_count}")
    print(f"OK             : {ok_count}")
    print(f"errors         : {error_count}")
    print(f"summary_csv    : {summary_csv}")
    print(f"errors_csv     : {errors_csv}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
