#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

FP2_ROOT="${FP2_ROOT:-${REPO_ROOT}/FP2}"
SUBMISSION_DIR="${SUBMISSION_DIR:-${FP2_ROOT}/submission_example}"
OUT_DIR="${OUT_DIR:-${SUBMISSION_DIR}}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash 06_tar_submit.sh <group>

Examples:
  bash 06_tar_submit.sh 1
  bash 06_tar_submit.sh G01

Environment overrides:
  SUBMISSION_DIR=/path/to/submission_example
  OUT_DIR=/path/to/output_tar_dir
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

raw_group="$1"
if [[ "${raw_group}" =~ ^[Gg]([0-9]{1,2})$ ]]; then
  group_num="${BASH_REMATCH[1]}"
elif [[ "${raw_group}" =~ ^[0-9]{1,2}$ ]]; then
  group_num="${raw_group}"
else
  echo "ERROR: group must be a number like 4 or a tag like G04." >&2
  usage
  exit 2
fi

group_num=$((10#${group_num}))
if (( group_num < 1 || group_num > 99 )); then
  echo "ERROR: group number must be between 1 and 99." >&2
  exit 2
fi

printf -v group_tag "G%02d" "${group_num}"
group_dir_name="CA_FP2_${group_tag}"
group_dir="${SUBMISSION_DIR}/${group_dir_name}"
tar_path="${OUT_DIR}/${group_dir_name}.tar.gz"

if [[ ! -d "${group_dir}" ]]; then
  echo "ERROR: missing group directory: ${group_dir}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

tar -C "${SUBMISSION_DIR}" -czf "${tar_path}" "${group_dir_name}"

echo "Created ${tar_path}"
