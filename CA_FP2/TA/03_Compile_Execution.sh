#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

SYSTEMC_DIR="${SYSTEMC_DIR:-${REPO_ROOT}/SystemC}"
FP2_ROOT="${FP2_ROOT:-${REPO_ROOT}/FP2}"
TA_DIR="${TA_DIR:-${SCRIPT_DIR}}"
DEFAULT_SUBMISSION_DIR="${FP2_ROOT}/submission"
if [[ ! -d "${DEFAULT_SUBMISSION_DIR}" && -d "${FP2_ROOT}/submission_example" ]]; then
  DEFAULT_SUBMISSION_DIR="${FP2_ROOT}/submission_example"
fi
SUBMISSION_DIR="${SUBMISSION_DIR:-${DEFAULT_SUBMISSION_DIR}}"
OUTCOME_DIR="${OUTCOME_DIR:-${FP2_ROOT}/outcome}"
CHECKER="${CHECKER:-${TA_DIR}/00_check_fp2_naming.py}"
CHECK_CSV="${CHECK_CSV:-${OUTCOME_DIR}/fp2_submission_check.csv}"
GROUP_LIST="${GROUP_LIST:-${OUTCOME_DIR}/fp2_groups_to_run.txt}"
PATTERN_DIR="${PATTERN_DIR:-${TA_DIR}/pattern}"
DRAM_JSON_DIR="${DRAM_JSON_DIR:-${SYSTEMC_DIR}/DRAM_json}"
TEST_TOOL_DIR="${TEST_TOOL_DIR:-${SYSTEMC_DIR}/test_tool}"
BUILD_DIR="${BUILD_DIR:-${TEST_TOOL_DIR}/build}"
STUDENT_HW_DST="${STUDENT_HW_DST:-${SYSTEMC_DIR}/include/student_hw.h}"
MAIN_EXE="${MAIN_EXE:-${BUILD_DIR}/main}"

mkdir -p "${OUTCOME_DIR}" "${DRAM_JSON_DIR}"

if [[ -f "${CHECKER}" ]]; then
  python3 "${CHECKER}" --submission-dir "${SUBMISSION_DIR}" --outcome-dir "${OUTCOME_DIR}" >/dev/null || true
fi

if [[ ! -f "${CHECK_CSV}" ]]; then
  echo "ERROR: missing naming check CSV: ${CHECK_CSV}" >&2
  echo "Run ${CHECKER} first." >&2
  exit 1
fi

python3 -c 'import csv, re, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        group = row.get("group", "")
        if (re.fullmatch(r"G(0[1-9]|1[0-9]|2[0-6])", group)
                and row.get("submitted") == "YES"
                and row.get("status") == "OK"
                and row.get("naming_error_count") == "0"):
            print(group)
' "${CHECK_CSV}" > "${GROUP_LIST}"
mapfile -t FP2_GROUPS < "${GROUP_LIST}"

if [[ ${#FP2_GROUPS[@]} -eq 0 ]]; then
  echo "No submitted groups with status=OK and naming_error_count=0."
  exit 0
fi

if [[ ! -f "${PATTERN_DIR}/input_tensor.txt" ]]; then
  echo "ERROR: missing ${PATTERN_DIR}/input_tensor.txt" >&2
  exit 1
fi

if [[ ! -f "${PATTERN_DIR}/input_tensor.bin" ]]; then
  echo "ERROR: missing ${PATTERN_DIR}/input_tensor.bin" >&2
  echo "The simulator reads input_tensor.bin; regenerate FP2 pattern input first." >&2
  exit 1
fi

overall_status=0

echo "========== FP2 Compile & Execution =========="
echo "groups: ${FP2_GROUPS[*]}"
echo "check_csv: ${CHECK_CSV}"
echo "outcome_dir: ${OUTCOME_DIR}"
echo ""

for group in "${FP2_GROUPS[@]}"; do
  group_dir="${SUBMISSION_DIR}/CA_FP2_${group}"
  weights_dir="${group_dir}/weights_${group}"
  out_dir="${OUTCOME_DIR}/${group}"
  sim_log="${out_dir}/sim.log"

  mkdir -p "${out_dir}"

  echo "========== ${group} ==========" | tee "${sim_log}"

  if [[ ! -f "${group_dir}/student_hw_${group}.h" ]]; then
    echo "ERROR: missing ${group_dir}/student_hw_${group}.h" | tee -a "${sim_log}"
    overall_status=1
    continue
  fi

  cp -f "${PATTERN_DIR}/input_tensor.txt" "${DRAM_JSON_DIR}/input_tensor.txt"
  cp -f "${PATTERN_DIR}/input_tensor.bin" "${DRAM_JSON_DIR}/input_tensor.bin"
  cp -f "${group_dir}/student_hw_${group}.h" "${STUDENT_HW_DST}"
  python3 - "${STUDENT_HW_DST}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

# Some submissions were based on baseline_hw.h and kept namespace baseline_hw.
# The simulator includes both baseline_hw.h and student_hw.h, so the working
# copy must expose student_hw::Config to avoid namespace redefinition.
if "namespace student_hw" not in text and "namespace baseline_hw" in text:
    text = text.replace("namespace baseline_hw", "namespace student_hw")
    text = re.sub(r"}\s*//\s*namespace\s+baseline_hw", "} // namespace student_hw", text)
    path.write_text(text, encoding="utf-8")
PY
  rm -f "${out_dir}/output_tensor.bin"
  rm -rf "${out_dir}/state"

  echo "Copied input_tensor.txt/bin to ${DRAM_JSON_DIR}" | tee -a "${sim_log}"
  echo "Copied and normalized student_hw_${group}.h to ${STUDENT_HW_DST}" | tee -a "${sim_log}"
  echo "Configuring ${BUILD_DIR} with HW_CONFIG_CHOICE=2" | tee -a "${sim_log}"
  if ! cmake -S "${TEST_TOOL_DIR}" -B "${BUILD_DIR}" -DHW_CONFIG_CHOICE=2 >>"${sim_log}" 2>&1; then
    echo "ERROR: configure failed for ${group}; see ${sim_log}" | tee -a "${sim_log}"
    overall_status=1
    continue
  fi

  echo "Building ${MAIN_EXE}" | tee -a "${sim_log}"
  if ! cmake --build "${BUILD_DIR}" --target main -j >>"${sim_log}" 2>&1; then
    echo "ERROR: build failed for ${group}; see ${sim_log}" | tee -a "${sim_log}"
    overall_status=1
    continue
  fi

  if [[ ! -x "${MAIN_EXE}" ]]; then
    echo "ERROR: executable not found after build: ${MAIN_EXE}" | tee -a "${sim_log}"
    overall_status=1
    continue
  fi

  echo "Running simulation" | tee -a "${sim_log}"
  (
    cd "${TEST_TOOL_DIR}" && \
      ASMAP_TOKEN_IN_PATH="${DRAM_JSON_DIR}/input_tensor.bin" \
      ASMAP_STATE_DIR="${out_dir}/state" \
      "${MAIN_EXE}" \
        --no-vcd \
        --run-dir "${group_dir}" \
        --instr-dir "${group_dir}" \
        --dram-dir "${weights_dir}" \
        --sim-out-dir "${out_dir}"
  ) >>"${sim_log}" 2>&1
  run_status=$?

  if [[ ${run_status} -ne 0 ]]; then
    echo "ERROR: simulation failed for ${group} with exit code ${run_status}; see ${sim_log}" | tee -a "${sim_log}"
    overall_status=1
    continue
  fi

  if [[ ! -f "${out_dir}/output_tensor.bin" ]]; then
    echo "ERROR: simulation finished but output_tensor.bin was not created in ${out_dir}" | tee -a "${sim_log}"
    overall_status=1
    continue
  fi

  echo "DONE: ${group}" | tee -a "${sim_log}"
  echo "sim_log: ${sim_log}"
  echo "output : ${out_dir}/output_tensor.bin"
  echo ""
done

exit ${overall_status}
