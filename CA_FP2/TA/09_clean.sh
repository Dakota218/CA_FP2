#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

FP2_ROOT="${FP2_ROOT:-${REPO_ROOT}/FP2}"
OUTCOME_DIR="${OUTCOME_DIR:-${FP2_ROOT}/outcome}"

RESOLVED_OUTCOME_DIR="$(realpath -m "${OUTCOME_DIR}")"
RESOLVED_FP2_ROOT="$(realpath -m "${FP2_ROOT}")"
EXPECTED_OUTCOME_DIR="$(realpath -m "${RESOLVED_FP2_ROOT}/outcome")"

if [[ "${RESOLVED_OUTCOME_DIR}" != "${EXPECTED_OUTCOME_DIR}" ]]; then
  echo "ERROR: refusing to clean unexpected outcome directory: ${OUTCOME_DIR}" >&2
  echo "Expected: ${EXPECTED_OUTCOME_DIR}" >&2
  exit 1
fi

mkdir -p "${RESOLVED_OUTCOME_DIR}"

find "${RESOLVED_OUTCOME_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

echo "Cleaned ${RESOLVED_OUTCOME_DIR}"
