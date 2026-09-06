#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"
EXECUTION_CONFIG="${REPOSITORY_ROOT}/config/whisper_lora_execution_v1.json"
EXPERIMENT_CONFIG="${REPOSITORY_ROOT}/config/whisper_lora_experiment_v1.json"
PRIORITY_TERMS="${REPOSITORY_ROOT}/config/domain_hotwords.txt"

if (( $# != 5 )); then
  echo "usage: $0 PYTHON_BIN ARTIFACT_ROOT CONVERSION_DIR CLEAN_REPORT OUTPUT_DIR" >&2
  exit 1
fi

PYTHON_BIN="$1"
ARTIFACT_ROOT="$2"
CONVERSION_DIR="$3"
CLEAN_REPORT="$4"
OUTPUT_DIR="$5"
CONVERSION_REPORT="${CONVERSION_DIR}/conversion-report.json"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python runtime must be executable" >&2
  exit 1
fi
for directory in "${ARTIFACT_ROOT}" "${CONVERSION_DIR}"; do
  if [[ "${directory}" != /* || ! -d "${directory}" || -L "${directory}" ]]; then
    echo "input directories must be absolute non-symlink directories" >&2
    exit 1
  fi
done
for file in "${CONVERSION_REPORT}" "${CLEAN_REPORT}"; do
  if [[ "${file}" != /* || ! -f "${file}" || -L "${file}" ]]; then
    echo "required input must be an absolute regular non-symlink file: ${file}" >&2
    exit 1
  fi
done
if [[ "${OUTPUT_DIR}" != /* || -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "output must be a new absolute path" >&2
  exit 1
fi

EVALUATOR_REVISION="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
REMOTE_MAIN_REVISION="$(
  git -C "${REPOSITORY_ROOT}" ls-remote origin refs/heads/main | awk '{print $1}'
)"
if [[ "${EVALUATOR_REVISION}" != "${REMOTE_MAIN_REVISION}" ]]; then
  echo "local revision must equal the currently advertised main commit" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain)" ]]; then
  echo "repository must be clean before the wind development evaluation" >&2
  exit 1
fi

OUTPUT_PARENT="$(dirname "${OUTPUT_DIR}")"
mkdir -p "${OUTPUT_PARENT}"
chmod 700 "${OUTPUT_PARENT}"
if [[ -L "${OUTPUT_PARENT}" || ! -d "${OUTPUT_PARENT}" ]]; then
  echo "output parent must be a non-symlink directory" >&2
  exit 1
fi
mkdir -m 700 "${OUTPUT_DIR}"

run_arm() {
  local arm_name="$1"
  local arm_output="${OUTPUT_DIR}/${arm_name}"
  PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" \
    -m chemicheck119_speech.lora_dev_evaluation \
    --execution-config "${EXECUTION_CONFIG}" \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --conversion-report "${CONVERSION_REPORT}" \
    --priority-terms "${PRIORITY_TERMS}" \
    --arm "${arm_name}" \
    --output-dir "${arm_output}"
  chmod 700 "${arm_output}"
  chmod 600 "${arm_output}/summary.json" "${arm_output}/records.private.jsonl"
}

run_arm "B_same_conversion_base_control"
run_arm "C_lora_merged_candidate"

PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" \
  -m chemicheck119_speech.lora_wind_report \
  --conversion-report "${CONVERSION_REPORT}" \
  --clean-report "${CLEAN_REPORT}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --priority-terms "${PRIORITY_TERMS}" \
  --b-summary "${OUTPUT_DIR}/B_same_conversion_base_control/summary.json" \
  --b-records "${OUTPUT_DIR}/B_same_conversion_base_control/records.private.jsonl" \
  --c-summary "${OUTPUT_DIR}/C_lora_merged_candidate/summary.json" \
  --c-records "${OUTPUT_DIR}/C_lora_merged_candidate/records.private.jsonl" \
  --output "${OUTPUT_DIR}/wind-development-evaluation.json" \
  --evaluator-revision "${EVALUATOR_REVISION}"
chmod 600 "${OUTPUT_DIR}/wind-development-evaluation.json"

echo "wind development evaluation completed: ${OUTPUT_DIR}/wind-development-evaluation.json"
