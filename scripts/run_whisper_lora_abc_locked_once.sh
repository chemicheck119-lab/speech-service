#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"
EXPERIMENT_CONFIG="${REPOSITORY_ROOT}/config/whisper_lora_experiment_v1.json"
PRIORITY_TERMS="${REPOSITORY_ROOT}/config/domain_hotwords.txt"

if (( $# != 5 )); then
  echo "usage: $0 PYTHON_BIN CONVERSION_DIR OPERATIONAL_MODEL_DIR VALIDATION_DIR OUTPUT_DIR" >&2
  exit 1
fi

PYTHON_BIN="$1"
CONVERSION_DIR="$2"
OPERATIONAL_MODEL_DIR="$3"
VALIDATION_DIR="$4"
OUTPUT_DIR="$5"
CONVERSION_REPORT="${CONVERSION_DIR}/conversion-report.json"
AUDIO_ARCHIVE="${VALIDATION_DIR}/VS_광주_화재.zip"
LABEL_ARCHIVE="${VALIDATION_DIR}/VL_광주_화재.zip"
DATASET_MANIFEST="${VALIDATION_DIR}/manifest.json"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python runtime must be executable" >&2
  exit 1
fi
for directory in "${CONVERSION_DIR}" "${OPERATIONAL_MODEL_DIR}" "${VALIDATION_DIR}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "input directories must be non-symlink directories" >&2
    exit 1
  fi
done
for file in "${CONVERSION_REPORT}" "${AUDIO_ARCHIVE}" "${LABEL_ARCHIVE}" "${DATASET_MANIFEST}"; do
  if [[ ! -f "${file}" || -L "${file}" ]]; then
    echo "required input must be a regular non-symlink file: ${file}" >&2
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
  echo "repository must be clean before the locked evaluation" >&2
  exit 1
fi

MODEL_PATHS="$({
  PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" - \
    "${CONVERSION_REPORT}" "${OPERATIONAL_MODEL_DIR}" <<'PY'
from pathlib import Path
import sys

from chemicheck119_speech.lora_conversion import validate_conversion_output

report_path = Path(sys.argv[1])
operational = Path(sys.argv[2])
report = validate_conversion_output(report_path)
arms = report["arms"]
expected_revision = arms["A_operational_baseline"]["revision"]
if operational.name != expected_revision or not (operational / "model.bin").is_file():
    raise SystemExit("operational model does not match the pinned revision")
print(operational)
print(report_path.parent / arms["B_same_conversion_base_control"]["path"])
print(report_path.parent / arms["C_lora_merged_candidate"]["path"])
PY
})" || {
  echo "model preflight failed" >&2
  exit 1
}
MODEL_A="$(printf '%s\n' "${MODEL_PATHS}" | sed -n '1p')"
MODEL_B="$(printf '%s\n' "${MODEL_PATHS}" | sed -n '2p')"
MODEL_C="$(printf '%s\n' "${MODEL_PATHS}" | sed -n '3p')"
if [[ -z "${MODEL_A}" || -z "${MODEL_B}" || -z "${MODEL_C}" ]] || \
  [[ "$(printf '%s\n' "${MODEL_PATHS}" | wc -l | tr -d ' ')" != "3" ]]; then
  echo "model preflight returned an invalid arm count" >&2
  exit 1
fi

PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" - \
  "${DATASET_MANIFEST}" "${AUDIO_ARCHIVE}" "${LABEL_ARCHIVE}" <<'PY'
from pathlib import Path
import sys

from chemicheck119_speech.provenance import validate_evaluation_manifest

report = validate_evaluation_manifest(
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
)
if report["evaluation_id"] != "speech_aihub119_gwangju_fire_validation_77":
    raise SystemExit("dataset is not the locked Gwangju Validation evaluation")
if report["record_count"] != 77:
    raise SystemExit("locked evaluation record count must be 77")
print({"status": "validated", "record_count": report["record_count"]})
PY

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
  local model_path="$2"
  local arm_output="${OUTPUT_DIR}/${arm_name}"
  PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" \
    -m chemicheck119_speech.cli \
    --audio-archive "${AUDIO_ARCHIVE}" \
    --label-archive "${LABEL_ARCHIVE}" \
    --dataset-manifest "${DATASET_MANIFEST}" \
    --hotwords-file "${PRIORITY_TERMS}" \
    --variants baseline \
    --model "${model_path}" \
    --device cpu \
    --compute-type int8 \
    --cpu-threads 4 \
    --local-files-only \
    --output-dir "${arm_output}"
  chmod 700 "${arm_output}"
  chmod 600 "${arm_output}/summary.json" "${arm_output}/records.private.jsonl"
}

run_arm "A_operational_baseline" "${MODEL_A}"
run_arm "B_same_conversion_base_control" "${MODEL_B}"
run_arm "C_lora_merged_candidate" "${MODEL_C}"

PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" \
  -m chemicheck119_speech.lora_abc_report \
  --conversion-report "${CONVERSION_REPORT}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --priority-terms "${PRIORITY_TERMS}" \
  --a-summary "${OUTPUT_DIR}/A_operational_baseline/summary.json" \
  --a-records "${OUTPUT_DIR}/A_operational_baseline/records.private.jsonl" \
  --b-summary "${OUTPUT_DIR}/B_same_conversion_base_control/summary.json" \
  --b-records "${OUTPUT_DIR}/B_same_conversion_base_control/records.private.jsonl" \
  --c-summary "${OUTPUT_DIR}/C_lora_merged_candidate/summary.json" \
  --c-records "${OUTPUT_DIR}/C_lora_merged_candidate/records.private.jsonl" \
  --output "${OUTPUT_DIR}/abc-locked-evaluation.json" \
  --evaluator-revision "${EVALUATOR_REVISION}"
chmod 600 "${OUTPUT_DIR}/abc-locked-evaluation.json"

echo "locked A/B/C evaluation completed: ${OUTPUT_DIR}/abc-locked-evaluation.json"
