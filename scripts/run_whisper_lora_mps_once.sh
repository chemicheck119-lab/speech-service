#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"
EXECUTION_CONFIG="${REPOSITORY_ROOT}/config/whisper_lora_execution_v1.json"
EXPERIMENT_CONFIG="${REPOSITORY_ROOT}/config/whisper_lora_experiment_v1.json"

if (( $# != 4 )); then
  echo "usage: $0 PYTHON_BIN ARTIFACT_ROOT COST_QUOTE OUTPUT_DIR" >&2
  exit 1
fi

PYTHON_BIN="$1"
ARTIFACT_ROOT="$2"
COST_QUOTE="$3"
OUTPUT_DIR="$4"

for command in git gcloud caffeinate; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "required command not found: ${command}" >&2
    exit 1
  }
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python runtime must be an executable file" >&2
  exit 1
fi
if [[ ! -d "${ARTIFACT_ROOT}" || -L "${ARTIFACT_ROOT}" ]]; then
  echo "artifact root must be a non-symlink directory" >&2
  exit 1
fi
if [[ ! -f "${COST_QUOTE}" || -L "${COST_QUOTE}" ]]; then
  echo "cost quote must be a regular non-symlink file" >&2
  exit 1
fi
if [[ "${OUTPUT_DIR}" != /* || -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "output must be a new absolute path" >&2
  exit 1
fi

RUNNER_REVISION="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
REMOTE_MAIN_REVISION="$(git -C "${REPOSITORY_ROOT}" ls-remote origin refs/heads/main | awk '{print $1}')"
if [[ "${RUNNER_REVISION}" != "${REMOTE_MAIN_REVISION}" ]]; then
  echo "local revision must equal the currently advertised main commit" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain)" ]]; then
  echo "repository must be clean before the bounded local run" >&2
  exit 1
fi
if [[ "$(gcloud config get-value project 2>/dev/null)" != "chemi-check" ]]; then
  echo "active gcloud project must be chemi-check" >&2
  exit 1
fi

OUTPUT_PARENT="$(dirname "${OUTPUT_DIR}")"
mkdir -p "${OUTPUT_PARENT}"
chmod 700 "${OUTPUT_PARENT}"
if [[ -L "${OUTPUT_PARENT}" || ! -d "${OUTPUT_PARENT}" ]]; then
  echo "output parent must be a non-symlink directory" >&2
  exit 1
fi

AUTHORIZATION_ID="$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["authorization"]["id"])' \
  "${COST_QUOTE}")"
AUTHORIZATION_CLAIM="${OUTPUT_PARENT}/.${AUTHORIZATION_ID}.claim.json"
if [[ -e "${AUTHORIZATION_CLAIM}" || -L "${AUTHORIZATION_CLAIM}" ]]; then
  echo "local authorization claim path already exists" >&2
  exit 1
fi
GCS_PREFIX="$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["private_output"]["gcs_prefix"])' \
  "${EXECUTION_CONFIG}")"
COST_QUOTE_GCS_URI="${GCS_PREFIX}/inputs/${AUTHORIZATION_ID}.cost-quote.json"
AUTHORIZATION_CLAIM_GCS_URI="${GCS_PREFIX}/authorizations/${AUTHORIZATION_ID}.claimed.json"

PYTHONPATH="${REPOSITORY_ROOT}/src" "${PYTHON_BIN}" - \
  "${EXECUTION_CONFIG}" "${COST_QUOTE}" "${RUNNER_REVISION}" <<'PY'
import json
from pathlib import Path
import sys

from chemicheck119_speech.lora_training import validate_cost_quote, validate_gpu_runtime

execution = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = validate_cost_quote(
    quote_path=Path(sys.argv[2]),
    execution_config=execution,
    runner_revision=sys.argv[3],
)
validate_gpu_runtime(execution)
print({"authorization_id": decision.authorization_id, "quoted_total_krw": decision.quoted_total_krw_with_contingency})
PY

gcloud storage cp --if-generation-match=0 \
  "${COST_QUOTE}" "${COST_QUOTE_GCS_URI}" >/dev/null
"${PYTHON_BIN}" - "${COST_QUOTE}" "${AUTHORIZATION_CLAIM}" \
  "${AUTHORIZATION_ID}" "${RUNNER_REVISION}" "${AUTHORIZATION_CLAIM_GCS_URI}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

quote = Path(sys.argv[1])
claim = Path(sys.argv[2])
payload = {
    "schema_version": "1.0.0",
    "protocol_id": "whisper-small-lora-authorization-claim-v1",
    "authorization_id": sys.argv[3],
    "speech_revision": sys.argv[4],
    "cost_quote_sha256": hashlib.sha256(quote.read_bytes()).hexdigest(),
    "claimed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "remote_claim_created": True,
    "remote_object_uri": sys.argv[5],
}
with claim.open("x", encoding="utf-8") as destination:
    destination.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
claim.chmod(0o600)
PY
gcloud storage cp --if-generation-match=0 \
  "${AUTHORIZATION_CLAIM}" "${AUTHORIZATION_CLAIM_GCS_URI}" >/dev/null

EXTERNAL_TIMEOUT_SECONDS="$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["external_timeout_seconds"])' \
  "${EXECUTION_CONFIG}")"
CHILD_PID=""
WATCHDOG_PID=""

cleanup_private_staging() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import shutil
import sys

output = Path(sys.argv[1])
if not output.is_absolute() or output.name in {"", ".", ".."}:
    raise SystemExit("unsafe output path")
for suffix in ("stage", "work"):
    target = output.parent / f".{output.name}.{suffix}"
    if target.is_symlink():
        raise SystemExit(f"refusing to follow staging symlink: {target}")
    if target.is_dir():
        shutil.rmtree(target)
PY
}

cleanup() {
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
  fi
  if [[ -n "${WATCHDOG_PID}" ]] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
  fi
  cleanup_private_staging
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

caffeinate -dimsu env PYTHONPATH="${REPOSITORY_ROOT}/src" \
  "${PYTHON_BIN}" -m chemicheck119_speech.lora_training \
  --execution-config "${EXECUTION_CONFIG}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --cost-quote "${COST_QUOTE}" \
  --authorization-claim "${AUTHORIZATION_CLAIM}" \
  --output-dir "${OUTPUT_DIR}" \
  --confirm-bounded-experiment RUN_BOUNDED_LORA_ONCE \
  --runner-revision "${RUNNER_REVISION}" &
CHILD_PID="$!"

(
  sleep "${EXTERNAL_TIMEOUT_SECONDS}"
  if kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    sleep 60
    kill -KILL "${CHILD_PID}" 2>/dev/null || true
  fi
) &
WATCHDOG_PID="$!"

set +e
wait "${CHILD_PID}"
STATUS="$?"
set -e
CHILD_PID=""
kill "${WATCHDOG_PID}" 2>/dev/null || true
wait "${WATCHDOG_PID}" 2>/dev/null || true
WATCHDOG_PID=""

if (( STATUS != 0 )); then
  exit "${STATUS}"
fi
if [[ ! -f "${OUTPUT_DIR}/training-report.json" ]]; then
  echo "training completed without the registered aggregate report" >&2
  exit 1
fi

echo "bounded local MPS training completed: ${OUTPUT_DIR}/training-report.json"
