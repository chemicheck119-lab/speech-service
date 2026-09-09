#!/usr/bin/env bash
set -euo pipefail

image_name="chemicheck119-speech-api:local-smoke-$$"
container_name="chemicheck119-speech-api-smoke-$$"
smoke_dir="$(mktemp -d)"

cleanup() {
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  docker image rm "${image_name}" >/dev/null 2>&1 || true
  rm -f "${smoke_dir}/live.json" "${smoke_dir}/ready.json" "${smoke_dir}/unauthorized.json"
  rmdir "${smoke_dir}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build \
  --file Dockerfile.api \
  --build-arg EMBED_WHISPER_MODEL=false \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --tag "${image_name}" \
  .

docker run --detach \
  --name "${container_name}" \
  --read-only \
  --tmpfs /tmp/chemicheck119-speech:rw,noexec,nosuid,size=32m,uid=10001,gid=10001 \
  --memory 2g \
  --cpus 2 \
  --publish 127.0.0.1::8080 \
  --env CHEMICHECK119_SPEECH_API_KEY=container-smoke-placeholder \
  "${image_name}" >/dev/null

published_port="$(docker port "${container_name}" 8080/tcp | sed -E 's/.*:([0-9]+)$/\1/')"
if [[ ! "${published_port}" =~ ^[0-9]+$ ]]; then
  echo "failed to resolve published port" >&2
  exit 1
fi

live_status=""
for _ in $(seq 1 30); do
  live_status="$(curl --silent --output "${smoke_dir}/live.json" \
    --write-out '%{http_code}' "http://127.0.0.1:${published_port}/health/live" || true)"
  [[ "${live_status}" == "200" ]] && break
  sleep 1
done
[[ "${live_status}" == "200" ]]
[[ "$(docker exec "${container_name}" id -u)" == "10001" ]]

ready_status="$(curl --silent --output "${smoke_dir}/ready.json" \
  --write-out '%{http_code}' "http://127.0.0.1:${published_port}/health/ready")"
[[ "${ready_status}" == "503" ]]

unauthorized_status="$(curl --silent --output "${smoke_dir}/unauthorized.json" \
  --write-out '%{http_code}' \
  --request POST \
  --header 'Content-Type: audio/wav' \
  --data-binary '' \
  "http://127.0.0.1:${published_port}/api/v1/transcriptions")"
[[ "${unauthorized_status}" == "401" ]]

python - "${smoke_dir}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
live = json.loads((root / "live.json").read_text())
ready = json.loads((root / "ready.json").read_text())
unauthorized = json.loads((root / "unauthorized.json").read_text())
assert live["status"] == "LIVE"
assert ready["status"] == "NOT_READY"
assert unauthorized["error"]["code"] == "UNAUTHORIZED"
PY

echo "Speech API container smoke passed: live=200 ready=503 unauthorized=401 uid=10001"
