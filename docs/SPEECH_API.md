# Bounded Speech API

## 목적과 사실 상태

- 사실 상태: **부분 구현 또는 개발용 데모**
- 입력: 최대 16MiB·60초의 8~48kHz mono/stereo 16-bit PCM WAV
- 출력: 원문을 그대로 보존한 전사문, 구간·타임스탬프, 모델 고유 품질 신호
- 제외: CAS 확정, 화학적 호환성·위험도 판단, CAMEO 실행, 현장 지시

테스트는 생성 WAV와 fake transcriber를 사용합니다. 따라서 실제 신고음성 정확도, 현장 무전
정확도, 스트리밍 지연, 상용 동시성, 현장 안전성을 증명하지 않습니다.

## 실행

배포 환경은 모델을 시작 시점에 내려받지 않도록 검증된 로컬 모델 경로를 사용합니다.
API Key는 명령 인자가 아니라 환경변수로 전달합니다.

```bash
export CHEMICHECK119_SPEECH_API_KEY=local-development-key
export CHEMICHECK119_SPEECH_MODEL=/private/models/faster-whisper-small
export CHEMICHECK119_SPEECH_DEVICE=cpu
export CHEMICHECK119_SPEECH_COMPUTE_TYPE=int8
export CHEMICHECK119_SPEECH_LOCAL_FILES_ONLY=true
chemicheck119-speech-api
```

```bash
curl --fail-with-body \
  -X POST http://127.0.0.1:8080/api/v1/transcriptions \
  -H "Content-Type: audio/wav" \
  -H "X-API-Key: ${CHEMICHECK119_SPEECH_API_KEY}" \
  -H "X-Request-Id: REQ-DEMO-0001" \
  --data-binary @/private/audio/sample.wav
```

## 요청 경계

| 항목 | 동작 |
|---|---|
| Content-Type | `audio/wav`, `audio/x-wav`, `audio/wave`만 허용 |
| 파일 크기 | 기본 16MiB 초과 시 `413` |
| WAV | 8~48kHz, mono/stereo, 16-bit PCM만 허용 |
| 재생시간 | 기본 60초 초과 시 `422` |
| 동시 추론 | 기본 1개, 대기 한도 초과 시 retryable `429` |
| 임시파일 | `0600`, 요청 종료 시 삭제, 원본 미보관 |
| 인증 | `X-API-Key`, 비로컬 bind에서는 익명 모드 금지 |
| 추론 실패 | 내부 오류 내용을 숨기고 retryable `503` |
| 모델 출력 이상 | 비정상 구간·시간·품질 신호를 `502`로 차단 |

## 응답 해석

- `TRANSCRIBED`: 비어 있지 않은 전사문을 반환했습니다. 정확하거나 확인됐다는 뜻은 아닙니다.
- `ABSTAINED_NO_TRANSCRIPT`: 음성 또는 인식 가능한 발화가 없어 빈 전사로 기권했습니다.
- `quality_signals`: faster-whisper가 제공한 보정되지 않은 디코딩 신호입니다. 정답 확률로
  표시하거나 CAS 자동확정 threshold로 사용하지 않습니다.
- `safety_boundary`: Speech Service가 물질 식별·CAS 확인·위험 판단을 수행하지 않았음을
  명시합니다.

다음 단계에서 Analysis와 연결할 때도 같은 `X-Request-Id`를 전달하되, 전사문은 후보 탐색의
입력일 뿐입니다. Resolver 후보는 사용자 확인 전까지 Rule Engine 입력으로 승격할 수 없습니다.
