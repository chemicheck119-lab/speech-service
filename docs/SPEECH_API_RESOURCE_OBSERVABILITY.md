# Speech API runtime resource 관측

## 사실 상태

- cgroup·process memory counter 수집: **구현 완료**
- numeric-only structured log와 fail-open 계측 경계: **구현 완료**
- 단위·회귀·OpenAPI 계약 검사: **구현 완료**
- immutable image build·Cloud Run candidate 배포: **부분 구현 또는 개발용 데모 — 검증 완료**
- 성공 요청별 process high-water mark·cgroup current 관측: **부분 구현 또는 개발용 데모 — 검증 완료**
- 실제 cgroup peak와 8GiB 축소 가능성: **검증되지 않은 가설**
- GPU latency·비용 우위: **검증되지 않은 가설**

제한된 실제 Cloud Run 값을 측정했지만 “메모리가 충분하다”, “4GiB로 줄일 수 있다” 또는
“GPU가 필요 없다”고 말하지 않습니다. 동일 합성 입력 성공 3건의 process high-water mark는
안전한 최소 memory가 아니며 cgroup peak도 관측되지 않았습니다.

## 왜 추가했는가

현재 Cloud Run request log는 상태와 E2E latency를 보여주지만 개별 전사 직후 memory
high-water mark를 제공하지 않습니다. 기존 application `INFO` event도 배포된 preview의 Cloud
Logging에서 관찰되지 않아, 분 단위 container utilization만으로 4 CPU·8GiB 구성을 줄일 수
있는지 방어하기 어려웠습니다.

이 계측은 전사 성공 응답을 만든 직후 Linux kernel이 제공하는 숫자 counter를 읽습니다.
모델 출력이나 전사문을 다시 읽지 않으며 API 응답 schema도 바꾸지 않습니다.

## 기록하는 필드

| 필드 | 의미 | 한계 |
|---|---|---|
| `cgroup_memory_current_bytes` | 관측 시점의 container cgroup 사용량 | 한 요청만의 memory가 아님 |
| `cgroup_memory_peak_bytes` | 현재 container 생명주기의 cgroup peak | 어느 요청에서 발생했는지 단독 확정 불가 |
| `cgroup_memory_limit_bytes` | kernel에 보이는 memory limit | 플랫폼 설정과 별도 대조 필요 |
| `process_current_rss_bytes` | Linux `/proc/self/statm`의 process RSS | 공유 page 해석 한계가 있음 |
| `process_max_rss_bytes` | process 생명주기의 `ru_maxrss` | 요청별 peak가 아니라 누적 high-water mark |
| `processing_seconds` | 해당 성공 요청의 모델 추론시간 | client queue·network 시간 제외 |
| `audio_seconds` | 입력 WAV 재생시간 | 음성 내용은 포함하지 않음 |

cgroup v2를 우선 사용하고, 제한된 v1 counter만 fallback으로 읽습니다. counter는 최대 32자
ASCII 정수, 최대 `2^60` bytes로 제한합니다. 파일 내용이 `max`, 비수치, 과도한 길이이거나
읽을 수 없으면 `null`로 기권합니다.

## 로그·실패 경계

성공 전사 뒤 다음 형태의 event를 한 줄 JSON으로 기록합니다.

```json
{
  "event": "speech_resource_sample",
  "request_id": "REQ-...",
  "processing_seconds": 3.2,
  "audio_seconds": 7.4,
  "resource_observation_available": true,
  "cgroup_version": "v2",
  "cgroup_memory_current_bytes": 0,
  "cgroup_memory_peak_bytes": 0,
  "cgroup_memory_limit_bytes": 0,
  "process_current_rss_bytes": 0,
  "process_max_rss_bytes": 0
}
```

예시의 `0`은 schema 설명용이며 측정값이 아닙니다. production entrypoint는 application
logger에 단일 `StreamHandler`를 idempotent하게 구성하고 Uvicorn access log는 계속 끕니다.

계측이 실패하면 오류 메시지·경로·파일 내용을 버리고 예외 유형만 기록합니다. 관측 실패는
전사 성공을 5xx로 바꾸지 않습니다. 다음 정보는 event에 넣지 않습니다.

- 음성 bytes·파일 경로
- 전사문·segment·token
- API Key·ID token
- 물질명·CAS·위험도·CAMEO 결과

## 실제 배포 Gate

1. main merge commit으로 immutable Speech API image를 build합니다.
2. 현재 CPU 4·8GiB, min 0, max 1, concurrency 4를 바꾸지 않은 candidate revision부터
   배포합니다.
3. 제한된 합성 WAV의 warm sequence와 5×5 burst를 한 번만 실행합니다.
4. `speech_resource_sample`의 numeric allowlist와 Cloud Run request log를 request ID로
   대조합니다.
5. image digest·revision·입력 SHA-256·계측 source SHA-256·집계 보고서 SHA-256을 기록합니다.
6. OOM·5xx·민감정보 log가 있거나 cgroup limit가 배포 설정과 다르면 후보를 기각합니다.

## 2026-09-08 실제 개발용 preview 결과

| 항목 | 관찰값 | 해석 한계 |
|---|---:|---|
| revision | `chemicheck119-speech-api-preview-tsfix` | 상용 revision 아님 |
| image digest | `sha256:6788bbd3b6ee061457c2015bcb5a3f46ddee42dc08446777b5283a341bd00bd5` | immutable image 식별자 |
| 입력 | 30초 공개 합성 파생물 | 실제 신고전화·현장 무전 아님 |
| protocol | 동시 2요청 × 3 batch | 장시간·대규모 부하 아님 |
| 응답 | HTTP 200 3건 + 앱 429 3건 | 플랫폼 429·5xx 0건 |
| 성공 RTF | median 0.2578, max 0.2917 | 정확도 지표 아님 |
| cgroup current | median 1.2947GiB, max 1.2949GiB | 관측 시점 값, peak 아님 |
| process current RSS | median 1.4354GiB, max 1.4356GiB | cgroup과 직접 차감 금지 |
| process max RSS | median 1.5360GiB, max 1.5363GiB | process 생명주기 high-water mark |
| cgroup limit | 8GiB | 배포 설정과 일치 |
| cgroup peak | 관측 불가 | Cloud Run의 v1 경로에 counter가 없었음 |

성공 request ID 3건과 resource event 3건이 모두 일치했고 numeric allowlist·counter 범위·8GiB
limit·원음 및 전사문 미수집을 포함한 안전 Gate 13개가 모두 통과했습니다. 비공개 집계
보고서 SHA-256은
`fd3cff29c56c3bbb5bd6322f4b8cf37ec94c0f2f6463718c56463f4424221f9f`입니다.

첫 30초 실행은 모델 마지막 segment end가 30.26초로 입력 30.00초를 0.26초 초과해 API의
기존 0.10초 경계에서 502로 중단됐습니다. PR #52에서 최대 0.5초의 마지막 end 초과만 입력
길이로 제한하고 더 큰 초과는 계속 fail-closed 처리했습니다. 전사문은 바꾸지 않았고 전체
unittest 109개 통과 뒤 같은 입력으로 재검증했습니다.

결정은 **계측 방법과 이번 수치 채택**, **8GiB 축소 보류**입니다. cgroup peak를 얻지 못했고
입력 다양성도 없으므로 다음 축소 후보는 별도 revision·동일 protocol·rollback 경계에서만
비교합니다. 자세한 실행 근거는 infra PR #34와 infra #27에 있습니다.

8GiB 축소 실험은 실제 peak가 확인된 뒤 별도 revision에서 수행합니다. 축소 후보는 같은 image,
같은 입력, 같은 request protocol을 사용하고 cold start·warm RTF·E2E tail·429·OOM을 함께
비교합니다. peak 한 건만 보고 바로 메모리를 줄이지 않습니다.

## GPU 판단과의 관계

memory 관측은 GPU 채택 실험이 아닙니다. GPU는 다음 중 하나가 사전 정의된 평가에서 반복될
때만 별도 A/B 대상으로 올립니다.

- 장문 또는 모의 통신 왜곡 입력에서 CPU RTF가 목표를 넘음
- Backend Gate를 적용한 뒤에도 모델 계산시간이 latency SLO의 주된 실패 원인임
- 검증된 Whisper LoRA 후보가 CPU service 범위를 벗어남

GPU A/B에서도 CER·WER뿐 아니라 우선용어 F1, false insertion, STT→Resolver Top-3,
잘못된 단일 CAS 승격 0건을 함께 확인합니다. 속도만 빨라졌다는 이유로 모델을 채택하지
않습니다.
