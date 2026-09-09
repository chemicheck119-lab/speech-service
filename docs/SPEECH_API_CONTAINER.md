# Speech API container runtime

## 사실 상태

**부분 구현 또는 개발용 데모**입니다. 별도 API image, non-root·read-only smoke,
offline model loading과 인증 실패 경계는 구현했습니다. 실제 Cloud Run 배포, IAM, Secret
Manager, cold start, 동시 부하와 현장 음성은 아직 검증하지 않았습니다.

기존 `Dockerfile`은 batch 평가 Job 전용입니다. `Dockerfile.api`는 bounded Speech API
서비스 전용이며 두 목적을 암묵적으로 전환하지 않습니다.

## production image

production build는 `Systran/faster-whisper-small`의 revision
`536b0662742c02347bc0e980a01041f333bce120`을 build 단계에서 image에 포함합니다.
`model.bin` SHA-256이 고정값과 다르면 build를 중단합니다. runtime은 manifest와 실제
파일을 다시 대조하며 `local_files_only=true`라서 시작 중 외부 model registry에 접속하지
않습니다.

```bash
docker build \
  --file Dockerfile.api \
  --build-arg EMBED_WHISPER_MODEL=true \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --tag asia-northeast3-docker.pkg.dev/PROJECT/REPOSITORY/speech-api:COMMIT \
  .
```

API key는 build argument나 image layer에 넣지 않습니다. Cloud Run에서는 Secret Manager로
`CHEMICHECK119_SPEECH_API_KEY`를 주입하고 ingress와 IAM을 함께 제한해야 합니다.

## local boundary smoke

model을 포함하지 않는 smoke image는 liveness가 200이어도 readiness가 503이어야 합니다.
이는 “서버 process가 켜짐”과 “검증된 model이 준비됨”을 구분하는 fail-closed 검사입니다.

```bash
bash scripts/smoke_speech_api_container.sh
```

로컬 Docker가 없는 환경을 위해 `Speech API Container` workflow는 container 관련 파일이
바뀐 PR/push에서만 같은 smoke를 실행합니다. 일반 문서·평가 코드 변경에는 실행하지 않아
기본 CI 시간을 늘리지 않습니다.

검사 항목:

- container UID 10001
- read-only root filesystem + 제한된 tmpfs
- model 미포함 시 `/health/live` 200, `/health/ready` 503
- API key 없는 전사 요청 401
- 원본 음성·모델 weight·private data가 build context에 들어가지 않음

smoke의 placeholder key는 실제 Secret이 아니며 image에 저장되지 않습니다. 이 smoke는
실제 faster-whisper 추론 성공이나 정확도·처리량을 검증하지 않습니다.

## Cloud Run 목표 경계

| 항목 | 목표 구성 | 현재 상태 |
|---|---|---|
| ingress | internal 또는 load balancer 제한 | 설계 완료·구현 전 |
| invoker | BE service account만 허용 | 설계 완료·구현 전 |
| API key | Secret Manager runtime 주입 | 설계 완료·구현 전 |
| instance | CPU 4, memory 실측 후 고정, max instances 1 | 검증되지 않은 가설 |
| concurrency | service 1, application semaphore 1 | 설계 완료·구현 전 |
| min instances | 0으로 비용 제한 | 설계 완료·구현 전 |
| model | image digest에 포함, runtime download 금지 | container 구현·배포 전 |

CPU·memory·timeout은 실제 model image의 cold/warm 측정 전 확정값으로 주장하지 않습니다.
배포 전 Artifact Registry image 크기와 Cloud Run request 최대시간, startup probe, 월 비용
상한을 다시 계산해야 합니다.
