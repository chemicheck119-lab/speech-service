# 케미체크119 Speech Service

신고·무전 음성을 텍스트와 구간별 신뢰 정보로 변환하는 독립 ASR(Automatic Speech Recognition) 서비스입니다.

## 책임

```text
음성 입력
→ 포맷 검증·전처리·VAD
→ faster-whisper 전사
→ 화학용어 보존·후처리
→ 전사문·구간·타임스탬프·신뢰 정보 출력
```

- 이 저장소는 음성을 전사하지만 물질의 CAS를 확정하거나 위험을 판단하지 않습니다.
- Parser·Resolver·Retriever·Agent·CAMEO 규칙은 `analysis-engine`의 책임입니다.
- 인증·사고 상태·감사 기록은 `back`의 책임입니다.
- 원본 음성과 모델 가중치는 Git에 저장하지 않습니다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 저장소·CI 골격 | 구현 완료 |
| faster-whisper 1.2.1 배치 평가 하네스 | 구현 완료 |
| AIHub 신고음성 ZIP 로딩·고정 77건 평가 | 구현·실행 완료 |
| 기본 전사 vs hotword 힌트 A/B | 측정 완료·hotword 기본값 제외 |
| `radio-sim-v1` paired 강건성 평가 실행기 | 부분 구현 또는 개발용 데모 |
| 서울·인천 신고음성·모의 통신 왜곡 평가 | 구현·실행 완료(실제 현장 무전 아님) |
| 실시간 스트리밍 API·패드 연동 | 설계·구현 전 |
| Whisper tokenizer·data preflight | 구현·실행 완료 |
| 제한 LoRA local MPS 학습 harness | 구현 완료·1차 FP16 실행 수치 불안정으로 기각 |
| LoRA numeric smoke Gate | full FP32 2-step 실행·통과 |
| LoRA A/B/C 변환 pipeline | 구현 완료·유효 adapter 부재로 실행 전 |
| LoRA A/B/C 잠금 평가 판정기 | 구현 완료·유효 adapter 부재로 실행 전 |
| LoRA A/B/C 잠금 평가 runner | 구현 완료·유효 adapter 부재로 실행 전 |
| 화학용어 사후 자동교정 | 미구현; 원문 보존 원칙상 현재 범위 제외 |
| 현장 무전 성능 | 검증되지 않음 |

## 고정 비교실험

두 조건은 `small`, beam 5, 한국어 고정, VAD 사용, 이전 문맥 비사용으로 동일합니다. B 조건에만 사전 등록한 우선 용어를 faster-whisper의 `hotwords` 인자로 제공합니다. 평가 순서는 레코드마다 교차해 순서 편향을 줄입니다.

- 주 지표: 정규화 CER, 보조 지표: WER
- 안전 관련 지표: 우선 용어 presence recall·precision·false insertion
- 실행 지표: 처리시간과 RTF(real-time factor)
- 불확실성: 레코드 단위 paired bootstrap 95% 구간, seed 119
- 고정 평가셋: AIHub 광주 화재 Validation 77건

`--limit`를 사용한 실행은 항상 `development smoke`로 기록됩니다. 77건 전체가 확인된 경우에만 고정 평가 ID를 부여합니다.

AIHub 자료는 신고접수 전화 음성입니다. 이 결과를 잡음·통신 왜곡이 다른 현장 무전기의 성능으로 표현하면 안 됩니다. 구간의 `avg_log_probability`도 정답 확률이 아닌 보정되지 않은 디코딩 점수입니다.

```bash
python -m pip install .
chemicheck119-speech-eval \
  --audio-archive /secure/VS_광주_화재.zip \
  --label-archive /secure/VL_광주_화재.zip \
  --dataset-manifest /secure/aihub-71768-gwangju-fire-validation.json \
  --hotwords-file config/domain_hotwords.txt \
  --model small --device cpu --compute-type int8 \
  --output-dir outputs
```

`records.private.jsonl`에는 참조·가설 전사문이 들어가므로 비공개 버킷에만 저장합니다. 콘솔에는 진행 건수와 최종 위치만 출력합니다.

고정 평가 ID는 manifest의 데이터 버전·77건 선언과 두 ZIP의 SHA-256이 모두 일치할 때만 부여됩니다. ZIP 멤버는 WAV 32MiB, 라벨 4MiB, 압축비 200배, 음성 300초로 제한합니다. GPU 초기화가 실패하면 CPU int8로 안전하게 전환하고 실제 device와 오류 유형을 결과에 기록합니다.

고정 77건 평가에서는 기본 전사의 CER가 43.75%, hotword 조건이 56.28%였습니다. hotword는 우선 용어 재현율을 높였지만 실제로 없던 용어 삽입을 크게 늘려 기본값에서 제외했습니다. 수치의 정의, 실행 해시, 제한 사항은 [AIHub 광주 화재 음성 평가 보고서](docs/AIHUB_광주_화재_음성_평가.md)에 기록했습니다.

## 교차지역 기준선 평가

서울·인천 화재 Validation은 광주에서 고정한 설정을 바꾸지 않고 `baseline`만 실행합니다.
지역별 manifest의 ID·건수·archive SHA-256을 검증하며, 광주에서 기각한 hotword를 다시
실행해 비용과 선택 편향을 늘리지 않습니다. 우선용어 파일은 평가 지표 계산에만 사용됩니다.

```bash
chemicheck119-speech-eval \
  --audio-archive gs://PRIVATE_BUCKET/raw/aihub/71768/seoul-fire/VS_서울_화재.zip \
  --label-archive gs://PRIVATE_BUCKET/raw/aihub/71768/seoul-fire/VL_서울_화재.zip \
  --dataset-manifest gs://PRIVATE_BUCKET/manifests/aihub-71768-seoul-fire-validation.json \
  --hotwords-file config/domain_hotwords.txt \
  --variants baseline \
  --model small --device cpu --compute-type int8 \
  --output-dir outputs
```

교차지역 결과도 신고접수 전화 성능이며 현장 무전 성능이 아닙니다. 자세한 사전 계획과
채택 기준은 [교차지역 STT 평가 계획](docs/교차지역_STT_평가_계획.md)에 기록합니다.

세 지역의 고정 summary가 모두 준비된 뒤 다음 명령으로 설정 일치와 사전 등록 게이트를
재현합니다. 이 보고서는 서로 다른 지역의 CER·WER 차이를 paired 변화로 해석하지 않고,
집계값만으로 LoRA 실행을 결정하지 않습니다.

```bash
chemicheck119-speech-capture-runtime-provenance \
  --project PROJECT_ID --region asia-northeast3 \
  --gwangju-execution chemicheck119-speech-eval-cpu-EXECUTION_ID \
  --gwangju-summary /private/gwangju/summary.json \
  --incheon-execution chemicheck119-speech-cross-region-cpu-EXECUTION_ID \
  --incheon-summary /private/incheon/summary.json \
  --seoul-execution chemicheck119-speech-seoul-cpu-EXECUTION_ID \
  --seoul-summary /private/seoul/summary.json \
  --output /private/cross-region-runtime-provenance.json

chemicheck119-speech-cross-region-report \
  --gwangju-summary /private/gwangju/summary.json \
  --incheon-summary /private/incheon/summary.json \
  --seoul-summary /private/seoul/summary.json \
  --runtime-provenance /private/cross-region-runtime-provenance.json \
  --evaluator-git-commit "$(git rev-parse HEAD)" \
  --output /private/cross-region-report.json
```

`runtime-provenance`에는 지역별 Cloud Run execution 이름, 성공 상태, 시작·완료 시각,
immutable container image digest와 summary SHA-256만 기록합니다. 평가기는 인천·서울이
동일 image digest를 사용했고 각 summary가 해당 execution snapshot에 결합됐을 때만
보고서를 만듭니다. 광주는 이미 관찰한 legacy 기준선이므로 image가 달라도 runtime 설정이
같을 때 비교 기준으로만 사용합니다. 수집기는 완료·성공한 execution만 허용하고 `gcloud`
응답을 메모리에서 allowlist 필드로 축소한 뒤 결과를 `0600` 권한으로 새로 생성합니다.
원본 annotation·환경변수·계정 정보는 파일에 기록하지 않으며 기존 결과는 덮어쓰지 않습니다.

비공개 레코드의 반복 오류를 확인할 때는 원문 대신 공개 우선용어별 누락·오삽입과 길이·
CER·WER·RTF 분포만 집계합니다. 모델의 `avg_log_probability` 등은 보정된 정확도 확률이
아닙니다. 이 보고서의 CER·WER 평균은 레코드별 macro 평균이며, 기본 평가 요약의
문자·단어 수 가중 corpus CER·WER와 서로 다른 집계입니다.

```bash
chemicheck119-speech-failure-analysis \
  --records-private /private/incheon/records.private.jsonl \
  --summary /private/incheon/summary.json \
  --priority-terms config/domain_hotwords.txt \
  --evaluator-revision "$(git rev-parse HEAD)" \
  --output /private/incheon/failure-analysis.json
```

## 모의 통신 왜곡 강건성 평가

`data-pipeline`의 고정 `radio-sim-v1` 실행이 만든 clean 대조군과 17개 왜곡 조건을 동일
레코드끼리 비교합니다. 일부 조건만 빠진 실행은 공식 강건성 평가로 받지 않으며, 원본
manifest·우선용어 목록·파생 manifest·audio/label archive의 SHA-256을 모두 확인한 뒤에만
모델을 초기화합니다.

```bash
chemicheck119-speech-robustness-eval \
  --run-summary gs://PRIVATE_BUCKET/derived/aihub/71768/seoul-fire/radio-sim-v1/run-summary.json \
  --simulation-root gs://PRIVATE_BUCKET/derived/aihub/71768/seoul-fire/radio-sim-v1 \
  --priority-terms config/domain_hotwords.txt \
  --model small --device cpu --compute-type int8 \
  --max-total-audio-hours 24 \
  --output-dir outputs/seoul-radio-sim-v1
```

실행 전 18개 archive의 총 음성시간을 계산하며 24시간을 넘으면 모델 로딩 전에
중단합니다. 이는 계획한 4 vCPU·최대 6시간 평가와 전체 70,000원 비용 한도를 지키기 위한
사전 방어선입니다. 현재 배포된 기존 평가 Job의 timeout은 2시간이므로 실제 실행 전에
별도 강건성 Job을 만들거나 표본 수를 줄여야 합니다. 결과에는 조건별
CER·WER·RTF·우선용어 지표와 clean 대비
paired CER bootstrap 구간, WER·용어 F1·false insertion 변화가 기록됩니다.
`records.private.jsonl`에는 참조·가설 전사문이 있으므로 비공개 GCS에만 저장합니다.
archive 한 개는 512MiB, 전체 materialize는 4GiB로 별도 제한합니다.

이 평가는 **AIHub 신고 전화의 모의 통신 왜곡 강건성**일 뿐 실제 현장 무전 검증이
아닙니다. STT 결과만으로 CAS를 확정하거나 Rule Engine을 실행할 수도 없습니다. 세부
게이트는 [모의 통신 왜곡 STT 평가 계획](docs/모의_통신_왜곡_STT_평가_계획.md)을 따릅니다.

### 서울·인천 실행 provenance

두 지역의 Cloud Run execution 완료 상태·고정 Job 이름·container digest와 STT summary
SHA-256을 결합합니다. 동일 runtime·우선용어 목록과 서로 다른 source manifest 여부만
검사하며, 이 서비스에서 Parser·Resolver 안전 지표나 최종 LoRA 결정을 만들지 않습니다.

```bash
chemicheck119-speech-capture-radio-sim-provenance \
  --project chemi-check \
  --gcp-region asia-northeast3 \
  --incheon-summary /private/incheon/summary.json \
  --incheon-execution INCHEON_EXECUTION \
  --seoul-summary /private/seoul/summary.json \
  --seoul-execution SEOUL_EXECUTION \
  --collector-git-commit "$(git rev-parse HEAD)" \
  --output /private/radio-sim-runtime-provenance.json
```

최종 Gate는 이 provenance와 동일 Model API runtime으로 생성한 두 downstream 보고서를
`analysis-engine`의 단일 비교기에 입력해 판정합니다. 서울·인천 결과는 학습·튜닝에 사용할
수 없고, Gate 통과도 LoRA의 성능 개선이나 채택을 뜻하지 않습니다.

### Whisper LoRA data preflight

광주 Training에서 만든 immutable clean·`wind_snr0` train/dev artifact는 학습 전에 전체
archive·manifest SHA-256, partition membership, record 중복, utterance 길이, 60:40 조건
배정을 다시 검증합니다. 결과는 집계값만 포함하며 원문 전사·recordId·주소를 기록하지
않습니다.

```bash
chemicheck119-speech-lora-data-preflight \
  --execution-config config/whisper_lora_execution_v1.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --artifact-root /secure/gwangju-lora-artifacts-v1 \
  --output /secure/lora-data-preflight.json
```

화자·사고 overlap은 원본 ID 부재로 검증할 수 없고 tokenizer token 상한도 GPU runtime에서
확인해야 하므로 `status=limited`입니다. 이 결과는 LoRA 학습, 성능 개선, 현장 무전 성능을
승인하지 않으며 `automatic_training_allowed=false`를 유지합니다.

고정 `openai/whisper-small` tokenizer로 별도 검사를 실행하면 모든 label을 truncation 없이
세되, 결과에는 구간별 원문이나 recordId를 남기지 않습니다. 현재 광주 train/dev artifact의
최대치는 58 tokens였고 160-token 상한 초과는 0건입니다. 이 수치는 학습 적합성 중
token-length 조건만 확인하며 모델 개선을 뜻하지 않습니다.

```bash
chemicheck119-speech-lora-tokenizer-preflight \
  --execution-config config/whisper_lora_execution_v1.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --artifact-root /secure/gwangju-lora-artifacts-v1 \
  --output /secure/lora-tokenizer-preflight.json
```

### 제한 LoRA local MPS 학습 harness

학습 harness는 학습만 수행하고 adapter를 `trained_unvalidated`로 저장합니다. 성능 평가,
base 병합, CTranslate2 변환, 운영 채택·배포는 수행하지 않습니다. 실행 전 다음 조건을 모두
fail-closed로 확인합니다.

- 등록된 config·data artifact SHA-256과 tokenizer 160-token 상한
- 소유한 M4·24GB의 Python 3.11, PyTorch 2.9.x, arm64·MPS
- 24시간 이내의 증분 서버비 0원 확인서와 70,000원 전체 비용 상한
- record 단위 고정 60:40 clean/`wind_snr0` 배정과 발화 1회 학습
- commit-bound 단일 사용 authorization과 원격 원자적 claim
- 명시적 1회 실행 확인문, 42,600초 내부 deadline, retry 0, CPU fallback 금지

Compute Engine은 서울·도쿄 T4 네 zone의 재고 부족 뒤 서울 L4로 변경했지만, 신설 프로젝트의
global GPU quota 0과 quota 요청 자동 거절로 차단됐습니다. 현재 profile은 이미 소유한 M4를
사용하므로 추가 server·disk·network 비용은 0원입니다. 그래도 실행 중복을 막기 위해
`whisper-small-lora-cost-quote-v1` 확인서와 GCS remote claim을 유지합니다.

```bash
scripts/run_whisper_lora_mps_once.sh \
  /private/venv/bin/python \
  /private/gwangju-lora-artifacts-v1 \
  /private/current-local-cost-quote.json \
  /private/training-run-UNIQUE
```

runner 내부 명령은 다음 계약을 고정합니다.

```bash
chemicheck119-speech-lora-train \
  --execution-config config/whisper_lora_execution_v1.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --artifact-root /secure/gwangju-lora-artifacts-v1 \
  --cost-quote /secure/current-cost-quote.json \
  --authorization-claim /secure/authorization-claim.json \
  --output-dir /secure/training-run-UNIQUE \
  --confirm-bounded-experiment RUN_BOUNDED_LORA_ONCE \
  --runner-revision SPEECH_SERVICE_MERGE_COMMIT_SHA
```

local runner는 현재 main commit과 clean worktree만 허용하고 `caffeinate`로 sleep을 방지합니다.
42,900초에 `TERM`, 60초 뒤 강제 종료하는 watchdog을 두며 같은 authorization을 다시 쓰면
학습 전에 실패합니다.

원본 full-call WAV는 선택된 조건에서 한 번만 권한 `0600` 임시 공간으로 풀고, 발화
timestamp 구간만 8kHz→16kHz로 재표본화해 학습합니다. 임시 음성과 Trainer 작업 디렉터리는
정상 종료 또는 `TERM` 시 제거합니다. runner는 예측 가능한 private staging 경로만 추가로
정리합니다. 결과 보고서에는 aggregate loss·속도·artifact hash만 남고 전사문·주소·recordId는
포함하지 않습니다. 2026-09-07 첫 FP16 MPS 실행은 25 step에서 `loss=12157.4975`,
`grad_norm=NaN`을 보여 30 step에서 중단·기각했습니다. adapter와 training report는 생성되지
않았고 추가 서버 비용은 0원입니다. 이 실패는 LoRA 자체의 효과를 기각한 결과가 아니라 해당
수치 설정의 전체 실행을 기각한 결과입니다. 뒤이은 FP32 master weight·FP16 autocast smoke도
2-step optimizer 직전 비유한 gradient를 감지해 기각했습니다. FP16 autocast·GradScaler를
모두 끈 full FP32 smoke가 유한 loss·gradient와 LoRA parameter update를 확인하기 전에는
재실행하지 않습니다.

numeric smoke는 full FP32 외에는 전체 학습과 같은 batch·gradient accumulation·learning
rate로 optimizer 2 step만 실행합니다. 각 step에서 유한 gradient와 parameter를 검사하고,
loss가 비유한 값이거나 100을 넘으면 즉시 실패합니다. LoRA tensor가 실제 변경되고 표본 base
tensor가 그대로인 것도 확인합니다. 성공 보고서도 전체 학습을 자동 허용하거나 성능 향상을
증명하지 않습니다.

2026-09-07 full FP32 실제 실행에서는 두 loss가 10.8512·11.3878, gradient norm이
9.3629·8.8864로 모두 유한했습니다. LoRA tensor 144개가 모두 변경됐고 표본 frozen base
tensor는 유지됐습니다. 따라서 **full FP32 수치 안정성 Gate만 채택**했습니다. report
SHA-256은 `1538dacd6eba183c41f6db736647dde1ce66a01b039603c0679799cd6928617d`이며,
다음 단계는 새 single-use authorization을 사용하는 전체 1회 학습입니다.

```bash
scripts/run_whisper_lora_mps_once.sh \
  /private/venv/bin/python \
  /private/gwangju-lora-artifacts-v1 \
  /private/current-local-cost-quote.json \
  /private/numeric-smoke-UNIQUE \
  numeric-smoke
```

### LoRA A/B/C 변환 pipeline

유효한 `trained_unvalidated` adapter가 생긴 뒤에만 B와 C를 같은 CTranslate2 4.8.2·FP16
변환 설정으로 생성합니다. A는 현재 운영 기준선, B는 고정 OpenAI base의 재변환 control,
C는 같은 base에 adapter를 안전 병합한 candidate입니다. 변환 효과와 LoRA 효과를 분리하기
위해 C는 우선 B와 비교합니다.

변환기는 training report와 config hash, 모든 adapter artifact hash, 추가 서버 비용 0원,
학습 commit 및 변환기 commit을 확인·기록합니다. 변조되거나 symlink인 입력, 기존 output,
8GiB 미만 여유 공간은 거부합니다. 변환 성공도 정확도 향상이나 채택을 뜻하지 않습니다.

```bash
chemicheck119-speech-lora-convert \
  --training-dir /private/training-run-UNIQUE \
  --execution-config config/whisper_lora_execution_v1.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --output-dir /private/conversion-run-UNIQUE \
  --converter-revision "$(git rev-parse HEAD)"
```

adapter가 생겨도 잠금 dev와 downstream 안전평가 전 사실 상태는 **부분 구현 또는 개발용
데모**입니다.

### LoRA A/B/C 잠금 평가

A·B·C를 각각 같은 광주 화재 Validation 77건, `baseline` 단일 조건, CPU int8로 실행한 뒤
private record와 summary를 함께 판정기에 전달합니다. 판정기는 dataset fingerprint와 77개
record pairing, reference·audio duration, 추론 설정, model arm 결합을 확인하고 private
record에서 CER·WER·우선용어 지표를 다시 계산합니다.

```bash
chemicheck119-speech-lora-abc-report \
  --conversion-report /private/conversion-run/conversion-report.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --priority-terms config/domain_hotwords.txt \
  --a-summary /private/A/summary.json --a-records /private/A/records.private.jsonl \
  --b-summary /private/B/summary.json --b-records /private/B/records.private.jsonl \
  --c-summary /private/C/summary.json --c-records /private/C/records.private.jsonl \
  --output /private/abc-locked-evaluation.json \
  --evaluator-revision "$(git rev-parse HEAD)"
```

A↔B에서는 converter drift, B↔C에서는 LoRA effect를 분리하고 paired bootstrap CER·WER를
계산합니다. clean 회귀·false insertion Gate를 통과해도 결과는 wind·downstream Gate로
진행할 자격일 뿐이며 자동 채택을 허용하지 않습니다. 원본 전사문이 있는 세 records 파일은
권한 `0600`인 비공개 경로만 허용합니다.

clean main commit에서 세 arm을 순차 실행하고 판정까지 완료하려면 다음 runner를 사용합니다.
동일한 Validation archive·manifest·priority terms, `baseline`, CPU int8, local model만
허용하며 기존 output은 덮어쓰지 않습니다.

```bash
scripts/run_whisper_lora_abc_locked_once.sh \
  /private/venv/bin/python \
  /private/conversion-run \
  /private/models/Systran--faster-whisper-small/PINNED_REVISION \
  /private/gwangju-fire-validation \
  /private/abc-evaluation-UNIQUE
```

## 기본 검증

```bash
python -m compileall -q src tests
PYTHONPATH=src python -m unittest discover -s tests -v
```
