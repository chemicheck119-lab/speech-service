# Whisper LoRA 제한 실험 계획

기준일: 2026-09-06

## 목표와 현재 상태

- 목표: 광주 Training 내부 dev에서 clean 성능과 오삽입을 지키면서 `wind_snr0`의 `연기`
  누락을 줄일 수 있는지 검증
- 현재 상태: data·tokenizer preflight와 GPU harness는 **구현 완료**, GPU 학습은 **실행 전**
- 실행 허용: 현재 가격표와 고정 T4 runtime 검증 뒤 명시적 1회 실행만 허용하며 자동 실행은 금지
- 데이터 범위: AIHub 신고전화와 절차적 모의 왜곡, 실제 현장 무전 아님

서울·인천 `radio-sim-v1`에서 같은 공개 용어가 반복 누락되어 LoRA **실험 설계** Gate가
열렸습니다. 이는 LoRA가 개선된다는 증거나 학습 자동 승인과 다릅니다.

## 비교군

1. A: 현재 `Systran/faster-whisper-small` 운영 기준선
2. B: 고정한 `openai/whisper-small`을 후보와 같은 도구로 새로 변환한 base control
3. C: LoRA adapter를 base에 병합한 뒤 B와 같은 도구로 변환한 candidate

B를 두는 이유는 model revision·CTranslate2 converter 차이와 LoRA 효과를 분리하기
위해서입니다. C의 개선은 우선 B와 비교하고, A 대비 운영 회귀도 따로 확인합니다.

## 학습 데이터와 설정

- 광주 Training: train 527건, dev 132건
- 같은 `recordId`의 모든 발화·clean·파생본은 같은 partition
- stable speaker ID 없음, cross-record incident ID 없음
- train record의 60%는 clean, 40%는 `wind_snr0`; seed 9119로 한 번 고정
- 각 발화는 1 epoch에서 한 번만 사용
- Whisper small + PEFT LoRA(`q_proj`, `v_proj`, rank 8, alpha 16)
- 서울·인천·광주 Validation은 학습 및 하이퍼파라미터 선택 금지

LoRA는 전체 weight를 다시 학습하지 않고 attention projection에 작은 저랭크 행렬만
학습합니다. 비용과 저장량은 줄지만 계산 자체가 사라지지 않고, 작은 데이터에 과적합할
가능성도 있으므로 full fine-tuning의 자동 대체제가 아닙니다.

## 채택 기준

- wind CER 또는 WER 상대 5% 이상 개선, 해당 paired bootstrap 95% 상한 0 미만
- wind `연기` Recall 절대 10%p 이상 개선
- wind 전체 우선용어 F1 절대 3%p 이상 개선
- clean CER 회귀 1%p 이하, WER 회귀 1.5%p 이하
- clean·wind false insertion 증가 0
- current operational baseline과 새 conversion control의 CER·WER 차이 각각 0.5%p 이하
- RTF 0.5 이하
- downstream 실버 Top-3 비회귀
- 잘못된 단일 CAS 승격·2-CAS Gate 조기 실행 각각 0

위 조건을 모두 만족해도 untouched 지역이 없으므로 최대 상태는 **조건부 채택·development
preview 전용**입니다. 기본 운영 모델 교체와 현장 안전성 주장은 허용하지 않습니다.

## 비용 Gate

- 서울 두 zone의 재고 부족 확인 후 도쿄 리전 standard T4 1장, `n1-standard-4`, 최대 3시간
- instance 1, retry 0, 실험 hard cap 20,000원
- 전체 추가 개발 서버 비용 70,000원 이내
- 실행 직전 현재 SKU·누적 비용·quota·artifact 잔존 여부를 다시 확인

## preflight

```bash
chemicheck119-speech-lora-preflight \
  --config config/whisper_lora_experiment_v1.json \
  --split-manifest /secure/aihub-71768-gwangju-fire-training-lora-split-v2.json \
  --audio-archive /secure/TS_광주_화재.zip \
  --label-archive /secure/TL_광주_화재.zip \
  --priority-terms config/domain_hotwords.txt \
  --output /secure/lora-preflight.json
```

결과는 화자·사고 중복을 측정할 ID가 없어 `status=limited`이며,
`automatic_training_allowed=false`를 유지합니다. 다음 Gate는 reviewed training
harness, immutable clean/wind artifact, 실행 직전 비용 견적입니다.

## immutable data artifact preflight

```bash
chemicheck119-speech-lora-data-preflight \
  --execution-config config/whisper_lora_execution_v1.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --artifact-root /secure/gwangju-lora-artifacts-v1 \
  --output /secure/lora-data-preflight.json
```

이 검사는 네 audio archive와 두 label archive, private ledger, 네 manifest의 SHA-256과
크기를 확인합니다. 또한 train/dev `recordId` 중복 0, clean·`wind_snr0` membership 동일성,
12초 발화 상한, seed 9119의 60:40 record 배정, 각 utterance 1회 선택을 집계값으로
검증합니다. 전사문·주소·recordId는 보고서나 콘솔에 남기지 않습니다.

화자와 cross-record 사고 ID가 없어 해당 overlap은 `not_evaluated`이고, pinned Whisper
tokenizer 검사에서도 이 한계는 해소되지 않습니다. 따라서 결과 상태는 `limited`, 자동 학습
허용은 `false`입니다.

## tokenizer preflight 실제 결과

- 고정 tokenizer: `openai/whisper-small` revision
  `973afd24965f72e36ca33b3055d56a652f456b4d`
- 광주 train/dev 최대 label 길이: 58 tokens
- 등록된 160-token 상한 초과: 0건
- 허용 주장: label token-length 학습 준비도
- 금지 주장: LoRA 성능 개선, 현장 무전 정확도, 안전성

## GPU 학습 실행 Gate

학습 harness는 다음 순서로 실패 폐쇄합니다.

1. config와 비공개 data artifact hash를 다시 검증합니다.
2. 24시간 이내 가격표를 항목별로 재계산하고 실험 20,000원·전체 70,000원 상한을 확인합니다.
3. Python 3.12·CUDA 12.9·PyTorch 2.9.x·고정 package·단일 T4만 허용합니다.
4. record별 clean 60% / `wind_snr0` 40%를 seed 9119로 선택하고 각 발화를 한 번만 사용합니다.
5. 1 epoch 뒤 adapter와 processor, 집계 전용 보고서만 비공개 경로에 원자적으로 저장합니다.
6. 임시 음성·Trainer 파일은 성공과 실패 모두 제거합니다.

```bash
timeout --signal=TERM --kill-after=60s 9900 chemicheck119-speech-lora-train \
  --execution-config config/whisper_lora_execution_v1.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --artifact-root /secure/gwangju-lora-artifacts-v1 \
  --cost-quote /secure/current-cost-quote.json \
  --authorization-claim /secure/authorization-claim.json \
  --output-dir /secure/training-run-UNIQUE \
  --confirm-bounded-experiment RUN_BOUNDED_LORA_ONCE \
  --runner-revision SPEECH_SERVICE_MERGE_COMMIT_SHA
```

견적 authorization은 정확한 speech commit과 1회 실행에 결합합니다. GPU runner는 같은 ID의
원격 claim을 GCS에 원자적으로 처음 생성한 실행만 허용합니다. 내부 Python deadline은
9,600초, 외부 process timeout은 9,900초, VM 자동삭제는 10,800초로 계층화합니다. `TERM`은
Python cleanup을 실행하고, 60초 뒤 강제 kill에도 auto-delete boot disk가 임시 음성을 남기지
않습니다. 재실험 견적은 직전 독립 비용 ceiling을 누적해야 하며 전체 70,000원을 넘으면
거부됩니다.

학습 성공 직후 상태도 `trained_unvalidated`, 사실 상태는 **부분 구현 또는 개발용 데모**입니다.
A/B/C 변환·잠금 dev·downstream 안전 Gate를 모두 통과하기 전에는 정확도 향상이나 채택을
주장하지 않습니다. 리전 간 data transfer $0.25 ceiling을 포함한 3시간 비용 상한은
9,032원이며, 이전 개발비 ceiling 50,000원을 더한 전체 상한은 59,032원입니다. 실행 직전 실제 견적은
이와 별도로 생성·해시 고정합니다.
