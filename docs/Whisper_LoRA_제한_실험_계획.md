# Whisper LoRA 제한 실험 계획

기준일: 2026-09-06

## 목표와 현재 상태

- 목표: 광주 Training 내부 dev에서 clean 성능과 오삽입을 지키면서 `wind_snr0`의 `연기`
  누락을 줄일 수 있는지 검증
- 현재 상태: data·tokenizer preflight와 local MPS harness는 **구현 완료**, 첫 FP16 MPS
  실행은 수치 불안정으로 **기각**, FP32 원인 분리 smoke는 실행 전
- 실행 허용: 증분 서버비 0원 확인과 고정 MPS runtime 검증 뒤 명시적 1회 실행만 허용
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
- Whisper small + generic PEFT LoRA(`q_proj`, `v_proj`, rank 8, alpha 16)
- 서울·인천·광주 Validation은 학습 및 하이퍼파라미터 선택 금지

LoRA는 전체 weight를 다시 학습하지 않고 attention projection에 작은 저랭크 행렬만
학습합니다. 비용과 저장량은 줄지만 계산 자체가 사라지지 않고, 작은 데이터에 과적합할
가능성도 있으므로 full fine-tuning의 자동 대체제가 아닙니다.
PEFT의 text seq2seq wrapper는 `input_ids`를 전달하지만 Whisper encoder는
`input_features`를 받으므로 task-specific wrapper 대신 generic forwarding을 고정합니다.

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

- GCP global GPU quota 0과 quota 요청 거절로 Compute Engine GPU 경로는 blocked
- 소유한 M4·24GB, MPS process 1개, retry 0, 최대 12시간
- 현재 실행의 증분 서버 비용 0원
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

## local MPS 학습 실행 Gate

학습 harness는 다음 순서로 실패 폐쇄합니다.

1. config와 비공개 data artifact hash를 다시 검증합니다.
2. 24시간 이내 확인서로 증분 서버비 0원과 전체 70,000원 상한을 확인합니다.
3. Python 3.11·arm64·MPS·PyTorch 2.9.x·고정 package만 허용합니다.
4. record별 clean 60% / `wind_snr0` 40%를 seed 9119로 선택하고 각 발화를 한 번만 사용합니다.
5. 1 epoch 뒤 adapter와 processor, 집계 전용 보고서만 비공개 경로에 원자적으로 저장합니다.
6. 임시 음성·Trainer 파일은 성공과 실패 모두 제거합니다.

```bash
scripts/run_whisper_lora_mps_once.sh \
  /private/venv/bin/python \
  /private/gwangju-lora-artifacts-v1 \
  /private/current-local-cost-quote.json \
  /private/training-run-UNIQUE
```

authorization은 정확한 speech commit과 1회 실행에 결합하고, 같은 ID의 원격 claim을 GCS에
원자적으로 처음 생성한 실행만 허용합니다. 내부 Python deadline은 42,600초, 외부 process
timeout은 42,900초입니다. `TERM`은 Python cleanup을 실행하고 runner도 정확한 output 이름의
private staging 경로만 정리합니다.

학습 성공 직후 상태도 `trained_unvalidated`, 사실 상태는 **부분 구현 또는 개발용 데모**입니다.
A/B/C 변환·잠금 dev·downstream 안전 Gate를 모두 통과하기 전에는 정확도 향상이나 채택을
주장하지 않습니다. 소유한 M4 실행의 증분 서버 비용은 0원이며, 기존 보수적 개발비 ceiling
50,000원은 그대로 기록합니다. 실행 직전 확인서는 별도로 생성·해시 고정합니다.

## 1차 FP16 MPS 실패 기록

- authorization: `lora-20260907-006`(소진, 재사용 금지)
- 관측 지점: 25/1136 step
- 관측값: loss 12157.4975, gradient norm NaN
- 조치: 30 step에서 수동 중단
- 산출물: adapter·training report 없음, private staging/work 정리 확인
- 추가 서버 비용: 0원
- 결정: **해당 FP16 실행 기각, 기준선 유지**

이 결과로 LoRA의 성능을 판단할 수는 없습니다. 다음 실험은 동일 데이터의 소규모 FP32 MPS
smoke에서 유한 loss·gradient, 실제 LoRA parameter update를 모두 확인하는 원인 분리
실험입니다. 이를 통과한 경우에만 새 commit·확인서·single-use authorization으로 전체
학습을 한 번 실행합니다.

## A/B/C 변환 Gate

유효 adapter의 training report가 모든 출력 artifact hash와 일치할 때만 다음을 생성합니다.

1. A: 외부에 고정한 현재 `Systran/faster-whisper-small` 기준선
2. B: 고정 OpenAI base를 C와 같은 CTranslate2 4.8.2·FP16 설정으로 변환한 control
3. C: adapter를 base에 `safe_merge`한 뒤 B와 같은 설정으로 변환한 candidate

변환 보고서는 학습 commit과 변환기 commit, base revision, converter version, B·C artifact
hash를 기록합니다. 변환 성공은 비교 가능성만 뜻하며 정확도·안전·채택 주장을 허용하지
않습니다.
