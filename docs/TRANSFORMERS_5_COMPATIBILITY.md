# Transformers 5 LoRA 호환성 검증

## 결론

`transformers 4.57.6`의 보안 경고를 해소하기 위해 `5.10.1`을 검증했습니다. 단순 dependency
1줄 변경만으로는 기존 CTranslate2 변환이 실패했으며, `WhisperFeatureExtractor`를 별도로
저장하도록 보완한 뒤 실제 기존 adapter의 load·safe merge와 control/candidate 변환이 모두
통과했습니다.

판정은 **조건부 채택**입니다. LoRA 개발 환경의 dependency와 변환 호환성만 채택하며,
기존 `wind_snr0` LoRA 후보의 성능 기각 결정은 바꾸지 않습니다. 새 버전으로 전체 학습을
재실행하지 않았고, 신고·현장 무전 정확도 또는 실제 안전성을 검증하지 않았습니다.

현재 `run_whisper_lora_mps_once.sh`와 `run_whisper_lora_wind_dev_once.sh`, 재현 문서의 능동
실행 명령은 모두 v2 execution config를 사용합니다. v1 config는 기존 학습·평가 artifact의
역사적 검증용으로만 보존하며 현재 `.[lora]` 환경에서 새 실행을 시작하는 설정이 아닙니다.

## 목표와 채택 조건

- `transformers 5.10.1`, `peft 0.20.0`, `accelerate 1.14.0`에서 Whisper와 generic PEFT
  wrapper가 동작할 것
- 잠긴 `openai/whisper-small` tokenizer가 전체 train/dev label을 처리할 것
- 160-token 상한 초과가 새로 생기지 않을 것
- MPS 2-step 합성 smoke에서 loss·gradient·parameter가 유한하고 LoRA tensor가 갱신될 것
- 기존 실제 adapter가 load·safe merge되고 CTranslate2 4.8.2 control/candidate로 변환될 것
- 기존 v1 실행 config와 결과 hash를 덮어쓰지 않을 것

## 관측 결과

### Tokenizer preflight

| 항목 | 4.57.6 역사적 결과 | 5.10.1 호환성 결과 |
|---|---:|---:|
| train utterance | 18,171 | 18,171 |
| dev utterance | 4,750 | 4,750 |
| train 최대 token | 58 | 60 |
| dev 최대 token | 57 | 59 |
| 160-token 초과 | 0 | 0 |
| 자동 학습 허용 | false | false |

tokenization은 동일하지 않았고 최대 길이가 2 tokens 증가했습니다. 따라서 기존 4.57.6
결과에 새 버전 결과를 소급하지 않습니다. 다만 사전 등록된 160-token 제한을 넘은 label은
0건이므로 token-length readiness Gate는 통과했습니다.

- v2 execution config SHA-256:
  `53d9908df32d653828e2dd0cfe69107302b555a8801f095f5daaae8befdd2fe2`
- 5.10.1 tokenizer report SHA-256:
  `a3eb94ad3feea7547d1395d04469a158ede8980988a5e038bb19254839e9f137`
- 모델: `openai/whisper-small`
- revision: `973afd24965f72e36ca33b3055d56a652f456b4d`
- 사실 상태: **부분 구현 또는 개발용 데모**

### MPS 2-step dependency smoke

실제 학습 데이터 대신 작은 합성 Whisper config를 사용했습니다.

| 항목 | 관측값 |
|---|---:|
| device | `mps:0` |
| optimizer step | 2 |
| loss | 4.8147, 4.8182 |
| gradient 검사 | 24 |
| parameter 검사 | 24 |
| 변경된 LoRA tensor | 12/12 |

이는 package API와 MPS 수치 경로의 최소 호환성 검사입니다. 실제 Whisper-small 전체 학습
안정성이나 성능 개선 근거가 아닙니다.

### 기존 adapter·CTranslate2 변환

역사적 v1 학습 adapter를 새 runtime에서 읽어 `safe_merge=true`로 병합했습니다. 첫 변환은
`WhisperProcessor.save_pretrained()`가 Transformers 5에서 `processor_config.json`만 만들고,
CTranslate2 4.8.2가 요구하는 `preprocessor_config.json`을 만들지 않아 실패했습니다.

`processor.feature_extractor.save_pretrained()`를 명시적으로 호출하도록 수정한 뒤 다음을
관측했습니다.

- 기존 adapter load·safe merge 성공
- 병합 모델 parameter: 241,734,912
- control `model.bin`·`config.json` 생성 성공
- candidate `model.bin`·`config.json` 생성 성공
- 두 `model.bin` 크기: 각각 483,546,977 bytes
- 임시 변환 결과는 검사 직후 삭제

이는 변환 가능성만 뜻합니다. 기존 LoRA 후보의 CER·WER·RTF·우선용어 지표를 다시 측정하거나
성능 판정을 변경하지 않았습니다.

## 재현 범위

전체 label tokenizer preflight는 다음처럼 실행합니다. 원본 label archive와 결과 JSON은
Git이 아닌 비공개 경로에 둡니다.

```bash
python -m chemicheck119_speech.lora_tokenizer_preflight \
  --execution-config config/whisper_lora_execution_v2.json \
  --experiment-config config/whisper_lora_experiment_v1.json \
  --artifact-root <private-data>/experiments/speech/lora/gwangju-v1/data-artifacts \
  --output <private-data>/experiments/speech/lora/gwangju-v1/tokenizer-preflight-transformers-5.10.1.json
```

저장소 회귀 검사는 다음을 포함합니다.

```bash
python -m compileall -q src tests scripts
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/export_speech_openapi.py --check
```

## 사실 상태와 주장 경계

| 사실 상태 | 현재 근거 |
|---|---|
| 구현 완료 | v1·v2 config 분리, 능동 runner의 v2 고정, config 기반 tokenizer version 검증, Transformers 5 feature extractor 저장 호환 코드 |
| 부분 구현 또는 개발용 데모 | 전체 label tokenizer preflight, 합성 MPS 2-step, 기존 adapter load·merge·임시 CTranslate2 변환 |
| 설계 완료·구현 전 | 새 runtime으로 Whisper-small 전체 LoRA 재학습과 A/B/C 재평가 |
| 검증되지 않은 가설 | 새 버전이 LoRA 정확도·현장 강건성·안전성을 개선한다는 주장 |

현재 후보 모델은 계속 기각 상태이며 운영 faster-whisper small CPU int8 기준선에는 이 변경을
적용하지 않습니다.

## 남은 Accelerate 경고

2026-09-09 재확인 결과 `accelerate<=1.14.0`의 sharded checkpoint `weight_map` path traversal·
denial-of-service 경고(CVE-2026-69112)가 추가로 열려 있으며 공개된 수정 버전이 없습니다.
따라서 본 변경을 “보안 경고 0건”으로 표현하지 않습니다.

현재 코드에서 취약 함수 `load_checkpoint_in_model`과 `load_checkpoint_and_dispatch`를 직접
호출하지 않습니다. LoRA 경로도 `openai/whisper-small`과 40자리 고정 revision만 허용하고,
`trust_remote_code`를 사용하지 않으며, local adapter는 training report의 전체 artifact
SHA-256과 대조합니다. 이는 노출을 줄이는 임시 경계이지 취약점 자체의 수정은 아닙니다.
수정 버전이 공개되기 전에는 외부·사용자 제공 checkpoint와 index JSON을 이 개발 환경에서
로드하지 않습니다.
