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
| faster-whisper 1.2.1 평가 하네스 | 구현 완료·실데이터 실행 전 |
| AIHub 신고음성 ZIP 로딩·평가 | 구현 완료·77건 실행 전 |
| 기본 전사 vs hotword 힌트 A/B | 구현 완료·효과 미측정 |
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

## 기본 검증

```bash
python -m compileall -q src tests
PYTHONPATH=src python -m unittest discover -s tests -v
```
