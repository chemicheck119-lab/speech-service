# 모의 통신 왜곡 STT 평가 계획

기준일: 2026-09-05

## 목표

광주에서 고정한 `faster-whisper small·CPU int8·hotword 미사용` 설정이 AIHub 서울·인천
화재 신고 전화의 절차적 통신 왜곡에서도 물질 관련 정보를 얼마나 보존하는지, clean
대조군과 동일 레코드 단위로 비교합니다.

## 실패 사례와 가설

- 실패 사례: clean에서 인식한 우선용어가 왜곡에서 누락되거나, 원문에 없는 우선용어가
  삽입되어 downstream Resolver 후보를 오염시킴
- 가설: SNR 0dB와 복합 스트레스에서 CER·WER가 가장 크게 증가하고 우선용어 F1이
  감소함
- 안전 가설: 낮은 품질에서 CAS 자동확정 대신 후보·불확실성·기권을 유지하면 잘못된
  단일 CAS 승격을 0건으로 유지할 수 있음

마지막 안전 가설은 STT 지표만으로 검증할 수 없으며 Parser·Resolver 연동 평가가
필요합니다.

## 입력과 실행 전 게이트

1. `radio-sim-v1`의 clean+17개 왜곡 조건이 정확히 모두 있어야 합니다.
2. 실행 summary, 원본 manifest, 우선용어 파일, 파생 manifest, audio/label archive의
   SHA-256이 모두 일치해야 합니다.
3. 모든 조건의 레코드 수가 같고 200건 이하이어야 합니다.
4. 전체 입력 음성이 24시간 이하이어야 합니다.
5. audio archive 하나는 512MiB, 전체 archive는 4GiB 이하여야 합니다.
6. 모델은 위 검사가 끝난 다음 한 번만 초기화합니다.

현재 GCP의 기존 기준선 Job timeout은 2시간입니다. 24시간 입력 상한은 광주 기준 RTF
0.214에서 약 5.14시간의 추론시간에 해당하므로, 실제 입력시간을 확인한 뒤 재시도 0인
별도 최대 6시간 Job을 만들거나 층별 표본 수를 낮춥니다. timeout 변경 전에는 실행하지
않습니다.

## 지표

- 조건별 CER·WER·RTF·실패 레코드 수
- 우선용어 presence Precision·Recall·F1·false insertion
- clean 대비 CER paired bootstrap 95% 구간(seed 119)
- clean 대비 WER·Recall·Precision·F1·false insertion 변화
- 후속 연동: STT→Parser 물질명 Recall, STT→Resolver Top-1·Top-3·coverage·기권율
- 안전 불변식: 잘못된 단일 CAS 승격 0건

## 판정

- 실행기 채택: fixture 및 실제 archive에서 해시·전체 조건·paired record 불변식과 비용
  상한 검사가 모두 통과할 때
- 기준선 조건부 채택: 왜곡별 실패 범위를 명시하고 downstream 안전 불변식이 유지될 때
- LoRA 진행: 서울·인천 두 지역 이상에서 동일한 음향 오류가 반복되고, 광주 Training만으로
  검증 가능한 학습 가설이 생길 때
- LoRA 기각: 우선용어 Recall만 오르고 Precision·F1이 하락하거나 false insertion·잘못된
  CAS 승격이 증가할 때

## 실제 실행 결과

사실 상태는 **부분 구현 또는 개발용 데모**입니다. AIHub 신고전화 각 40건에 절차적으로
만든 clean 포함 18조건을 적용했으며, 실제 무전 녹음이 아닙니다.

| 지역·조건 | CER | WER | 우선용어 Recall | Precision | false insertion |
|---|---:|---:|---:|---:|---:|
| 인천 clean | 35.91% | 53.64% | 96.30% | 100.00% | 0 |
| 인천 `wind_snr0` | 58.73% | 72.14% | 66.67% | 100.00% | 0 |
| 서울 clean | 39.40% | 59.59% | 93.33% | 100.00% | 0 |
| 서울 `wind_snr0` | 53.18% | 69.80% | 76.67% | 95.83% | 1 |

표의 Recall은 11개 우선용어 전체 presence 집계입니다. LoRA 진입 Gate에서 사용한 개별
공개 용어 `연기` Recall은 인천 66.67%, 서울 72.22%로 서로 다른 지표이므로 섞지 않습니다.
두 지역 모두 같은 `wind_snr0`에서 반복 누락이 확인돼 판정은
`ELIGIBLE_FOR_BOUNDED_LORA_EXPERIMENT_DESIGN`이었습니다. 이는 제한 실험을 설계할 근거일
뿐 LoRA의 개선이나 자동 학습·운영 채택을 뜻하지 않습니다.

두 지역 전체 downstream 실행에서 API 오류·분석 누락·후보 자동 승격·2-CAS Gate 우회·
확인 전 위험 출력·Rule 조기 실행은 모두 0건이었습니다. CAS 사람 정답이 없으므로 Resolver
정확도 또는 실제 안전성 지표로 표현하지 않습니다.

## 사실 상태

| 항목 | 상태 |
|---|---|
| 전체 18조건·해시·24시간 상한 검사 | 구현 완료 |
| 조건별 STT·paired 변화 집계 fixture 테스트 | 구현 완료 |
| 서울·인천 execution·summary runtime provenance 결합 | 구현 완료 |
| 서울·인천 승인 데이터 실행 | 부분 구현 또는 개발용 데모 — 각 40건×18조건 완료 |
| STT→Parser→Resolver 안전 평가 | 부분 구현 또는 개발용 데모 — 안전 위반 0, CAS 정답 없음 |
| 제한 Whisper LoRA | 부분 구현 또는 개발용 데모 — #18에서 진행 중 |
| 실제 현장 무전 성능 | 검증되지 않은 가설 |

주요 artifact SHA-256은 인천 summary
`6f85ffcd0b0f65c28721cff74351d1bdedffa037ce2ccdd26dc850c0befc57a6`, 서울 summary
`9144c4adf028aabd13257374689f25296ab45b8cdce6ccbf9a957919c15ab2fd`, execution provenance
`0a88ebda81062ca2010a85b166b9f5862b9479d4062d0e2be3673d2caeb764ce`, 최종 LoRA Gate
`2e681817780ecb1385271cff41adedcad8245ac25d985d64ff54dfe2569e13e9`입니다.

## 주장 경계

결과는 특정 AIHub 신고 전화 표본과 사전 등록한 절차적 왜곡에만 적용됩니다. 실제 무전기
제조사·코덱·주파수 환경, 현장 소음, 화학시설 현장 정확도나 안전성을 증명하지 않습니다.
