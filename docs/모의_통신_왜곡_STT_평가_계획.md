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

## 사실 상태

| 항목 | 상태 |
|---|---|
| 전체 18조건·해시·24시간 상한 검사 | 구현 완료 |
| 조건별 STT·paired 변화 집계 fixture 테스트 | 구현 완료 |
| 서울·인천 승인 데이터 실행 | 설계 완료·구현 전 |
| STT→Parser→Resolver 안전 평가 | 설계 완료·구현 전 |
| 실제 현장 무전 성능 | 검증되지 않은 가설 |

## 주장 경계

결과는 특정 AIHub 신고 전화 표본과 사전 등록한 절차적 왜곡에만 적용됩니다. 실제 무전기
제조사·코덱·주파수 환경, 현장 소음, 화학시설 현장 정확도나 안전성을 증명하지 않습니다.
