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
- Parser·Resolver·Retriever·CAMEO 규칙은 `analysis-engine`의 책임입니다.
- 인증·사고 상태·감사 기록은 `back`의 책임입니다.
- 원본 음성과 모델 가중치는 Git에 저장하지 않습니다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 저장소·CI 골격 | 구현 완료 |
| faster-whisper 기준선 | 설계 전·구현 전 |
| AIHub 신고음성 전처리 | 설계 전·구현 전 |
| 화학용어 후처리 | 검증되지 않은 가설 |
| 현장 무전 성능 | 검증되지 않음 |

## 기본 검증

```bash
python -m compileall -q src tests
PYTHONPATH=src python -m unittest discover -s tests -v
```
