# Probabilistic Akinator Engine

아키네이터(Akinator) 스타일의 확률 모델(베이지안 업데이트 기반) 구현입니다.

## 핵심 아이디어
- 가설(hypothesis): 추측 대상 (예: 동물, 캐릭터)
- 질문(question): yes/no 형태 특징 질문
- 우도(likelihood): `P(yes | hypothesis)`
- 사후확률(posterior): 답변을 받을 때마다 베이즈 규칙으로 갱신
- 질문 선택: **기대 정보이득(Expected Information Gain)**이 최대인 질문 선택

## 파일
- `akinator.py`: 확률 모델 엔진
- `example_cli.py`: 간단한 인터랙티브 데모
- `tests/test_akinator.py`: 기본 동작 테스트

## 실행
```bash
python example_cli.py
```

## 테스트
```bash
python -m pytest -q
```
