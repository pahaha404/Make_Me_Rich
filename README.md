# Food Akinator (음식 아키네이터)

사용자에게 아키네이터처럼 질문을 던지면서, **지금 가장 먹고 싶은 음식**을 확률적으로 추정하는 알고리즘입니다.
단순히 "맞추기"에 그치지 않고, 답변에서 추론한 취향을 바탕으로 **사용자도 미처 몰랐던 추천(discovery)**까지 제공합니다.

## 알고리즘 요약
1. 음식 후보마다 취향 특성(trait) 점수를 둡니다. (예: spicy, warm, soupy, light)
2. 질문은 특정 trait와 연결됩니다. (예: "매콤한 음식이 당기나요?")
3. 사용자의 답변(yes/no/unknown)을 받을 때마다 베이즈 업데이트로 각 음식의 사후확률을 갱신합니다.
4. 다음 질문은 기대 정보이득(Expected Information Gain)이 가장 큰 질문을 선택합니다.
5. 최종 결과:
   - **Likely food**: 사용자가 지금 원할 가능성이 가장 높은 음식
   - **Discovery recommendations**: 추론된 취향 + 새로움(낮은 대중성) 보너스를 반영한 추천

## 핵심 수식(직관)
- 업데이트:
  `P(food | answer) ∝ P(answer | food) * P(food)`
- 질문 선택:
  `IG(question) = H(current) - E[H(after answer)]`
- 디스커버리 추천 점수:
  `score = preference_match + serendipity * (1 - popularity)`

## 파일
- `food_akinator.py`: 음식 아키네이터 엔진
- `food_cli.py`: CLI 데모
- `tests/test_food_akinator.py`: 음식 도메인 테스트

## 실행
```bash
python food_cli.py
```

## 테스트
```bash
python -m pytest -q
```
