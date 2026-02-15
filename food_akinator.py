"""Food Akinator: discover what the user *really* wants to eat.

This module builds an Akinator-like conversational recommender for food.
It asks high-information questions, maintains a posterior over foods,
and returns both:
1) the most likely food the user had in mind
2) a "discovery" recommendation (serendipity-aware) the user may not have considered
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Dict, Iterable, List, Mapping, Optional


Answer = str  # yes | no | unknown


@dataclass(frozen=True)
class Food:
    name: str
    traits: Mapping[str, float]  # trait intensity in [0, 1]
    popularity: float = 0.5  # 1.0 means common choice, 0.0 means niche


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    trait: str
    positive: bool = True


@dataclass(frozen=True)
class FoodScore:
    name: str
    score: float


class FoodAkinator:
    def __init__(self, foods: List[Food], questions: List[Question], priors: Optional[Mapping[str, float]] = None) -> None:
        if not foods:
            raise ValueError("foods must not be empty")
        if not questions:
            raise ValueError("questions must not be empty")

        self._foods = {f.name: f for f in foods}
        self._questions = {q.key: q for q in questions}
        self._asked: List[str] = []

        if priors is None:
            p = 1.0 / len(self._foods)
            self._posterior = {name: p for name in self._foods}
        else:
            self._posterior = self._normalize(dict(priors), required_keys=set(self._foods))

    @property
    def posterior(self) -> Dict[str, float]:
        return dict(self._posterior)

    @property
    def asked_questions(self) -> List[str]:
        return list(self._asked)

    def ask_next_question(self) -> Optional[Question]:
        remaining = [q for key, q in self._questions.items() if key not in self._asked]
        if not remaining:
            return None

        scored = {q.key: self._information_gain(q) for q in remaining}
        best_key = max(scored, key=scored.get)
        return self._questions[best_key]

    def update(self, question_key: str, answer: Answer) -> None:
        if answer not in {"yes", "no", "unknown"}:
            raise ValueError("answer must be one of: yes, no, unknown")
        if question_key not in self._questions:
            raise KeyError(f"Unknown question: {question_key}")

        if question_key not in self._asked:
            self._asked.append(question_key)

        if answer == "unknown":
            return

        q = self._questions[question_key]
        updated: Dict[str, float] = {}
        for food_name, prior in self._posterior.items():
            food = self._foods[food_name]
            trait_v = float(food.traits.get(q.trait, 0.5))
            yes_prob = trait_v if q.positive else (1.0 - trait_v)
            evidence = yes_prob if answer == "yes" else (1.0 - yes_prob)
            updated[food_name] = prior * max(evidence, 1e-9)

        self._posterior = self._normalize(updated)

    def likely_foods(self, k: int = 3) -> List[FoodScore]:
        ranked = sorted(self._posterior.items(), key=lambda x: x[1], reverse=True)
        return [FoodScore(name=name, score=score) for name, score in ranked[:k]]

    def inferred_taste_profile(self) -> Dict[str, float]:
        traits: Dict[str, float] = {}
        for food_name, prob in self._posterior.items():
            for t, val in self._foods[food_name].traits.items():
                traits[t] = traits.get(t, 0.0) + prob * float(val)
        return traits

    def recommend_discovery(self, k: int = 3, serendipity: float = 0.2) -> List[FoodScore]:
        """Recommend foods balancing inferred preference and novelty.

        score = preference_match + serendipity * (1 - popularity)
        """
        pref = self.inferred_taste_profile()

        scored: List[FoodScore] = []
        for name, food in self._foods.items():
            match = 0.0
            weight_sum = 0.0
            for trait, pref_val in pref.items():
                f_val = float(food.traits.get(trait, 0.5))
                match += pref_val * f_val
                weight_sum += pref_val
            preference_match = (match / weight_sum) if weight_sum > 0 else 0.0
            novelty = 1.0 - float(food.popularity)
            final_score = preference_match + serendipity * novelty
            scored.append(FoodScore(name=name, score=final_score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    def confident(self, threshold: float = 0.82) -> bool:
        return max(self._posterior.values()) >= threshold

    def _information_gain(self, q: Question) -> float:
        current = self._entropy(self._posterior.values())

        p_yes = 0.0
        yes_post: Dict[str, float] = {}
        no_post: Dict[str, float] = {}
        for name, p in self._posterior.items():
            trait_v = float(self._foods[name].traits.get(q.trait, 0.5))
            yes_prob = trait_v if q.positive else (1.0 - trait_v)
            p_yes += p * yes_prob
            yes_post[name] = p * yes_prob
            no_post[name] = p * (1.0 - yes_prob)

        p_no = 1.0 - p_yes
        if p_yes <= 0 or p_no <= 0:
            return 0.0

        yes_post = self._normalize(yes_post)
        no_post = self._normalize(no_post)
        expected = p_yes * self._entropy(yes_post.values()) + p_no * self._entropy(no_post.values())
        return current - expected

    @staticmethod
    def _entropy(values: Iterable[float]) -> float:
        return -sum(p * log2(p) for p in values if p > 0)

    @staticmethod
    def _normalize(values: Dict[str, float], required_keys: Optional[set[str]] = None) -> Dict[str, float]:
        if required_keys is not None:
            if set(values) != required_keys:
                raise ValueError("priors keys must match food names")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("Cannot normalize non-positive values")
        return {k: v / total for k, v in values.items()}


def default_foods() -> List[Food]:
    return [
        Food("김치찌개", {"spicy": 0.85, "warm": 0.95, "soupy": 0.9, "light": 0.25, "meaty": 0.55}, popularity=0.9),
        Food("비빔밥", {"spicy": 0.5, "warm": 0.7, "soupy": 0.1, "light": 0.6, "meaty": 0.3}, popularity=0.85),
        Food("마라탕", {"spicy": 0.95, "warm": 0.95, "soupy": 0.8, "light": 0.35, "meaty": 0.6}, popularity=0.8),
        Food("초밥", {"spicy": 0.1, "warm": 0.15, "soupy": 0.05, "light": 0.8, "meaty": 0.55}, popularity=0.88),
        Food("쌀국수", {"spicy": 0.35, "warm": 0.85, "soupy": 0.95, "light": 0.7, "meaty": 0.45}, popularity=0.72),
        Food("샐러드볼", {"spicy": 0.1, "warm": 0.1, "soupy": 0.05, "light": 0.95, "meaty": 0.2}, popularity=0.65),
        Food("라멘", {"spicy": 0.45, "warm": 0.95, "soupy": 0.95, "light": 0.35, "meaty": 0.65}, popularity=0.83),
        Food("포케", {"spicy": 0.2, "warm": 0.2, "soupy": 0.05, "light": 0.88, "meaty": 0.45}, popularity=0.6),
        Food("인도커리", {"spicy": 0.85, "warm": 0.9, "soupy": 0.4, "light": 0.25, "meaty": 0.55}, popularity=0.58),
        Food("메밀소바", {"spicy": 0.1, "warm": 0.25, "soupy": 0.55, "light": 0.82, "meaty": 0.2}, popularity=0.5),
    ]


def default_questions() -> List[Question]:
    return [
        Question("q_spicy", "매콤한 음식이 당기나요", "spicy", True),
        Question("q_warm", "따뜻한 음식이 좋나요", "warm", True),
        Question("q_soup", "국물 있는 음식이 좋나요", "soupy", True),
        Question("q_light", "가벼운 식사가 좋나요", "light", True),
        Question("q_meat", "고기/단백질 느낌이 중요하나요", "meaty", True),
        Question("q_not_spicy", "맵지 않은 게 더 좋나요", "spicy", False),
    ]
