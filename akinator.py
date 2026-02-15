"""Probabilistic Akinator-style engine.

This module implements a lightweight Bayesian question-answering model:
- hypotheses are candidate entities (e.g., animals, characters)
- each hypothesis has a probability of answering "yes" for each question
- posterior probabilities are updated after each user answer
- the next question is chosen by expected information gain
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Dict, Iterable, List, Mapping, Optional


Answer = str  # "yes" | "no" | "unknown"


@dataclass
class Guess:
    name: str
    probability: float


class ProbabilisticAkinator:
    """A Bayesian guessing engine inspired by Akinator.

    Parameters
    ----------
    likelihoods:
        Mapping from hypothesis name -> question -> P(answer=yes | hypothesis).
    priors:
        Optional prior probabilities over hypotheses. If omitted, uniform prior.
    """

    def __init__(
        self,
        likelihoods: Mapping[str, Mapping[str, float]],
        priors: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not likelihoods:
            raise ValueError("likelihoods must not be empty")

        self._questions = self._collect_questions(likelihoods)
        self._likelihoods: Dict[str, Dict[str, float]] = {
            h: {q: self._validate_prob(v) for q, v in qs.items()}
            for h, qs in likelihoods.items()
        }

        if priors is None:
            uniform = 1.0 / len(self._likelihoods)
            self._posterior = {h: uniform for h in self._likelihoods}
        else:
            self._posterior = self._normalize(dict(priors), required_keys=set(self._likelihoods))

        self._asked: List[str] = []

    @property
    def posterior(self) -> Dict[str, float]:
        return dict(self._posterior)

    @property
    def asked_questions(self) -> List[str]:
        return list(self._asked)

    @property
    def questions(self) -> List[str]:
        return list(self._questions)

    def ask_next_question(self) -> Optional[str]:
        """Return the best next question (max expected information gain)."""
        remaining = [q for q in self._questions if q not in self._asked]
        if not remaining:
            return None

        scores = {q: self._information_gain(q) for q in remaining}
        return max(scores, key=scores.get)

    def update(self, question: str, answer: Answer) -> None:
        """Update posterior from an answer to a question.

        answer can be:
        - "yes": uses P(yes|h)
        - "no": uses 1 - P(yes|h)
        - "unknown": no update
        """
        if question not in self._questions:
            raise KeyError(f"Unknown question: {question}")
        if answer not in {"yes", "no", "unknown"}:
            raise ValueError("answer must be one of: yes, no, unknown")

        if question not in self._asked:
            self._asked.append(question)

        if answer == "unknown":
            return

        new_posterior: Dict[str, float] = {}
        for h, prior in self._posterior.items():
            py = self._likelihoods[h].get(question, 0.5)
            evidence = py if answer == "yes" else (1.0 - py)
            # small floor avoids hard zeros and helps with noisy users
            new_posterior[h] = prior * max(evidence, 1e-9)

        self._posterior = self._normalize(new_posterior)

    def top_guesses(self, k: int = 3) -> List[Guess]:
        ranked = sorted(self._posterior.items(), key=lambda x: x[1], reverse=True)
        return [Guess(name=h, probability=p) for h, p in ranked[:k]]

    def is_confident(self, threshold: float = 0.85) -> bool:
        best = max(self._posterior.values())
        return best >= threshold

    def reset(self) -> None:
        if not self._posterior:
            return
        uniform = 1.0 / len(self._posterior)
        self._posterior = {h: uniform for h in self._posterior}
        self._asked.clear()

    def _information_gain(self, question: str) -> float:
        current_entropy = self._entropy(self._posterior.values())

        p_yes = sum(self._posterior[h] * self._likelihoods[h].get(question, 0.5) for h in self._posterior)
        p_no = 1.0 - p_yes

        if p_yes <= 0 or p_no <= 0:
            return 0.0

        posterior_yes = {}
        posterior_no = {}
        for h in self._posterior:
            py = self._likelihoods[h].get(question, 0.5)
            posterior_yes[h] = self._posterior[h] * py
            posterior_no[h] = self._posterior[h] * (1.0 - py)

        posterior_yes = self._normalize(posterior_yes)
        posterior_no = self._normalize(posterior_no)

        expected_entropy = (
            p_yes * self._entropy(posterior_yes.values())
            + p_no * self._entropy(posterior_no.values())
        )
        return current_entropy - expected_entropy

    @staticmethod
    def _entropy(values: Iterable[float]) -> float:
        total = 0.0
        for p in values:
            if p > 0:
                total -= p * log2(p)
        return total

    @staticmethod
    def _validate_prob(value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Probability out of range: {value}")
        return float(value)

    @staticmethod
    def _collect_questions(likelihoods: Mapping[str, Mapping[str, float]]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for qs in likelihoods.values():
            for q in qs:
                if q not in seen:
                    seen.add(q)
                    ordered.append(q)
        return ordered

    @staticmethod
    def _normalize(
        values: Dict[str, float],
        required_keys: Optional[set[str]] = None,
    ) -> Dict[str, float]:
        if required_keys is not None:
            missing = required_keys - values.keys()
            extra = values.keys() - required_keys
            if missing:
                raise ValueError(f"Missing prior(s): {sorted(missing)}")
            if extra:
                raise ValueError(f"Unknown prior(s): {sorted(extra)}")

        total = sum(values.values())
        if total <= 0:
            raise ValueError("Cannot normalize non-positive values")
        return {k: v / total for k, v in values.items()}
