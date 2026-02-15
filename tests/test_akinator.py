from akinator import ProbabilisticAkinator


LIKELIHOODS = {
    "cat": {"bark": 0.05, "meow": 0.95, "fly": 0.01},
    "dog": {"bark": 0.95, "meow": 0.02, "fly": 0.01},
    "eagle": {"bark": 0.01, "meow": 0.01, "fly": 0.98},
}


def test_update_pushes_probability_to_matching_hypothesis():
    game = ProbabilisticAkinator(LIKELIHOODS)
    game.update("meow", "yes")
    top = game.top_guesses(1)[0]

    assert top.name == "cat"
    assert top.probability > 0.8


def test_question_selection_returns_remaining_question():
    game = ProbabilisticAkinator(LIKELIHOODS)
    q1 = game.ask_next_question()
    assert q1 in {"bark", "meow", "fly"}

    game.update(q1, "unknown")
    q2 = game.ask_next_question()
    assert q2 is not None
    assert q2 != q1


def test_confidence_becomes_true_after_strong_evidence():
    game = ProbabilisticAkinator(LIKELIHOODS)
    game.update("fly", "yes")
    assert game.is_confident(0.8)
