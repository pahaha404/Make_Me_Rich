from akinator import ProbabilisticAkinator


def main() -> None:
    # Example domain: guessing an animal
    model = {
        "cat": {
            "does_it_bark": 0.05,
            "does_it_meow": 0.95,
            "can_it_fly": 0.01,
            "lives_in_water": 0.05,
        },
        "dog": {
            "does_it_bark": 0.95,
            "does_it_meow": 0.02,
            "can_it_fly": 0.01,
            "lives_in_water": 0.05,
        },
        "eagle": {
            "does_it_bark": 0.01,
            "does_it_meow": 0.01,
            "can_it_fly": 0.98,
            "lives_in_water": 0.02,
        },
        "dolphin": {
            "does_it_bark": 0.01,
            "does_it_meow": 0.01,
            "can_it_fly": 0.01,
            "lives_in_water": 0.95,
        },
    }

    game = ProbabilisticAkinator(model)

    print("Akinator-like demo 시작! 답변은 yes/no/unknown")
    while True:
        if game.is_confident(0.85) or len(game.asked_questions) >= len(game.questions):
            break

        q = game.ask_next_question()
        if q is None:
            break

        ans = input(f"Q: {q}? ").strip().lower()
        if ans not in {"yes", "no", "unknown"}:
            print("yes / no / unknown 중에서 입력해 주세요.")
            continue

        game.update(q, ans)
        print("Top guesses:")
        for g in game.top_guesses(3):
            print(f" - {g.name}: {g.probability:.3f}")

    best = game.top_guesses(1)[0]
    print(f"\n제 추측은: {best.name} (p={best.probability:.3f})")


if __name__ == "__main__":
    main()
