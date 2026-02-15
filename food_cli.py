from food_akinator import FoodAkinator, default_foods, default_questions


def main() -> None:
    game = FoodAkinator(default_foods(), default_questions())
    print("음식 아키네이터 시작! yes/no/unknown 으로 답해주세요.\n")

    max_round = 6
    for _ in range(max_round):
        q = game.ask_next_question()
        if q is None:
            break
        answer = input(f"Q. {q.text}? ").strip().lower()
        if answer not in {"yes", "no", "unknown"}:
            print(" -> yes/no/unknown 중 하나를 입력해 주세요.\n")
            continue

        game.update(q.key, answer)

        top = game.likely_foods(3)
        print("[현재 추정]")
        for t in top:
            print(f" - {t.name}: {t.score:.3f}")
        print()

        if game.confident():
            break

    final_guess = game.likely_foods(1)[0]
    discovery = game.recommend_discovery(3, serendipity=0.25)

    print(f"\n당신이 현재 가장 원할 가능성이 높은 음식: {final_guess.name} ({final_guess.score:.3f})")
    print("당신도 몰랐을 수 있는 취향 기반 추천:")
    for d in discovery:
        print(f" - {d.name}: {d.score:.3f}")


if __name__ == "__main__":
    main()
