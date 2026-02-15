from food_akinator import FoodAkinator, default_foods, default_questions


def test_spicy_yes_pushes_spicy_foods_up():
    g = FoodAkinator(default_foods(), default_questions())
    g.update("q_spicy", "yes")
    names = [x.name for x in g.likely_foods(3)]
    assert "마라탕" in names or "인도커리" in names


def test_not_spicy_yes_penalizes_spicy_foods():
    g = FoodAkinator(default_foods(), default_questions())
    g.update("q_not_spicy", "yes")
    top = g.likely_foods(3)
    top_names = {x.name for x in top}
    assert "마라탕" not in top_names


def test_discovery_recommendations_return_k_items():
    g = FoodAkinator(default_foods(), default_questions())
    g.update("q_light", "yes")
    recs = g.recommend_discovery(3, serendipity=0.3)
    assert len(recs) == 3
    assert recs[0].score >= recs[-1].score
