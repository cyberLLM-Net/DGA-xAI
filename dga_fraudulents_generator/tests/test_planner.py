from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.planner import build_generation_plan


def test_planner_with_categories():
    ins = [
        AlgorithmInspection(algorithm_code="a", path="/a", category="c1"),
        AlgorithmInspection(algorithm_code="b", path="/b", category="c1"),
        AlgorithmInspection(algorithm_code="c", path="/c", category="c2"),
    ]
    plans = build_generation_plan(ins, target_count=10)
    assert sum(p.target_count for p in plans.values()) == 10
    assert plans["a"].target_count + plans["b"].target_count in {5, 6}
    assert plans["c"].target_count in {4, 5}
