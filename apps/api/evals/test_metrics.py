from evals.metrics import mean_recall_at_k, recall_at_k, reciprocal_rank_fusion


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b"], ["a", "b", "c"], 2) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b"], ["a", "c"], 2) == 0.5


def test_recall_empty_relevant():
    assert recall_at_k([], ["a"], 5) == 1.0


def test_mean_recall():
    cases = [(["a"], ["a", "b"]), (["x"], ["y", "x"])]
    assert mean_recall_at_k(cases, 2) == 1.0


def test_rrf_prefers_consensus():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused[0][0] in {"a", "b"}
