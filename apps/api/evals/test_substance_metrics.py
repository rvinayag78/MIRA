from evals.metrics import (
    hit_rate_at_k,
    unsupported_claim_rate,
    uncertainty_appropriate,
)


def test_hit_rate_at_k():
    assert hit_rate_at_k(["a", "b"], ["x", "a", "y"], k=3) == 1.0
    assert hit_rate_at_k(["a"], ["x", "y"], k=2) == 0.0
    assert hit_rate_at_k([], ["x"], k=1) == 1.0


def test_unsupported_claim_rate_adversarial():
    rate = unsupported_claim_rate(
        "Yes, we went to Yosemite with Sarah when she was eight.",
        must_not_claim=["Yosemite", "Sarah"],
        expect_uncertainty=True,
        looks_uncertain=False,
    )
    assert rate == 1.0


def test_unsupported_claim_rate_uncertain_ok():
    rate = unsupported_claim_rate(
        "I don't think I ever recorded much about that.",
        must_not_claim=["Yosemite", "Sarah"],
        expect_uncertainty=True,
        looks_uncertain=True,
    )
    assert rate == 0.0


def test_uncertainty_appropriate():
    assert (
        uncertainty_appropriate(
            expect_uncertainty=True,
            must_refuse=True,
            refused=True,
            looks_uncertain=False,
        )
        == 1.0
    )
    assert (
        uncertainty_appropriate(
            expect_uncertainty=True,
            must_refuse=False,
            refused=False,
            looks_uncertain=False,
        )
        == 0.0
    )
