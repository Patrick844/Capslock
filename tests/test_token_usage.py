from types import SimpleNamespace

from tracker.schemas import TokenUsage
from tracker.token_usage import get_token_usage, sum_token_usages


def test_get_token_usage_returns_token_usage_model() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
    )

    usage = get_token_usage(response)

    assert usage is not None
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


def test_get_token_usage_returns_empty_usage_when_no_usage() -> None:
    response = SimpleNamespace(usage=None)

    usage = get_token_usage(response)

    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0
    assert usage.estimated_cost_usd == 0.0


def test_sum_token_usages_ignores_none_values() -> None:
    usage_1 = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.001,
    )

    usage_2 = TokenUsage(
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        estimated_cost_usd=0.002,
    )

    total = sum_token_usages([usage_1, None, usage_2])

    assert total is not None
    assert total.input_tokens == 300
    assert total.output_tokens == 150
    assert total.total_tokens == 450
    assert total.estimated_cost_usd == 0.003
