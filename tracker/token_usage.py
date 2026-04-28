from typing import Any

from yaml import Token

from tracker.schemas import TokenUsage

TokenLedger = list[TokenUsage | None]

_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    # model: (input_usd_per_1m, output_usd_per_1m)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def get_token_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=_estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def get_total_token_usage(responses: list[Any]) -> TokenUsage:
    usages = [get_token_usage(response) for response in responses]
    return sum_token_usages(TokenLedger(usages))


def sum_token_usages(usages: TokenLedger) -> TokenUsage:
    total = TokenUsage()
    has_usage = False

    for usage in usages:
        if usage is None:
            continue

        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens
        total.total_tokens += usage.total_tokens
        total.estimated_cost_usd += usage.estimated_cost_usd

    total.estimated_cost_usd = round(total.estimated_cost_usd, 6)

    return total


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    from tracker.config import settings

    pricing = _PRICING_PER_1M_TOKENS.get(settings.LLM_MODEL)

    if pricing is None:
        return 0.0

    input_price_per_1m, output_price_per_1m = pricing

    input_cost = input_tokens * input_price_per_1m / 1_000_000
    output_cost = output_tokens * output_price_per_1m / 1_000_000

    return round(input_cost + output_cost, 6)


def print_token_usage(response: Any) -> None:
    usage = get_token_usage(response)

    print("\n--- Token Usage ---")

    if usage is None:
        print("No token usage available.")
        return

    _print_usage(usage)


def print_total_token_usage(responses: list[Any]) -> None:
    usage = get_total_token_usage(responses)

    print("\n--- Total Token Usage ---")

    if usage is None:
        print("No token usage available.")
        return

    _print_usage(usage)


def _print_usage(usage: TokenUsage) -> None:
    print(f"Input tokens:  {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens:  {usage.total_tokens}")
    print(f"Estimated cost: ${usage.estimated_cost_usd:.6f}")
