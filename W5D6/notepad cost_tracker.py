import csv
from datetime import datetime


USD_TO_INR = 85.0


MODELS = {
    "GPT-4o": {
        "input": 5.00,
        "output": 15.00
    },

    "Claude Sonnet 4": {
        "input": 3.00,
        "output": 15.00
    },

    "Gemini 2.0 Flash": {
        "input": 0.10,
        "output": 0.40
    },

    "GPT-4o mini": {
        "input": 0.15,
        "output": 0.60
    }
}


def calculate_cost(
    model,
    input_tokens,
    output_tokens
):

    input_price = MODELS[model]["input"]

    output_price = MODELS[model]["output"]

    input_cost = (
        input_tokens / 1_000_000
    ) * input_price

    output_cost = (
        output_tokens / 1_000_000
    ) * output_price

    total_usd = (
        input_cost +
        output_cost
    )

    total_inr = (
        total_usd *
        USD_TO_INR
    )

    return total_usd, total_inr


requests = [
    ("GPT-4o", 2000, 300),
    ("GPT-4o", 1000, 200),
    ("GPT-4o mini", 2000, 300),
    ("Gemini 2.0 Flash", 2000, 300)
]


with open(
    "cost_log.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Timestamp",
        "Model",
        "Input Tokens",
        "Output Tokens",
        "USD Cost",
        "INR Cost"
    ])

    daily_total = 0

    for model, input_tokens, output_tokens in requests:

        usd, inr = calculate_cost(
            model,
            input_tokens,
            output_tokens
        )

        daily_total += inr

        writer.writerow([
            datetime.now().isoformat(),
            model,
            input_tokens,
            output_tokens,
            round(usd, 8),
            round(inr, 4)
        ])

        print(
            f"{model}: ₹{inr:.4f}"
        )


print(
    f"\nDaily total: ₹{daily_total:.4f}"
)cost_log.csv