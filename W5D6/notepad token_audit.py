import tiktoken


MODEL = "gpt-4o"

INPUT_PRICE_PER_MILLION = 5.00
OUTPUT_PRICE_PER_MILLION = 15.00


encoder = tiktoken.encoding_for_model(MODEL)


prompts = {
    "Small": (
        "Explain machine learning in simple terms."
    ),

    "Medium": (
        "Explain machine learning to a beginner. "
        "Include supervised learning, unsupervised learning, "
        "and one practical example."
    ),

    "Large": (
        "Explain machine learning to a beginner in detail. "
        "Discuss supervised learning, unsupervised learning, "
        "reinforcement learning, training data, features, "
        "labels, model evaluation, overfitting, underfitting, "
        "and practical applications."
    ),

    "Very Large": (
        "You are an AI tutor helping an engineering student. "
        "Explain machine learning comprehensively. "
        "Discuss supervised learning, unsupervised learning, "
        "reinforcement learning, classification, regression, "
        "clustering, feature engineering, model training, "
        "validation, testing, accuracy, precision, recall, "
        "F1-score, overfitting, underfitting, regularisation, "
        "and real-world applications."
    ),

    "RAG-style": (
        "You are answering a question using retrieved documents. "
        "Use the context below to answer accurately.\n\n"
        "Context:\n"
        + ("Machine learning is a field of AI. " * 250)
        + "\n\nQuestion: What is machine learning?"
    )
}


print("=" * 70)
print("W5D6 TOKEN ECONOMICS AUDIT")
print("=" * 70)


results = []


for name, prompt in prompts.items():

    tokens = encoder.encode(prompt)

    token_count = len(tokens)

    input_cost = (
        token_count / 1_000_000
    ) * INPUT_PRICE_PER_MILLION

    results.append(
        (
            name,
            token_count,
            input_cost
        )
    )

    print(f"\n{name}")
    print("-" * 40)
    print(f"Characters : {len(prompt)}")
    print(f"Tokens     : {token_count}")
    print(f"Input cost : ${input_cost:.8f}")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for name, tokens, cost in results:

    print(
        f"{name:15} "
        f"{tokens:6} tokens   "
        f"${cost:.8f}"
    )