from llmlingua import PromptCompressor
import tiktoken


encoder = tiktoken.encoding_for_model("gpt-4o")


context = """
Machine learning is a branch of artificial intelligence
that allows computers to learn patterns from data.

Supervised learning uses labelled training data.
Classification predicts categories.
Regression predicts numerical values.

Unsupervised learning works with data without predefined
labels. Clustering is a common unsupervised technique.

Model evaluation can use accuracy, precision, recall,
F1-score and AUC.

Overfitting occurs when a model learns the training data
too closely and performs poorly on unseen data.

Regularisation can help reduce overfitting.

Feature engineering transforms raw data into useful
features for machine learning models.

Cross-validation helps estimate how well a model will
generalise to unseen data.
"""


# Repeat the context to create a long RAG-style prompt.
long_context = context * 20

question = """
What are the main differences between supervised and
unsupervised machine learning, and how can overfitting
be reduced?
"""


prompt = f"""
You are an AI assistant answering questions using
retrieved documentation.

Context:
{long_context}

Question:
{question}
"""


original_tokens = len(
    encoder.encode(prompt)
)


print("=" * 70)
print("LLMLINGUA PROMPT COMPRESSION")
print("=" * 70)

print(
    f"Original tokens: {original_tokens}"
)


compressor = PromptCompressor()


result = compressor.compress_prompt(
    prompt,
    rate=0.4
)


compressed_prompt = result["compressed_prompt"]


compressed_tokens = len(
    encoder.encode(compressed_prompt)
)


reduction = (
    1 - compressed_tokens / original_tokens
) * 100


print(
    f"Compressed tokens: {compressed_tokens}"
)

print(
    f"Token reduction: {reduction:.2f}%"
)


print("\nCompressed prompt:")
print("-" * 70)
print(compressed_prompt)