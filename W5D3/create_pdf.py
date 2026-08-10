from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

pdf_path = "sample.pdf"

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=A4
)

styles = getSampleStyleSheet()

content = [
    ("Artificial Intelligence and Machine Learning", "Title"),

    ("Machine learning is a branch of artificial intelligence "
     "that enables computers to learn patterns from data.", "BodyText"),

    ("Supervised learning uses labeled data to train machine "
     "learning models.", "BodyText"),

    ("Unsupervised learning works with unlabeled data and "
     "discovers hidden patterns in datasets.", "BodyText"),

    ("Deep learning uses neural networks with multiple layers.",
     "BodyText"),

    ("Natural language processing allows computers to process "
     "and understand human language.", "BodyText"),

    ("Computer vision enables computers to analyze images "
     "and videos.", "BodyText"),

    ("Vector databases store numerical representations called "
     "embeddings. These embeddings allow semantic search over "
     "documents.", "BodyText"),

    ("Retrieval augmented generation retrieves relevant "
     "information from documents and provides the information "
     "to a language model.", "BodyText"),

    ("Ollama allows large language models to run locally on "
     "a computer.", "BodyText"),
]

story = []

for text, style in content:
    story.append(Paragraph(text, styles[style]))
    story.append(Spacer(1, 12))

doc.build(story)

print(f"Created {pdf_path}")