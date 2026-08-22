from pathlib import Path

from haystack.components.converters import PyPDFToDocument


pdf_folder = Path("pdfs")

converter = PyPDFToDocument()

pdf_files = list(pdf_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF documents.")

for pdf_file in pdf_files:
    print(f"\nReading: {pdf_file.name}")

    result = converter.run(sources=[pdf_file])

    documents = result["documents"]

    print(f"Pages/documents extracted: {len(documents)}")

    if documents:
        print("Preview:")
        print(documents[0].content[:300])