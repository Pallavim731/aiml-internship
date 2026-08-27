import os
import unittest


class TestMultiDocumentRAG(unittest.TestCase):

    def test_data_directory_exists(self):
        """Check that the data directory exists."""
        self.assertTrue(os.path.exists("data"))

    def test_document_count(self):
        """Check that at least four text documents exist."""

        files = [
            file
            for file in os.listdir("data")
            if file.endswith(".txt")
        ]

        self.assertGreaterEqual(len(files), 4)

    def test_required_documents_exist(self):
        """Check that all required documents are available."""

        required_files = [
            "artificial_intelligence.txt",
            "machine_learning.txt",
            "rag.txt",
            "vector_database.txt",
        ]

        for filename in required_files:
            file_path = os.path.join("data", filename)
            self.assertTrue(
                os.path.exists(file_path),
                f"Missing file: {filename}"
            )


if __name__ == "__main__":
    unittest.main()