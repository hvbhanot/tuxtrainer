"""
Tests for the PDF processor module.
"""

import tempfile
from pathlib import Path

import fitz  # PyMuPDF

import tuxtrainer.pdf_processor as pdf_processor_module
from tuxtrainer.pdf_processor import PDFProcessor, chunk_text, Chunk, DatasetStats


def _create_test_pdf(text: str, path: Path) -> Path:
    """Create a minimal PDF with the given text content."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class TestChunking:
    """Tests for text chunking."""

    def test_short_text(self):
        chunks = chunk_text("Hello world", chunk_size=100, min_chunk_size=5)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100)
        assert len(chunks) == 0

    def test_too_short_text(self):
        chunks = chunk_text("Hi", chunk_size=100, min_chunk_size=50)
        assert len(chunks) == 0

    def test_long_text_chunking(self):
        text = " ".join(["This is sentence number " + str(i) + "." for i in range(100)])
        chunks = chunk_text(text, chunk_size=200, overlap=50, min_chunk_size=20)
        assert len(chunks) > 1
        # Each chunk should respect the min size
        for chunk in chunks:
            assert len(chunk) >= 20

    def test_sentence_boundary_breaking(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = chunk_text(text, chunk_size=50, overlap=10, min_chunk_size=10)
        assert len(chunks) >= 2


class TestPDFProcessor:
    """Tests for the PDFProcessor class."""

    def test_process_single_pdf(self, tmp_path):
        # Create a test PDF
        pdf_path = _create_test_pdf(
            "This is a test document about machine learning and artificial intelligence. "
            "It contains several paragraphs of text that should be extractable.",
            tmp_path / "test.pdf",
        )

        processor = PDFProcessor(chunk_size=256, data_format="instruction")
        dataset, stats = processor.process([pdf_path])

        assert stats.total_chunks > 0
        assert stats.total_tokens_estimate > 0
        assert stats.num_sources == 1
        assert len(dataset) > 0
        assert "instruction" in dataset.column_names
        assert "output" in dataset.column_names

    def test_process_completion_format(self, tmp_path):
        pdf_path = _create_test_pdf(
            "This is test content for completion format processing.",
            tmp_path / "test.pdf",
        )

        processor = PDFProcessor(chunk_size=256, data_format="completion")
        dataset, stats = processor.process([pdf_path])

        assert "text" in dataset.column_names
        assert stats.format == "completion"

    def test_process_with_minimal_console(self, tmp_path, monkeypatch):
        pdf_path = _create_test_pdf(
            "Console-independent PDF processing should still work.",
            tmp_path / "test.pdf",
        )

        class StubConsole:
            def print(self, *args, **kwargs):
                return None

        monkeypatch.setattr(pdf_processor_module, "console", StubConsole())

        processor = PDFProcessor(chunk_size=256, data_format="instruction")
        dataset, stats = processor.process([pdf_path])

        assert len(dataset) > 0
        assert stats.total_chunks > 0

    def test_process_to_jsonl(self, tmp_path):
        pdf_path = _create_test_pdf(
            "Test content for JSONL export. " * 10,
            tmp_path / "test.pdf",
        )
        output_path = tmp_path / "output.jsonl"

        processor = PDFProcessor(chunk_size=256)
        stats = processor.process_to_jsonl([pdf_path], output_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) > 0

    def test_no_pdfs_raises(self):
        processor = PDFProcessor()
        try:
            processor.process([Path("/nonexistent/file.pdf")])
            assert False, "Should have raised"
        except FileNotFoundError:
            pass
