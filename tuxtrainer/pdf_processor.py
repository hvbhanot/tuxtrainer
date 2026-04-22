"""
PDF processing module — extract text from PDFs and prepare training data.

Supports two data formats:
  * "instruction" — each chunk becomes a Q&A-style pair
  * "completion"  — each chunk is fed as a text completion target

The module is designed to be dataset-size aware so the master model can
use token counts and chunk statistics when choosing hyperparameters.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from datasets import Dataset
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk extracted from a PDF."""
    text: str
    source: str           # PDF filename
    page: int             # page number (0-indexed)
    chunk_index: int      # chunk index within that page / document
    char_count: int = 0
    token_estimate: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        # Rough token estimate: ~4 chars per token for English, ~2 for CJK
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', self.text))
        latin_chars = self.char_count - cjk_chars
        self.token_estimate = max(1, latin_chars // 4 + cjk_chars // 2)


@dataclass
class DatasetStats:
    """Summary statistics about the prepared dataset — used by the master model."""
    total_chunks: int = 0
    total_tokens_estimate: int = 0
    avg_chunk_tokens: float = 0.0
    min_chunk_tokens: int = 0
    max_chunk_tokens: int = 0
    num_sources: int = 0
    format: str = "instruction"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from every page of a PDF.

    Returns a list of (page_number, page_text) tuples.  Page numbers are
    0-indexed to match PyMuPDF convention.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[tuple[int, str]] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages.append((page_num, text))
    finally:
        doc.close()

    logger.info("Extracted %d non-empty pages from %s", len(pages), pdf_path.name)
    return pages


def extract_text_from_pdfs(pdf_paths: list[Path]) -> list[tuple[str, int, str]]:
    """Extract text from multiple PDFs.

    Returns a list of (source_name, page_number, page_text) tuples.
    """
    all_pages: list[tuple[str, int, str]] = []
    for pdf_path in pdf_paths:
        pages = extract_text_from_pdf(pdf_path)
        for page_num, text in pages:
            all_pages.append((pdf_path.name, page_num, text))
    return all_pages


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_size: int = 50,
) -> list[str]:
    """Split text into overlapping chunks by character count.

    The chunker tries to break on sentence boundaries when possible so
    chunks don't end mid-sentence.

    Args:
        text: The raw text to split.
        chunk_size: Target chunk size in characters.
        overlap: Number of overlapping characters between consecutive chunks.
        min_chunk_size: Discard chunks shorter than this.

    Returns:
        A list of text chunks.
    """
    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < min_chunk_size:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            break

        # Try to break at the last sentence boundary within the window
        search_region = text[start:end]
        # Look for sentence-ending punctuation
        sentence_end = max(
            search_region.rfind('.'),
            search_region.rfind('!'),
            search_region.rfind('?'),
            search_region.rfind('\n'),
        )

        if sentence_end > chunk_size // 2:
            # Break at the sentence boundary
            end = start + sentence_end + 1

        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Training data formatting
# ---------------------------------------------------------------------------

def _format_instruction(chunk: Chunk, source_context: str) -> dict:
    """Format a chunk as an instruction-following example (Alpaca-style).

    The instruction asks the model to answer questions about the source
    material, and the output contains the actual chunk text.
    """
    return {
        "instruction": (
            f"Based on the following document content, provide a detailed "
            f"and accurate explanation about the topics covered. "
            f"Source: {source_context}"
        ),
        "input": "",
        "output": chunk.text,
    }


def _format_completion(chunk: Chunk) -> dict:
    """Format a chunk as a text-completion example."""
    return {
        "text": chunk.text,
    }


# ---------------------------------------------------------------------------
# Main PDF processor
# ---------------------------------------------------------------------------

class PDFProcessor:
    """Turn a collection of PDFs into a HuggingFace Dataset ready for fine-tuning.

    Usage::

        processor = PDFProcessor(chunk_size=512, overlap=64)
        dataset, stats = processor.process(pdf_paths, data_format="instruction")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
        data_format: str = "instruction",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.data_format = data_format

    def process(
        self,
        pdf_paths: list[Path],
        data_format: Optional[str] = None,
    ) -> tuple[Dataset, DatasetStats]:
        """Extract, chunk, and format PDFs into a training dataset.

        Args:
            pdf_paths: List of PDF file paths.
            data_format: Override the instance-level format if provided.

        Returns:
            A tuple of (HuggingFace Dataset, DatasetStats).
        """
        fmt = data_format or self.data_format

        console.print(f"[cyan]Extracting text from {len(pdf_paths)} PDF(s)...[/cyan]")

        all_chunks: list[Chunk] = []
        for pdf_path in pdf_paths:
            logger.info("Extracting PDF: %s", pdf_path)
            pages = extract_text_from_pdf(pdf_path)
            for page_num, page_text in pages:
                text_chunks = chunk_text(
                    page_text,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                    min_chunk_size=self.min_chunk_size,
                )
                for idx, tc in enumerate(text_chunks):
                    all_chunks.append(Chunk(
                        text=tc,
                        source=pdf_path.name,
                        page=page_num,
                        chunk_index=idx,
                    ))

        if not all_chunks:
            raise ValueError("No usable text extracted from the provided PDFs.")

        # Build stats
        token_estimates = [c.token_estimate for c in all_chunks]
        stats = DatasetStats(
            total_chunks=len(all_chunks),
            total_tokens_estimate=sum(token_estimates),
            avg_chunk_tokens=sum(token_estimates) / len(token_estimates),
            min_chunk_tokens=min(token_estimates),
            max_chunk_tokens=max(token_estimates),
            num_sources=len({c.source for c in all_chunks}),
            format=fmt,
        )

        console.print(
            f"\n[green]Extracted {stats.total_chunks} chunks "
            f"(~{stats.total_tokens_estimate:,} tokens) "
            f"from {stats.num_sources} PDFs[/green]"
        )

        # Format into training examples
        examples: list[dict] = []
        for chunk in all_chunks:
            source_ctx = f"{chunk.source} (page {chunk.page + 1})"
            if fmt == "instruction":
                examples.append(_format_instruction(chunk, source_ctx))
            elif fmt == "completion":
                examples.append(_format_completion(chunk))
            else:
                raise ValueError(f"Unknown data format: {fmt!r}")

        dataset = Dataset.from_list(examples)
        return dataset, stats

    def process_to_jsonl(
        self,
        pdf_paths: list[Path],
        output_path: Path,
        data_format: Optional[str] = None,
    ) -> DatasetStats:
        """Process PDFs and save the result as a JSONL file.

        Useful when you want to inspect / curate the data before training.

        Args:
            pdf_paths: List of PDF file paths.
            output_path: Where to save the JSONL file.
            data_format: Override the instance-level format.

        Returns:
            DatasetStats for the extracted data.
        """
        dataset, stats = self.process(pdf_paths, data_format)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        console.print(f"[green]Saved {len(dataset)} examples to {output_path}[/green]")
        return stats
