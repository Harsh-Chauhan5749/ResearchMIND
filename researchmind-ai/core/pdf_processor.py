"""
ResearchMind AI – PDF Processor
Handles text extraction, metadata detection, section parsing, and smart chunking.
Uses PyMuPDF (fast) + PDFPlumber (tables).
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import fitz          # PyMuPDF
import pdfplumber

from core.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class PageData:
    page_num: int
    text: str
    char_count: int


@dataclass
class TextChunk:
    text: str
    page_num: int
    chunk_index: int
    paper_id: int
    paper_filename: str
    section: str = ""


@dataclass
class PaperContent:
    metadata: Dict
    pages: List[PageData]
    full_text: str
    tables: List[Dict] = field(default_factory=list)


# ─── Section Patterns ─────────────────────────────────────────────────────────

SECTION_PATTERNS = [
    r"^(?:\d+[\.\d]*\s+)?abstract\b",
    r"^(?:\d+[\.\d]*\s+)?introduction\b",
    r"^(?:\d+[\.\d]*\s+)?related\s+work\b",
    r"^(?:\d+[\.\d]*\s+)?background\b",
    r"^(?:\d+[\.\d]*\s+)?methodology\b",
    r"^(?:\d+[\.\d]*\s+)?method\b",
    r"^(?:\d+[\.\d]*\s+)?approach\b",
    r"^(?:\d+[\.\d]*\s+)?experiment",
    r"^(?:\d+[\.\d]*\s+)?evaluation\b",
    r"^(?:\d+[\.\d]*\s+)?results?\b",
    r"^(?:\d+[\.\d]*\s+)?discussion\b",
    r"^(?:\d+[\.\d]*\s+)?conclusion",
    r"^(?:\d+[\.\d]*\s+)?future\s+work\b",
    r"^(?:\d+[\.\d]*\s+)?limitation",
    r"^(?:\d+[\.\d]*\s+)?references?\b",
]


class PDFProcessor:
    """Extracts and processes content from PDF research papers."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, pdf_path: str) -> PaperContent:
        """Full pipeline: extract → clean → detect metadata → extract tables."""
        doc = fitz.open(pdf_path)

        raw_metadata = doc.metadata or {}
        pages: List[PageData] = []
        full_text = ""

        for page_num in range(len(doc)):
            raw = doc[page_num].get_text("text")
            cleaned = self._clean_text(raw)
            pages.append(PageData(
                page_num=page_num + 1,
                text=cleaned,
                char_count=len(cleaned),
            ))
            full_text += f"\n\n[PAGE {page_num + 1}]\n{cleaned}"

        doc.close()

        metadata = self._extract_metadata(raw_metadata, full_text, pdf_path)
        tables   = self._extract_tables(pdf_path)

        return PaperContent(
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            tables=tables,
        )

    def create_chunks(
        self, pages: List[PageData], paper_id: int, filename: str
    ) -> List[TextChunk]:
        """Convert page list into overlapping semantic chunks with page tracking."""
        chunks: List[TextChunk] = []
        current_section = "Introduction"
        idx = 0

        for page in pages:
            if not page.text.strip():
                continue

            # Detect section headers per line
            lines = page.text.split("\n")
            buffer = ""

            for line in lines:
                section = self._detect_section(line)
                if section:
                    current_section = section

                buffer += line + "\n"

                if len(buffer) >= self.chunk_size:
                    split_chunks = self._split_text(buffer)
                    for c in split_chunks:
                        if len(c.strip()) >= 50:
                            chunks.append(TextChunk(
                                text=c.strip(),
                                page_num=page.page_num,
                                chunk_index=idx,
                                paper_id=paper_id,
                                paper_filename=filename,
                                section=current_section,
                            ))
                            idx += 1
                    # Keep last part as overlap seed
                    buffer = buffer[-self.chunk_overlap:] if len(buffer) > self.chunk_overlap else ""

            # Flush remaining buffer
            if buffer.strip() and len(buffer.strip()) >= 50:
                chunks.append(TextChunk(
                    text=buffer.strip(),
                    page_num=page.page_num,
                    chunk_index=idx,
                    paper_id=paper_id,
                    paper_filename=filename,
                    section=current_section,
                ))
                idx += 1

        return chunks

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove lone page numbers
        text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
        # Remove common header/footer noise
        text = re.sub(r"(?i)(arxiv|preprint|under review)[^\n]*\n", "", text)
        return text.strip()

    def _detect_section(self, line: str) -> Optional[str]:
        stripped = line.strip()
        if 3 < len(stripped) < 80:
            for pattern in SECTION_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    return stripped
        return None

    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks at sentence/paragraph boundaries."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to split at paragraph, then sentence, then word
            split = text.rfind("\n\n", start, end)
            if split == -1 or split <= start:
                split = text.rfind(". ", start, end)
            if split == -1 or split <= start:
                split = text.rfind(" ", start, end)
            if split == -1 or split <= start:
                split = end

            chunks.append(text[start : split + 1])
            start = max(split + 1 - self.chunk_overlap, start + 1)

        return chunks

    def _extract_metadata(self, raw_meta: Dict, full_text: str, pdf_path: str) -> Dict:
        title   = raw_meta.get("title", "").strip()
        authors = raw_meta.get("author", "").strip()

        # Heuristic title from first non-empty line if PDF metadata is blank
        if not title:
            for line in full_text.split("\n"):
                line = line.strip()
                if len(line) > 10 and len(line) < 200 and not line.startswith("[PAGE"):
                    title = line
                    break

        abstract = self._extract_abstract(full_text)
        year     = self._extract_year(full_text)
        sections = self._extract_section_names(full_text)
        citations= self._extract_citations(full_text)

        return {
            "title":       title or Path(pdf_path).stem,
            "authors":     authors,
            "abstract":    abstract,
            "year":        year,
            "sections":    sections,
            "citations":   citations,
            "total_pages": full_text.count("[PAGE "),
            "file_size":   os.path.getsize(pdf_path),
        }

    def _extract_abstract(self, text: str) -> str:
        pattern = (
            r"(?i)abstract[\s\n:]+(.+?)(?=\n\n(?:introduction|1[\.\s]|keywords|index terms))"
        )
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()[:2500]

        # Fallback: grab text between "abstract" keyword and first double newline
        lower = text.lower()
        idx = lower.find("abstract")
        if idx != -1:
            snippet = text[idx + 8 : idx + 2500]
            end = snippet.find("\n\n")
            if end != -1:
                return snippet[:end].strip()

        return ""

    def _extract_year(self, text: str) -> Optional[int]:
        matches = re.findall(r"\b(20[0-2]\d|19[89]\d)\b", text[:3000])
        if matches:
            return int(matches[0])
        return None

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from the References section."""
        lower = text.lower()
        idx = lower.rfind("\nreferences")
        if idx == -1:
            idx = lower.rfind("\nbibliography")
            
        if idx == -1:
            return []
            
        ref_text = text[idx:]
        refs = []
        for line in ref_text.split("\n"):
            line = line.strip()
            if re.match(r"^\[\d+\]", line) or re.match(r"^\d+\.", line):
                if len(line) > 10:
                    refs.append(line[:250])
        return refs

    def _extract_section_names(self, text: str) -> List[str]:
        sections = []
        for line in text.split("\n"):
            s = self._detect_section(line)
            if s and s not in sections:
                sections.append(s)
            if len(sections) >= 15:
                break
        return sections

    def _extract_tables(self, pdf_path: str) -> List[Dict]:
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    for tbl in (page.extract_tables() or []):
                        if tbl and len(tbl) > 1:
                            tables.append({"page_num": i + 1, "data": tbl})
        except Exception:
            pass
        return tables
