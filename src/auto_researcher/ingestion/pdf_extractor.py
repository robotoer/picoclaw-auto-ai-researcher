"""PDF content extraction using pymupdf."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from auto_researcher.models import Paper
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedContent:
    """Structured content extracted from a PDF."""

    full_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    tables: list[dict[str, str]] = field(default_factory=list)
    figures: list[dict[str, str]] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


# Regex patterns for structure extraction
_SECTION_PATTERN = re.compile(
    r"^(\d+\.?\d*\.?\d*)\s+([A-Z][^\n]{2,80})$", re.MULTILINE
)
_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\begin\{equation\}(.+?)\\end\{equation\}", re.DOTALL)
_CODE_PATTERN = re.compile(
    r"```(.+?)```|\\begin\{lstlisting\}(.+?)\\end\{lstlisting\}|\\begin\{verbatim\}(.+?)\\end\{verbatim\}",
    re.DOTALL,
)
_TABLE_REF_PATTERN = re.compile(r"Table\s+(\d+)[.:]\s*(.+?)(?:\n\n|\Z)", re.DOTALL)
_FIGURE_REF_PATTERN = re.compile(r"Figure\s+(\d+)[.:]\s*(.+?)(?:\n\n|\Z)", re.DOTALL)
_REFERENCE_PATTERN = re.compile(
    r"^\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", re.MULTILINE | re.DOTALL
)


class PDFExtractor:
    """Extracts structured content from research paper PDFs."""

    def __init__(self, download_dir: Path | None = None) -> None:
        self._download_dir = download_dir or Path("data/pdfs")
        self._client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)

    async def extract_from_paper(self, paper: Paper) -> Paper:
        """Download and extract content from a paper's PDF, returning updated Paper."""
        if not paper.metadata.pdf_url:
            logger.warning("no_pdf_url", arxiv_id=paper.arxiv_id)
            return paper

        try:
            pdf_bytes = await self._download_pdf(paper.metadata.pdf_url)
            content = self._extract_from_bytes(pdf_bytes)
            return paper.model_copy(
                update={
                    "full_text": content.full_text,
                    "sections": content.sections,
                    "tables": content.tables,
                    "figures": content.figures,
                    "equations": content.equations,
                    "code_snippets": content.code_snippets,
                    "references": content.references,
                }
            )
        except Exception:
            logger.exception("pdf_extraction_failed", arxiv_id=paper.arxiv_id)
            return paper

    async def _download_pdf(self, url: str) -> bytes:
        """Download a PDF from a URL."""
        resp = await self._client.get(url)
        resp.raise_for_status()
        return bytes(resp.content)

    def _extract_from_bytes(self, pdf_bytes: bytes) -> ExtractedContent:
        """Extract structured content from PDF bytes using pymupdf."""
        import pymupdf

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            return self._process_document(doc)
        finally:
            doc.close()

    def extract_from_file(self, path: Path) -> ExtractedContent:
        """Extract structured content from a local PDF file."""
        import pymupdf

        doc = pymupdf.open(str(path))
        try:
            return self._process_document(doc)
        finally:
            doc.close()

    def _process_document(self, doc: Any) -> ExtractedContent:
        """Process a pymupdf document into structured content."""
        pages_text: list[str] = []
        for page in doc:
            pages_text.append(page.get_text())

        full_text = "\n\n".join(pages_text)

        sections = self._extract_sections(full_text)
        tables = self._extract_tables(full_text)
        figures = self._extract_figures(full_text)
        equations = self._extract_equations(full_text)
        code_snippets = self._extract_code(full_text)
        references = self._extract_references(full_text)

        return ExtractedContent(
            full_text=full_text,
            sections=sections,
            tables=tables,
            figures=figures,
            equations=equations,
            code_snippets=code_snippets,
            references=references,
        )

    def _extract_sections(self, text: str) -> dict[str, str]:
        """Extract named sections from the document."""
        matches = list(_SECTION_PATTERN.finditer(text))
        if not matches:
            return {"full": text}

        sections: dict[str, str] = {}
        for i, match in enumerate(matches):
            section_num = match.group(1)
            section_name = match.group(2).strip()
            key = f"{section_num} {section_name}"
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[key] = text[start:end].strip()
        return sections

    def _extract_tables(self, text: str) -> list[dict[str, str]]:
        """Extract table references and captions."""
        return [
            {"number": m.group(1), "caption": m.group(2).strip()}
            for m in _TABLE_REF_PATTERN.finditer(text)
        ]

    def _extract_figures(self, text: str) -> list[dict[str, str]]:
        """Extract figure references and captions."""
        return [
            {"number": m.group(1), "caption": m.group(2).strip()}
            for m in _FIGURE_REF_PATTERN.finditer(text)
        ]

    def _extract_equations(self, text: str) -> list[str]:
        """Extract equations from the text."""
        equations: list[str] = []
        for m in _EQUATION_PATTERN.finditer(text):
            eq = m.group(1) or m.group(2) or m.group(3)
            if eq:
                equations.append(eq.strip())
        return equations

    def _extract_code(self, text: str) -> list[str]:
        """Extract code snippets from the text."""
        snippets: list[str] = []
        for m in _CODE_PATTERN.finditer(text):
            code = m.group(1) or m.group(2) or m.group(3)
            if code:
                snippets.append(code.strip())
        return snippets

    def _extract_references(self, text: str) -> list[str]:
        """Extract the reference list from the paper."""
        # Find the references section
        ref_section_idx = -1
        for marker in ["References\n", "REFERENCES\n", "Bibliography\n"]:
            idx = text.rfind(marker)
            if idx != -1:
                ref_section_idx = idx
                break

        if ref_section_idx == -1:
            return []

        ref_text = text[ref_section_idx:]
        refs: list[str] = []
        for m in _REFERENCE_PATTERN.finditer(ref_text):
            ref = m.group(2).strip()
            ref = " ".join(ref.split())
            refs.append(ref)
        return refs

    async def close(self) -> None:
        await self._client.aclose()
