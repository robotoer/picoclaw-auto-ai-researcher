"""Tests for PDFExtractor."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from auto_researcher.ingestion.pdf_extractor import (
    ExtractedContent,
    PDFExtractor,
)
from auto_researcher.models import Paper, PaperMetadata


def make_paper(pdf_url: str | None = "http://example.com/test.pdf") -> Paper:
    return Paper(
        metadata=PaperMetadata(
            arxiv_id="2401.00001v1",
            title="Test Paper",
            authors=["Author A"],
            abstract="Test abstract",
            categories=["cs.AI"],
            published=datetime(2024, 1, 15),
            pdf_url=pdf_url,
        ),
    )


@pytest.fixture
def extractor():
    return PDFExtractor()


class TestExtractSections:
    def test_numbered_sections_extracted(self, extractor):
        text = (
            "1 Introduction\nThis is the intro text.\n\n"
            "2 Methods\nThis is the methods text.\n\n"
            "3 Results\nThese are results."
        )
        sections = extractor._extract_sections(text)
        assert "1 Introduction" in sections
        assert "2 Methods" in sections
        assert "3 Results" in sections
        assert "intro text" in sections["1 Introduction"]

    def test_subsections_extracted(self, extractor):
        text = (
            "1 Introduction\nIntro text.\n\n"
            "1.1 Background\nBackground text.\n\n"
            "2 Methods\nMethods text."
        )
        sections = extractor._extract_sections(text)
        assert "1.1 Background" in sections

    def test_no_sections_returns_full(self, extractor):
        text = "Just plain text without any sections."
        sections = extractor._extract_sections(text)
        assert "full" in sections


class TestExtractEquations:
    def test_dollar_dollar_equations(self, extractor):
        text = "Some text $$E = mc^2$$ more text"
        eqs = extractor._extract_equations(text)
        assert len(eqs) == 1
        assert "E = mc^2" in eqs[0]

    def test_bracket_equations(self, extractor):
        text = r"Text \[x + y = z\] more"
        eqs = extractor._extract_equations(text)
        assert len(eqs) == 1

    def test_equation_environment(self, extractor):
        text = r"\begin{equation}a^2 + b^2 = c^2\end{equation}"
        eqs = extractor._extract_equations(text)
        assert len(eqs) == 1

    def test_no_equations(self, extractor):
        text = "No equations here."
        eqs = extractor._extract_equations(text)
        assert eqs == []


class TestExtractCode:
    def test_backtick_code(self, extractor):
        text = "Text ```python\nprint('hello')\n``` more"
        snippets = extractor._extract_code(text)
        assert len(snippets) == 1

    def test_lstlisting(self, extractor):
        text = r"\begin{lstlisting}x = 1\end{lstlisting}"
        snippets = extractor._extract_code(text)
        assert len(snippets) == 1

    def test_verbatim(self, extractor):
        text = r"\begin{verbatim}raw text\end{verbatim}"
        snippets = extractor._extract_code(text)
        assert len(snippets) == 1


class TestExtractTables:
    def test_table_references(self, extractor):
        text = "Table 1: Comparison of methods.\n\nSome text.\n\nTable 2. Results overview.\n\n"
        tables = extractor._extract_tables(text)
        assert len(tables) == 2
        assert tables[0]["number"] == "1"
        assert "Comparison" in tables[0]["caption"]


class TestExtractFigures:
    def test_figure_references(self, extractor):
        text = "Figure 1: Architecture diagram.\n\nMore text."
        figures = extractor._extract_figures(text)
        assert len(figures) == 1
        assert figures[0]["number"] == "1"


class TestExtractReferences:
    def test_extracts_numbered_refs(self, extractor):
        text = (
            "Some paper text.\n\n"
            "References\n"
            "[1] Smith et al. Some paper. 2023.\n"
            "[2] Jones. Another paper. 2024.\n"
        )
        refs = extractor._extract_references(text)
        assert len(refs) == 2
        assert "Smith" in refs[0]

    def test_no_references_section(self, extractor):
        text = "Paper without references section."
        refs = extractor._extract_references(text)
        assert refs == []


class TestExtractFromPaper:
    @pytest.mark.asyncio
    async def test_no_pdf_url_returns_unchanged(self, extractor):
        paper = make_paper(pdf_url=None)
        result = await extractor.extract_from_paper(paper)
        assert result.full_text is None

    @pytest.mark.asyncio
    async def test_successful_extraction(self, extractor, monkeypatch):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "1 Introduction\nContent here."
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.close = MagicMock()

        async def mock_download(self_inner, url):
            return b"fake-pdf-bytes"

        def mock_extract(self_inner, pdf_bytes):
            return ExtractedContent(
                full_text="1 Introduction\nContent here.",
                sections={"1 Introduction": "Content here."},
            )

        monkeypatch.setattr(PDFExtractor, "_download_pdf", mock_download)
        monkeypatch.setattr(PDFExtractor, "_extract_from_bytes", mock_extract)

        paper = make_paper()
        result = await extractor.extract_from_paper(paper)
        assert result.full_text is not None
        assert "Content here" in result.full_text

    @pytest.mark.asyncio
    async def test_extraction_failure_returns_original(self, extractor, monkeypatch):
        async def mock_download(self_inner, url):
            raise Exception("Download failed")

        monkeypatch.setattr(PDFExtractor, "_download_pdf", mock_download)
        paper = make_paper()
        result = await extractor.extract_from_paper(paper)
        assert result.full_text is None


class TestProcessDocument:
    def test_multi_page_joining(self, extractor):
        mock_doc = MagicMock()
        page1 = MagicMock()
        page1.get_text.return_value = "Page 1 content."
        page2 = MagicMock()
        page2.get_text.return_value = "Page 2 content."
        mock_doc.__iter__ = lambda self: iter([page1, page2])

        result = extractor._process_document(mock_doc)
        assert "Page 1 content" in result.full_text
        assert "Page 2 content" in result.full_text
        assert "\n\n" in result.full_text
