from __future__ import annotations

from pathlib import Path

from waam_rag.ingestion.parser import ResearchPaperParser
from waam_rag.schemas import PageText


def test_parser_uses_fallback_and_infers_metadata(workspace_tmp_path: Path, monkeypatch) -> None:
    pdf_path = workspace_tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    parser = ResearchPaperParser()

    def fail_primary(_path: Path):
        raise RuntimeError("primary failed")

    def succeed_fallback(_path: Path):
        return (
            [
                PageText(
                    page_number=1,
                    text="A Great WAAM Study\nAlice Smith, Bob Lee\nAbstract\nPorosity mitigation results.",
                )
            ],
            {},
        )

    monkeypatch.setattr(parser, "_parse_with_pypdf", fail_primary)
    monkeypatch.setattr(parser, "_parse_with_pymupdf", succeed_fallback)

    parsed = parser.parse(pdf_path)

    assert parsed.title == "A Great WAAM Study"
    assert parsed.authors == ["Alice Smith", "Bob Lee"]
    assert parsed.file_name == "paper.pdf"
