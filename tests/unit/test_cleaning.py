from __future__ import annotations

from waam_rag.ingestion.cleaning import TextCleaner
from waam_rag.schemas import PageText


def test_cleaner_removes_repeated_headers_and_repairs_hyphenation() -> None:
    cleaner = TextCleaner()
    pages = [
        PageText(
            page_number=1,
            text=(
                "WAAM Journal\n"
                "1\n"
                "Porosity is fre-\n"
                "quently observed in aluminum builds.\n"
                "Stable shielding gas reduces defects.\n"
            ),
        ),
        PageText(
            page_number=2,
            text=(
                "WAAM Journal\n"
                "2\n"
                "Porosity also depends on heat input.\n"
                "Experimental results confirm this.\n"
            ),
        ),
    ]

    cleaned = cleaner.clean_pages(pages)

    assert "WAAM Journal" not in cleaned[0].text
    assert "frequently" in cleaned[0].text
    assert cleaned[0].text.startswith("Porosity")
