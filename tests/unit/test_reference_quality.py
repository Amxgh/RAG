from __future__ import annotations

from waam_rag.config import Settings
from waam_rag.ingestion.quality import ChunkQualityAnalyzer
from waam_rag.schemas import ChunkRecord


def _chunk(text: str, *, section: str = "Results and Discussion") -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_file="paper.pdf",
        title="Paper",
        authors=["Alice Smith"],
        year=2022,
        venue="WAAM Journal",
        start_page=4,
        end_page=4,
        page_numbers=[4],
        section=section,
        subsection=None,
        text=text,
        token_count=20,
    )


def test_reference_heavy_chunk_is_flagged() -> None:
    settings = Settings()
    analyzer = ChunkQualityAnalyzer(settings)
    chunk = _chunk(
        "Smith, A. et al. (2020); Lee, B. et al. (2021); doi 10.1000/xyz. "
        "Journal of Welding, vol. 12, no. 3, pp. 10-22; https://doi.org/10.1000/xyz"
    )

    processed = analyzer.process_chunk(chunk)

    assert processed.reference_contamination_score >= settings.reference_exclusion_threshold
    assert processed.is_reference_heavy is True


def test_normal_scientific_paragraph_is_not_falsely_flagged() -> None:
    settings = Settings()
    analyzer = ChunkQualityAnalyzer(settings)
    chunk = _chunk(
        "Reducing heat input and improving argon shielding reduced porosity in WAAM aluminum during the experiments."
    )

    processed = analyzer.process_chunk(chunk)

    assert processed.reference_contamination_score < 0.4
    assert processed.is_reference_heavy is False


def test_reference_tail_is_trimmed_from_useful_chunk() -> None:
    settings = Settings()
    analyzer = ChunkQualityAnalyzer(settings)
    chunk = _chunk(
        "Stable argon shielding reduced porosity in WAAM aluminum walls.\n\n"
        "References\nSmith, A. et al. (2020). doi 10.1000/xyz"
    )

    processed = analyzer.process_chunk(chunk)

    assert "References" not in processed.text
    assert "doi" not in processed.text.lower()
    assert processed.metadata["reference_tail_trimmed"] is True
