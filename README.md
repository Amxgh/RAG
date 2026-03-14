# WAAM Research RAG

Production-style Retrieval-Augmented Generation backend for research papers on weld defects in Wire Arc Additive Manufacturing (WAAM), welding, and additive manufacturing process monitoring / quality control.

This project builds the RAG layer only:

- PDF ingestion
- text cleaning and scientific chunking
- metadata enrichment
- local indexing
- hybrid dense + sparse retrieval
- optional reranking
- citation-ready context packs
- FastAPI endpoints

It does not build the downstream mitigation-decision LLM or any UI.

## Why This Design

### Architecture

The pipeline is intentionally modular:

1. `ingestion.parser`
   Extracts text and front-matter metadata from PDFs with a primary backend and fallback parser.
2. `ingestion.cleaning`
   Removes repeated headers/footers, page numbers, and broken line-wrap artifacts.
3. `ingestion.structure`
   Detects section headings and tags paragraph blocks with section/subsection metadata.
4. `ingestion.chunking`
   Builds page-aware, section-aware chunks with overlap and stable chunk IDs.
5. `ingestion.enrichment`
   Adds defect, process, material, and rhetorical-role metadata using lightweight heuristics.
6. `indexing.repository`
   Persists documents, chunks, and dense vectors in a local SQLite-backed catalog.
7. `indexing.bm25` + `retrieval.service`
   Runs dense retrieval, BM25 retrieval, reciprocal-rank fusion, metadata boosts, and optional reranking.
8. `citations` + `services.rag_service`
   Formats citation-ready output and builds downstream LLM context packs.
9. `api.app`
   Exposes the RAG through FastAPI.

### Chunking Rationale

Scientific retrieval quality depends more on coherent reasoning units than on naive fixed windows. This implementation:

- cleans PDF noise before chunking
- detects section and subsection boundaries first
- chunks within sections rather than across them
- uses approximate token targets with overlap
- preserves `start_page`, `end_page`, `page_numbers`, `section`, and `subsection`
- keeps cause / mitigation / results passages together when possible

### Retrieval Rationale

Defect retrieval in WAAM papers benefits from hybrid search:

- dense retrieval helps with paraphrases and semantically similar mitigation concepts
- BM25 helps exact defect names, process terms, and alloy names
- reciprocal rank fusion combines both
- metadata-aware boosts improve ranking for direct defect matches, process matches, and stronger evidence sections
- reranking is optional and easy to disable

### Citation Strategy

Each returned result includes:

- `source_file`
- `title`
- `authors`
- `year`
- `pages`
- `section` / `subsection`
- `chunk_id`
- exact chunk text

Citation formatting is configurable. The default style is:

- `[Smith et al., 2022, pp. 4-5]`

## Vector Backend Choice

The shipped default is `sqlite_local`, a zero-server SQLite-backed vector catalog with persisted embeddings and metadata. This keeps the project runnable on a workstation or edge-adjacent machine without provisioning another service.

If you want a dedicated local vector database later, Qdrant local is the recommended next backend because it aligns well with metadata filtering and local deployment, and the current repository boundary makes that swap straightforward.

## Project Structure

```text
.
|-- config/
|   `-- settings.example.yaml
|-- examples/
|   |-- ingest_request.json
|   |-- query_request.json
|   |-- query_response.json
|   `-- retrieve_context_request.json
|-- src/
|   `-- waam_rag/
|       |-- api/
|       |   `-- app.py
|       |-- indexing/
|       |   |-- bm25.py
|       |   |-- embeddings.py
|       |   `-- repository.py
|       |-- ingestion/
|       |   |-- chunking.py
|       |   |-- cleaning.py
|       |   |-- enrichment.py
|       |   |-- parser.py
|       |   `-- structure.py
|       |-- retrieval/
|       |   |-- fusion.py
|       |   |-- query_builder.py
|       |   |-- reranker.py
|       |   `-- service.py
|       |-- services/
|       |   `-- rag_service.py
|       |-- citations.py
|       |-- config.py
|       |-- domain.py
|       |-- logging_utils.py
|       `-- schemas.py
|-- tests/
|   |-- integration/
|   `-- unit/
|-- .env.example
|-- main.py
|-- pyproject.toml
`-- README.md
```

## Features

- Accepts a folder of PDFs or uploaded PDF files
- Incremental ingestion using file checksums
- Metadata extraction for title, authors, year, venue, page numbers, sections
- Reference-section exclusion by configuration
- Defect / process / material metadata enrichment
- Dense retrieval with pluggable embeddings
- Sparse BM25 retrieval
- Hybrid fusion and score boosting
- Optional reranking
- Citation-ready JSON output
- Context-pack assembly for a downstream LLM
- Structured logging for ingestion and query stages

## Dependencies

Install the project in editable mode:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[ml]"
pip install -e ".[pdf-fallback]"
pip install -e ".[dev]"
```

Recommended optional ML stack:

- `sentence-transformers` for stronger dense embeddings and cross-encoder reranking

## Configuration

Configuration is available through YAML and environment variables.

Primary files:

- `config/settings.example.yaml`
- `.env.example`

Important settings:

- `vector_backend`
- `embedding_backend`
- `embedding_model`
- `chunk_size_tokens`
- `chunk_overlap_tokens`
- `top_k`
- `dense_top_k`
- `sparse_top_k`
- `reranker_enabled`
- `reranker_top_n`
- `bm25_enabled`
- `exclude_references`
- `enable_ocr_fallback`
- `citation_style`
- `metadata_enrichment_enabled`

## Running Locally

### 1. Configure

```bash
copy .env.example .env
```

### 2. Start the API

```bash
python main.py
```

Or:

```bash
uvicorn waam_rag.api.app:app --app-dir src --host 0.0.0.0 --port 8000
```

### 3. Ingest a PDF folder

```bash
curl -X POST http://localhost:8000/ingest ^
  -H "Content-Type: application/json" ^
  -d @examples/ingest_request.json
```

### 4. Query

```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d @examples/query_request.json
```

### 5. Build a context pack

```bash
curl -X POST http://localhost:8000/retrieve-context ^
  -H "Content-Type: application/json" ^
  -d @examples/retrieve_context_request.json
```

## FastAPI Endpoints

- `GET /health`
- `POST /ingest`
- `POST /reindex`
- `GET /documents`
- `GET /documents/{doc_id}`
- `POST /query`
- `POST /retrieve-context`

### `POST /ingest`

Supports either:

- JSON body with `folder_path`
- multipart upload with one or more files in `files`

### `POST /query`

Supports four query modes:

1. Defect-only
2. Defect + process parameters
3. Free-text question
4. Combined defect + parameters + question

## Example Query Payload

```json
{
  "defect_name": "porosity",
  "question": "What literature-backed mitigation strategies can reduce porosity in WAAM?",
  "process_parameters": {
    "current": 180,
    "voltage": 24,
    "travel_speed": 6.5,
    "wire_feed_speed": 7.0,
    "shielding_gas": "argon"
  },
  "filters": {
    "year_min": 2018
  },
  "top_k": 8
}
```

## Example Query Response Shape

```json
{
  "query_summary": "Combined retrieval for porosity with process parameters and user question.",
  "results": [
    {
      "rank": 1,
      "chunk_id": "44644a8700ed2dc6:0002",
      "score": 0.93,
      "source_file": "papers/waam_porosity_review.pdf",
      "title": "Gas Porosity Mitigation in WAAM Aluminum Builds",
      "authors": ["Alice Smith", "Brian Lee"],
      "year": 2022,
      "pages": [4, 5],
      "section": "Results and Discussion",
      "citation": "[Smith et al., 2022, pp. 4-5]",
      "text": "Stable argon shielding and reduced heat input lowered pore formation during WAAM deposition...",
      "metadata": {
        "defect_terms": ["porosity"],
        "process_types": ["waam"],
        "materials": ["aluminum"]
      }
    }
  ]
}
```

## API Notes

- The `/query` endpoint returns raw ranked evidence chunks.
- The `/retrieve-context` endpoint returns a downstream-LLM context pack with short citations and assembled context text.
- Citation format is configurable through `citation_style`.

## Logging

The service emits structured logs for:

- documents processed
- chunks created
- ingestion failures
- dense retrieval timing
- sparse retrieval timing
- fusion timing
- reranking timing

## Testing

Run the test suite:

```bash
pytest -q
```

Coverage includes:

- PDF ingestion behavior
- text cleaning
- chunking
- metadata enrichment
- retrieval ranking
- citation formatting
- ingestion + query integration
- API integration test when FastAPI is installed in the environment

## Current Limitations

- OCR fallback is only exposed as a config flag placeholder; no OCR engine is bundled yet.
- Parser heuristics are tuned for born-digital papers and can still miss unusual layouts.
- Metadata enrichment uses heuristic patterns, not a trained scientific IE model.
- The local vector backend is optimized for workstation-scale corpora, not very large multi-user deployments.
- No deduplicated bibliography graph or paper-to-paper citation network is built yet.

## Future Improvements

- Add a Qdrant-local backend for larger corpora and richer filtering
- Add OCR and table-aware extraction for scanned papers
- Add domain-adapted scientific sentence-transformer defaults
- Add query-time diversification across papers to reduce same-document redundancy
- Add richer rhetorical-role tagging for cause / mitigation / limitation evidence
- Add chunk summarization and chunk-level keyword caches as optional retrieval metadata
