"""Smart query construction for defect-focused literature retrieval."""

from __future__ import annotations

import re
from itertools import chain

from waam_rag.config import Settings
from waam_rag.domain import DEFECT_MECHANISM_TERMS, DEFECT_SYNONYMS, MATERIAL_PATTERNS, PROCESS_TYPES
from waam_rag.schemas import QueryBundle, QueryRequest
from waam_rag.utils.text import normalize_for_match


class QueryBuilder:
    """Combine defect names, process parameters, and questions into retrieval queries."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(self, request: QueryRequest) -> QueryBundle:
        defect_name = self._canonical_defect(request.defect_name)
        expansion_enabled = (
            request.query_expansion
            if request.query_expansion is not None
            else self.settings.query_expansion_enabled
        )
        expanded_terms = self._expanded_terms(defect_name, expansion_enabled)
        process_parameters = request.process_parameters
        parameter_categories = process_parameters.parameter_categories() if process_parameters else []
        parameter_values = process_parameters.normalized_values() if process_parameters else {}
        process_fragments = (
            [category.replace("_", " ") for category in parameter_categories]
            + (process_parameters.categorical_terms() if process_parameters else [])
        )
        question = (request.question or "").strip()
        explicit_materials = request.filters.materials if request.filters else []

        process_types = self._detect_matches(
            " ".join(filter(None, [question, defect_name or "", *process_fragments])),
            PROCESS_TYPES,
        )
        materials = list(
            dict.fromkeys(
                [
                    *self._detect_matches(" ".join(filter(None, [question, *process_fragments])), MATERIAL_PATTERNS),
                    *explicit_materials,
                ]
            )
        )

        subqueries = self._build_subqueries(
            defect_name=defect_name,
            expanded_terms=expanded_terms,
            parameter_categories=parameter_categories,
            process_types=process_types,
            materials=materials,
            question=question,
            categorical_terms=process_parameters.categorical_terms() if process_parameters else [],
        )

        query_parts = list(
            dict.fromkeys(
                part
                for part in chain(
                    subqueries.values(),
                    [question] if question else [],
                    self.settings.default_process_terms,
                )
                if part
            )
        )
        lexical_query = " ".join(query_parts)
        dense_query = lexical_query
        return QueryBundle(
            mode=self._infer_mode(request),
            dense_query=dense_query,
            lexical_query=lexical_query,
            query_summary=self._summarize(request, defect_name),
            defect_name=defect_name,
            expanded_terms=expanded_terms,
            process_fragments=process_fragments,
            parameter_categories=parameter_categories,
            parameter_values=parameter_values,
            materials=materials,
            process_types=process_types,
            subqueries=subqueries,
        )

    def _infer_mode(self, request: QueryRequest) -> str:
        if request.defect_name and request.question and request.process_parameters:
            return "combined"
        if request.defect_name and request.process_parameters:
            return "defect_with_process"
        if request.question and not request.defect_name:
            return "free_text"
        return "defect_only"

    def _canonical_defect(self, defect_name: str | None) -> str | None:
        if not defect_name:
            return None
        normalized = normalize_for_match(defect_name)
        for canonical, variants in DEFECT_SYNONYMS.items():
            normalized_variants = {normalize_for_match(term) for term in variants}
            if normalized == canonical or normalized in normalized_variants:
                return canonical
        return defect_name.strip().lower()

    def _expanded_terms(self, defect_name: str | None, enabled: bool) -> list[str]:
        if not defect_name:
            return []
        if not enabled:
            return [defect_name]
        return list(dict.fromkeys([defect_name, *DEFECT_SYNONYMS.get(defect_name, [])]))

    def _detect_matches(self, text: str, taxonomy: dict[str, list[str]]) -> list[str]:
        normalized = normalize_for_match(text)
        hits = []
        for canonical, patterns in taxonomy.items():
            if any(re.search(pattern, normalized) for pattern in patterns):
                hits.append(canonical)
        return hits

    def _summarize(self, request: QueryRequest, defect_name: str | None) -> str:
        if request.question and defect_name and request.process_parameters:
            return f"Combined retrieval for {defect_name} with process parameters and user question."
        if request.question and defect_name:
            return f"Defect-focused question answering for {defect_name}."
        if request.question:
            return "Free-text literature retrieval query."
        if request.process_parameters and defect_name:
            return f"Defect query for {defect_name} augmented with process parameters."
        return f"Defect-only retrieval for {defect_name or 'unspecified defect'}."

    def _build_subqueries(
        self,
        *,
        defect_name: str | None,
        expanded_terms: list[str],
        parameter_categories: list[str],
        process_types: list[str],
        materials: list[str],
        question: str,
        categorical_terms: list[str],
    ) -> dict[str, str]:
        if not defect_name and question:
            return {"free_text": question}

        mechanism_terms = list(DEFECT_MECHANISM_TERMS.get(defect_name or "", ()))
        process_terms = process_types or ["waam", "wire arc additive manufacturing"]
        material_terms = materials or []
        category_terms = [category.replace("_", " ") for category in parameter_categories]
        defect_terms = expanded_terms or ([defect_name] if defect_name else [])

        subqueries = {
            "defect_mitigation": " ".join(
                dict.fromkeys(
                    [
                        *defect_terms,
                        *process_terms,
                        *material_terms,
                        *categorical_terms,
                        "mitigation",
                        "reduce defect",
                        "recommendation",
                        "parameter tuning",
                    ]
                )
            ).strip(),
            "defect_parameter_relationship": " ".join(
                dict.fromkeys(
                    [
                        *defect_terms,
                        *process_terms,
                        *category_terms,
                        *categorical_terms,
                        "parameter relationship",
                        "process window",
                        "experimental results",
                    ]
                )
            ).strip(),
            "defect_mechanism": " ".join(
                dict.fromkeys(
                    [
                        *defect_terms,
                        *process_terms,
                        *material_terms,
                        *mechanism_terms,
                        "mechanism",
                        "cause",
                    ]
                )
            ).strip(),
            "defect_material_process": " ".join(
                dict.fromkeys(
                    [
                        *defect_terms,
                        *process_terms,
                        *material_terms,
                        *categorical_terms,
                        "welding additive manufacturing",
                    ]
                )
            ).strip(),
        }
        if question:
            subqueries["user_question"] = question
        return {key: value for key, value in subqueries.items() if value}
