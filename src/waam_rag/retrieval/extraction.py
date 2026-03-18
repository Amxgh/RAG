"""Structured evidence extraction for downstream reasoning."""

from __future__ import annotations

from waam_rag.domain import PARAMETER_ALIASES
from waam_rag.schemas import ChunkRecord, EvidenceTheme, ExtractedEvidence, ParameterEffect, QueryBundle
from waam_rag.utils.text import normalize_for_match, sentence_split


class EvidenceExtractor:
    """Convert ranked chunks into structured evidence objects."""

    def extract_many(
        self,
        chunks_with_citations: list[tuple[ChunkRecord, str]],
        *,
        query_bundle: QueryBundle | None = None,
    ) -> list[ExtractedEvidence]:
        return [
            self.extract(chunk, citation, query_bundle=query_bundle)
            for chunk, citation in chunks_with_citations
        ]

    def extract(
        self,
        chunk: ChunkRecord,
        citation: str,
        *,
        query_bundle: QueryBundle | None = None,
    ) -> ExtractedEvidence:
        sentences = sentence_split(chunk.text)
        strategy = self._select_sentence(sentences, ("mitigat", "reduce", "optimiz", "minimiz", "avoid"))
        mechanism = self._select_sentence(sentences, ("mechanism", "cause", "due to", "because", "promotes", "leads to"))
        recommendation = self._select_sentence(sentences, ("recommend", "should", "it is advised", "suggest"))
        outcome = self._select_sentence(sentences, ("results show", "observed", "reduced", "improved", "decreased"))
        parameter_effects = self._extract_parameter_effects(chunk, query_bundle)
        evidence_type = self._primary_evidence_type(chunk.evidence_types)
        directness = self._directness_label(chunk.evidence_directness_score)
        confidence = self._confidence(
            chunk=chunk,
            strategy=strategy,
            mechanism=mechanism,
            parameter_effects=parameter_effects,
            outcome=outcome,
            directness=directness,
        )
        return ExtractedEvidence(
            chunk_id=chunk.chunk_id,
            paper=chunk.title,
            citation=citation,
            defects=chunk.defect_terms,
            strategy=strategy,
            mechanism=mechanism,
            parameters=chunk.process_parameters,
            parameter_effects=parameter_effects,
            evidence_type=evidence_type,
            material=", ".join(chunk.materials) if chunk.materials else None,
            process=", ".join(chunk.process_types) if chunk.process_types else None,
            experimental_outcome=outcome,
            recommendation=recommendation,
            directness=directness,
            confidence=confidence,
            review_or_experimental=(chunk.review_or_experimental or "unclear"),
            metadata={
                "reference_contamination_score": chunk.reference_contamination_score,
                "feature_directness_score": chunk.evidence_directness_score,
            },
        )

    def summarize(
        self,
        evidence_entries: list[ExtractedEvidence],
        *,
        query_bundle: QueryBundle | None = None,
    ) -> tuple[list[EvidenceTheme], list[str]]:
        theme_map: dict[str, EvidenceTheme] = {}
        reasoning_hints: list[str] = []

        for entry in evidence_entries:
            for theme in self._themes_for_entry(entry):
                current = theme_map.setdefault(
                    theme,
                    EvidenceTheme(theme=theme),
                )
                current.supporting_chunk_ids.append(entry.chunk_id)
                current.supporting_citations.append(entry.citation)
            for effect in entry.parameter_effects:
                if effect.recommended_reasoning_hint and effect.recommended_reasoning_hint not in reasoning_hints:
                    reasoning_hints.append(effect.recommended_reasoning_hint)

        if query_bundle:
            requested = set(query_bundle.parameter_categories)
            if {"current", "voltage", "travel_speed"}.intersection(requested):
                hint = "Check whether the current-voltage-travel speed combination implies excessive heat input."
                if hint not in reasoning_hints:
                    reasoning_hints.append(hint)
            if "shielding_gas" in requested:
                hint = "Evaluate shielding gas flow and composition as a direct contributor to pore formation."
                if hint not in reasoning_hints:
                    reasoning_hints.append(hint)

        return list(theme_map.values()), reasoning_hints

    def _extract_parameter_effects(
        self,
        chunk: ChunkRecord,
        query_bundle: QueryBundle | None = None,
    ) -> list[ParameterEffect]:
        effects: list[ParameterEffect] = []
        candidate_parameters = list(
            dict.fromkeys(
                [
                    *(chunk.process_parameters or []),
                    *(query_bundle.parameter_categories if query_bundle else []),
                ]
            )
        )
        defect = (query_bundle.defect_name if query_bundle and query_bundle.defect_name else None) or (
            chunk.defect_terms[0] if chunk.defect_terms else None
        )
        for parameter in candidate_parameters:
            aliases = PARAMETER_ALIASES.get(parameter, (parameter.replace("_", " "),))
            sentences = [
                sentence
                for sentence in sentence_split(chunk.text)
                if any(alias in normalize_for_match(sentence) for alias in aliases)
            ]
            if not sentences:
                continue
            relationship_text = self._best_parameter_sentence(sentences)
            directionality = self._infer_directionality(relationship_text)
            effects.append(
                ParameterEffect(
                    parameter=parameter,
                    relationship_text=relationship_text,
                    directionality=directionality,
                    effect_on_defect=defect,
                    recommended_reasoning_hint=self._reasoning_hint(parameter, directionality),
                )
            )
        return effects

    def _select_sentence(self, sentences: list[str], cue_words: tuple[str, ...]) -> str | None:
        scored_sentences = []
        for sentence in sentences:
            normalized = normalize_for_match(sentence)
            score = sum(1 for cue in cue_words if cue in normalized)
            if score:
                scored_sentences.append((score, sentence.strip()))
        if not scored_sentences:
            return None
        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        return scored_sentences[0][1]

    def _best_parameter_sentence(self, sentences: list[str]) -> str:
        scored = []
        for sentence in sentences:
            normalized = normalize_for_match(sentence)
            score = sum(
                1
                for cue in ("increase", "decrease", "higher", "lower", "too high", "too low", "window", "range")
                if cue in normalized
            )
            scored.append((score, sentence.strip()))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _infer_directionality(self, text: str) -> str:
        normalized = normalize_for_match(text)
        if "non monotonic" in normalized or ("too high" in normalized and "too low" in normalized):
            return "non_monotonic"
        if any(token in normalized for token in ("optimal", "window", "range", "moderate")):
            return "optimal_window"

        high_terms = any(token in normalized for token in ("high ", "higher ", "too high", "excessive"))
        low_terms = any(token in normalized for token in ("low ", "lower ", "too low", "insufficient"))
        increase_terms = any(token in normalized for token in ("increase", "promote", "cause", "lead to", "risk"))
        reduction_terms = any(token in normalized for token in ("reduce", "decrease", "lowered", "minimize", "suppressed"))

        if high_terms and low_terms:
            return "non_monotonic"
        if high_terms and increase_terms:
            return "increase_risk_when_high"
        if low_terms and increase_terms:
            return "increase_risk_when_low"
        if low_terms and reduction_terms:
            return "increase_risk_when_high"
        if high_terms and reduction_terms:
            return "increase_risk_when_low"
        return "unclear"

    def _reasoning_hint(self, parameter: str, directionality: str) -> str | None:
        if directionality == "increase_risk_when_high":
            return f"Check whether {parameter.replace('_', ' ')} is too high relative to the defect mechanism."
        if directionality == "increase_risk_when_low":
            return f"Check whether {parameter.replace('_', ' ')} is too low and creating defect risk."
        if directionality == "non_monotonic":
            return f"Treat {parameter.replace('_', ' ')} as potentially non-monotonic and evaluate both high and low extremes."
        if directionality == "optimal_window":
            return f"Look for an optimal operating window for {parameter.replace('_', ' ')} rather than a one-directional trend."
        return None

    def _primary_evidence_type(self, evidence_types: list[str]) -> str | None:
        priority = ("mitigation", "result", "recommendation", "mechanism", "cause", "definition")
        for label in priority:
            if label in evidence_types:
                return label
        return evidence_types[0] if evidence_types else None

    def _directness_label(self, score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def _confidence(
        self,
        *,
        chunk: ChunkRecord,
        strategy: str | None,
        mechanism: str | None,
        parameter_effects: list[ParameterEffect],
        outcome: str | None,
        directness: str,
    ) -> float:
        score = 0.35
        if strategy:
            score += 0.15
        if mechanism:
            score += 0.12
        if parameter_effects:
            score += 0.15
        if outcome:
            score += 0.12
        if directness == "high":
            score += 0.08
        elif directness == "medium":
            score += 0.04
        if chunk.review_or_experimental == "experimental":
            score += 0.06
        score -= chunk.reference_contamination_score * 0.15
        return round(max(0.0, min(score, 0.98)), 2)

    def _themes_for_entry(self, entry: ExtractedEvidence) -> list[str]:
        themes: list[str] = []
        parameters = set(entry.parameters)
        if {"current", "voltage", "travel_speed"}.intersection(parameters):
            themes.append("optimize heat input-related parameters")
        if "shielding_gas" in parameters or (entry.strategy and "argon" in normalize_for_match(entry.strategy)):
            themes.append("stabilize shielding gas and gas coverage")
        if "wire_feed_speed" in parameters or "pulse_frequency" in parameters:
            themes.append("stabilize metal transfer and deposition behavior")
        if any(effect.directionality == "non_monotonic" for effect in entry.parameter_effects):
            themes.append("treat key process parameters as non-monotonic")
        if not themes and entry.strategy:
            themes.append("apply direct mitigation guidance from literature")
        return themes
