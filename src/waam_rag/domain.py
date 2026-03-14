"""Domain dictionaries and heuristics for WAAM defect retrieval."""

from __future__ import annotations

DEFECT_SYNONYMS: dict[str, list[str]] = {
    "porosity": ["porosity", "gas porosity", "pore formation", "pores", "voids"],
    "crack": ["crack", "cracking", "hot cracking", "cold cracking", "solidification cracking"],
    "lack of fusion": ["lack of fusion", "incomplete fusion", "poor fusion"],
    "undercut": ["undercut", "edge groove"],
    "humping": ["humping", "bead humping", "hump formation"],
    "spatter": ["spatter", "droplet spatter"],
    "lack of penetration": ["lack of penetration", "incomplete penetration"],
    "distortion": ["distortion", "warping", "residual distortion"],
}

PROCESS_TYPES: dict[str, list[str]] = {
    "waam": ["waam", "wire arc additive manufacturing", "wire + arc additive manufacturing"],
    "gmaw": ["gmaw", "gas metal arc welding", "mig", "mig welding", "cmt"],
    "gtaw": ["gtaw", "gas tungsten arc welding", "tig", "tig welding"],
    "arc_am": ["arc additive manufacturing", "welding additive manufacturing"],
}

PROCESS_PARAMETER_PATTERNS: dict[str, list[str]] = {
    "current": [
        r"\bcurrent\b",
        r"\bwelding current\b",
        r"\barc current\b",
        r"\b\d+(?:\.\d+)?\s*(?:a|amp|amps|amperes)\b",
    ],
    "voltage": [
        r"\bvoltage\b",
        r"\barc voltage\b",
        r"\b\d+(?:\.\d+)?\s*(?:v|volt|volts)\b",
    ],
    "wire_feed_speed": [
        r"\bwire feed speed\b",
        r"\bwire-feed speed\b",
        r"\bwfs\b",
    ],
    "travel_speed": [
        r"\btravel speed\b",
        r"\bwelding speed\b",
        r"\bscan speed\b",
        r"\bdeposition speed\b",
    ],
    "torch_angle": [
        r"\btorch angle\b",
        r"\bgun angle\b",
        r"\belectrode angle\b",
    ],
    "shielding_gas": [
        r"\bshielding gas\b",
        r"\bargon\b",
        r"\bhelium\b",
        r"\bco2\b",
        r"\bgas mixture\b",
    ],
    "heat_input": [
        r"\bheat input\b",
        r"\benergy input\b",
    ],
    "interpass_temperature": [
        r"\binterpass temperature\b",
        r"\binter-layer temperature\b",
    ],
    "layer_height": [
        r"\blayer height\b",
        r"\bdeposited layer height\b",
    ],
    "arc_length": [
        r"\barc length\b",
        r"\bstick-?out\b",
    ],
    "deposition_rate": [
        r"\bdeposition rate\b",
        r"\bmaterial deposition rate\b",
    ],
}

MATERIAL_PATTERNS: dict[str, list[str]] = {
    "aluminum": [r"\baluminum\b", r"\baluminium\b", r"\baa\d{4}\b", r"\b4043\b", r"\b5356\b"],
    "steel": [r"\bsteel\b", r"\b316l\b", r"\bstainless steel\b", r"\bmild steel\b"],
    "titanium": [r"\btitanium\b", r"\bti-6al-4v\b", r"\bti64\b"],
    "nickel": [r"\bnickel\b", r"\binconel\b", r"\bin718\b"],
    "magnesium": [r"\bmagnesium\b", r"\baz91\b"],
}

EVIDENCE_ROLE_PATTERNS: dict[str, list[str]] = {
    "definition": [r"\bis defined as\b", r"\brefers to\b", r"\bcharacterized by\b"],
    "cause": [r"\bcaused by\b", r"\bdue to\b", r"\battributed to\b", r"\boriginates from\b"],
    "mitigation": [r"\bmitigat", r"\breduce", r"\bavoid", r"\bsuppress", r"\bminimiz"],
    "result": [r"\bresults show\b", r"\bwe observed\b", r"\bthe experiments indicate\b"],
    "recommendation": [r"\bwe recommend\b", r"\bit is recommended\b", r"\bshould be\b"],
    "limitation": [r"\blimitation\b", r"\bhowever\b", r"\bchallenge\b"],
    "future work": [r"\bfuture work\b", r"\bfurther study\b", r"\bneeds additional investigation\b"],
}

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "background"),
    "methodology": ("methodology", "materials and methods", "experimental setup", "experimental procedure"),
    "results": ("results", "results and discussion", "discussion"),
    "conclusions": ("conclusion", "conclusions"),
    "recommendations": ("recommendation", "practical implication"),
    "references": ("references", "bibliography"),
}

SKIP_SECTIONS: set[str] = {"references", "bibliography", "acknowledgements", "appendix"}
