from __future__ import annotations

from waam_rag.schemas import ProcessParameters, QueryRequest


def test_process_parameter_schema_accepts_legacy_and_unit_qualified_keys() -> None:
    legacy = ProcessParameters.model_validate(
        {
            "current": 180,
            "voltage": 24,
            "travel_speed": 6.5,
            "wire_feed_speed": 7.0,
            "shielding_gas": "argon",
        }
    )
    unit_qualified = ProcessParameters.model_validate(
        {
            "current_A": 180,
            "voltage_V": 24,
            "travel_speed_mm_s": 6.5,
            "wire_feed_speed_m_min": 7.0,
            "shielding_gas": "argon",
        }
    )

    assert legacy.current == unit_qualified.current == 180
    assert legacy.normalized_values()["current_A"] == 180
    assert unit_qualified.normalized_values()["travel_speed_mm_s"] == 6.5


def test_query_request_remains_backward_compatible_with_old_process_payload() -> None:
    request = QueryRequest.model_validate(
        {
            "defect_name": "porosity",
            "process_parameters": {
                "current": 180,
                "voltage": 24,
                "shielding_gas": "argon",
            },
            "top_k": 4,
        }
    )

    assert request.process_parameters is not None
    assert request.process_parameters.current == 180
