from __future__ import annotations

from typing import Any, Dict, List


def build_snapshot(
    *,
    sport: str,
    data_source: str,
    is_fallback: bool,
    confidence_score: float,
    explanations: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "sport": sport,
        "data_source": data_source,
        "is_fallback": bool(is_fallback),
        "confidence_score": round(float(confidence_score), 3),
        "explanations": explanations,
        "metrics": metrics,
    }
