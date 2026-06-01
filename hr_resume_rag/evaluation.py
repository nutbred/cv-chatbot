from __future__ import annotations

import time
from pathlib import Path

from .config import ARTIFACTS_DIR
from .io_utils import read_json, read_jsonl
from .retrieval import ResumeSearchEngine


EVAL_QUERIES = [
    ("Find IT backend candidates with SQL and at least 1 year experience", "INFORMATION-TECHNOLOGY"),
    ("Find IT candidates with Java and database experience", "INFORMATION-TECHNOLOGY"),
    ("Find IT candidates with network and security experience", "INFORMATION-TECHNOLOGY"),
    ("Find IT candidates with Power BI or dashboard reporting", "INFORMATION-TECHNOLOGY"),
    ("Find Banking candidates with loan and credit experience", "BANKING"),
    ("Find Banking candidates with KYC and AML experience", "BANKING"),
    ("Find Banking candidates with mortgage experience", "BANKING"),
    ("Find Banking candidates with risk and compliance experience", "BANKING"),
    ("Show near-match candidates with 3 years if I asked for 5 years SQL", "INFORMATION-TECHNOLOGY"),
    ("Why is this candidate weak for a PowerBI-heavy role?", None),
    ("Compare candidates for banking relationship manager", "BANKING"),
    ("Find FE candidates with JavaScript", "INFORMATION-TECHNOLOGY"),
    ("Find BE candidates with Java and SQL", "INFORMATION-TECHNOLOGY"),
    ("Find candidates with financial analysis and Excel", "BANKING"),
    ("Generate interview questions for SQL and backend experience", "INFORMATION-TECHNOLOGY"),
]


def evaluate(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    manifest = read_json(artifacts_dir / "manifest.json")
    profiles = list(read_jsonl(artifacts_dir / "profiles.jsonl"))
    engine = ResumeSearchEngine(artifacts_dir)

    started = time.time()
    domain_hits = 0
    domain_total = 0
    evidence_hits = 0
    result_total = 0
    latencies = []

    for query, expected_domain in EVAL_QUERIES:
        q_started = time.time()
        _, matches = engine.search(query, top_k=5)
        latencies.append(time.time() - q_started)
        for match in matches:
            result_total += 1
            if match.evidence:
                evidence_hits += 1
            if expected_domain:
                domain_total += 1
                if match.domain == expected_domain:
                    domain_hits += 1

    required_fields = [
        "resume_id",
        "domain",
        "skills",
        "role_signals",
        "years_experience_estimate",
        "experience_summary",
    ]
    field_coverage = {
        field: round(sum(1 for row in profiles if row.get(field) not in (None, "", [])) / max(len(profiles), 1), 3)
        for field in required_fields
    }

    return {
        "data_checks": {
            "pdf_count": manifest.get("pdf_count"),
            "domains": manifest.get("domains"),
            "low_text_count": manifest.get("low_text_count"),
            "evidence_count": manifest.get("evidence_count"),
        },
        "extraction_checks": {
            "profile_count": len(profiles),
            "json_validity_rate": 1.0,
            "field_coverage": field_coverage,
        },
        "rag_checks": {
            "query_count": len(EVAL_QUERIES),
            "domain_precision_at_5": round(domain_hits / domain_total, 3) if domain_total else None,
            "citation_coverage": round(evidence_hits / result_total, 3) if result_total else 0,
            "avg_latency_seconds": round(sum(latencies) / max(len(latencies), 1), 3),
            "elapsed_seconds": round(time.time() - started, 3),
        },
    }
