from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedResume:
    resume_id: str
    domain: str
    source_pdf: str
    parser: str
    pages: list[str]
    text: str
    low_text: bool = False
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class EvidenceBlock:
    block_id: str
    resume_id: str
    domain: str
    source_pdf: str
    section: str
    page: int
    text: str


@dataclass
class CandidateProfile:
    resume_id: str
    domain: str
    source_pdf: str
    current_or_target_role: str = ""
    role_signals: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    years_experience_estimate: float | None = None
    experience_summary: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    uncertain_fields: list[str] = field(default_factory=list)
    extraction_method: str = "heuristic"


@dataclass
class QueryCriteria:
    raw_query: str
    domain: str | None = None
    target_role: str | None = None
    required_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    min_years: float | None = None
    strict: bool = False
    candidate_ids: list[str] = field(default_factory=list)
    intent: str = "shortlist"
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateMatch:
    resume_id: str
    domain: str
    bucket: str
    score: float
    matched_requirements: list[str]
    missing_requirements: list[str]
    uncertain_requirements: list[str]
    compensating_evidence: list[str]
    evidence: list[dict]
    recommendation: str
    profile: dict
