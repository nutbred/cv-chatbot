from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .config import ARTIFACTS_DIR, EMBEDDING_MODEL
from .glossary import find_terms
from .indexing import embed_query, load_bm25_index, load_faiss_index, tokenize_for_bm25
from .io_utils import read_jsonl
from .models import CandidateMatch, QueryCriteria
from .query import interpret_query


class ResumeSearchEngine:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.profiles = list(read_jsonl(artifacts_dir / "profiles.jsonl"))
        self.evidence = list(read_jsonl(artifacts_dir / "evidence.jsonl"))
        self.profiles_by_id = {row["resume_id"]: row for row in self.profiles}
        self.evidence_by_resume: dict[str, list[dict]] = {}
        for row in self.evidence:
            self.evidence_by_resume.setdefault(row["resume_id"], []).append(row)
        self.index = load_faiss_index(artifacts_dir)
        self.bm25 = load_bm25_index(artifacts_dir)

    def search(self, query: str, strict: bool = False, top_k: int = 5, criteria: QueryCriteria | None = None) -> tuple[QueryCriteria, list[CandidateMatch]]:
        criteria = criteria or interpret_query(query, strict=strict)
        vector_hits = self._vector_hits(query, top_n=max(30, top_k * 8))
        lexical_hits = self._bm25_hits(query, top_n=max(30, top_k * 8))
        retrieval_score_by_resume: dict[str, float] = {}
        for hit in vector_hits:
            retrieval_score_by_resume[hit["resume_id"]] = max(retrieval_score_by_resume.get(hit["resume_id"], 0.0), hit["score"])
        for hit in lexical_hits:
            retrieval_score_by_resume[hit["resume_id"]] = max(
                retrieval_score_by_resume.get(hit["resume_id"], 0.0),
                0.65 + hit["score"] * 0.35,
            )

        candidate_ids = criteria.candidate_ids or list(self.profiles_by_id)
        scored = []
        for resume_id in candidate_ids:
            profile = self.profiles_by_id.get(resume_id)
            if not profile:
                continue
            match = score_candidate(
                profile,
                self.evidence_by_resume.get(resume_id, []),
                criteria,
                retrieval_score_by_resume.get(resume_id, 0.0),
            )
            if match is None:
                continue
            scored.append(match)

        scored.sort(key=lambda item: (bucket_rank(item.bucket), item.score), reverse=True)
        return criteria, scored[:top_k]

    def _vector_hits(self, query: str, top_n: int) -> list[dict]:
        query_vec = embed_query(query, self.artifacts_dir, EMBEDDING_MODEL).astype("float32")
        try:
            import faiss

            faiss.normalize_L2(query_vec)
        except Exception:
            pass
        scores, indices = self.index.search(query_vec, min(top_n, len(self.evidence)))
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = dict(self.evidence[int(idx)])
            row["score"] = float(score)
            hits.append(row)
        return hits

    def _bm25_hits(self, query: str, top_n: int) -> list[dict]:
        if self.bm25 is None:
            return []
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype="float32")
        if scores.size == 0 or float(scores.max()) <= 0:
            return []
        top_indices = np.argsort(scores)[::-1][: min(top_n, len(scores))]
        max_score = float(scores[top_indices[0]]) or 1.0
        hits = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            row = dict(self.evidence[int(idx)])
            row["score"] = float(scores[idx] / max_score)
            hits.append(row)
        return hits


def score_candidate(profile: dict, evidence: list[dict], criteria: QueryCriteria, vector_score: float) -> CandidateMatch | None:
    if criteria.domain and profile.get("domain") != criteria.domain:
        return None

    matched: list[str] = []
    missing: list[str] = []
    uncertain: list[str] = []
    compensating: list[str] = []
    score = 0.0

    profile_terms = set(profile.get("skills", [])) | set(profile.get("tools", [])) | set(profile.get("role_signals", []))
    profile_text = " ".join(
        [
            profile.get("current_or_target_role", ""),
            " ".join(profile.get("skills", [])),
            " ".join(profile.get("tools", [])),
            " ".join(profile.get("role_signals", [])),
            profile.get("experience_summary", ""),
        ]
    )
    evidence_text = " ".join(row.get("text", "") for row in evidence)
    full_text_terms = set(find_terms(profile_text + " " + evidence_text))
    all_terms = profile_terms | full_text_terms

    if criteria.domain:
        matched.append(f"Domain: {criteria.domain}")
        score += 2.0

    for skill in criteria.required_skills:
        if skill in all_terms:
            matched.append(skill)
            score += 2.0
        else:
            missing.append(skill)
            score -= 1.2

    for skill in criteria.nice_to_have_skills:
        if skill in all_terms:
            matched.append(f"Nice-to-have: {skill}")
            score += 0.8
        else:
            uncertain.append(f"Nice-to-have not found: {skill}")

    years = profile.get("years_experience_estimate")
    below_years = False
    years_unclear = False
    if criteria.min_years is not None:
        if isinstance(years, (int, float)) and years >= criteria.min_years:
            matched.append(f"Experience: {years:g} years >= {criteria.min_years:g}")
            score += 2.0
        elif isinstance(years, (int, float)):
            below_years = True
            missing.append(f"Experience below target: {years:g} years vs {criteria.min_years:g}")
            score -= 1.5
        else:
            years_unclear = True
            uncertain.append(f"Experience threshold unclear: {criteria.min_years:g}+ years requested")
            score -= 0.5

    compensating = compensating_signals(profile, all_terms, evidence_text, criteria)
    score += min(len(compensating), 4) * 0.5
    score += vector_score

    if criteria.strict and (missing or below_years or years_unclear):
        return None

    if missing:
        bucket = "Near match" if compensating and not _missing_core(criteria, missing) else "Not recommended"
    elif years_unclear:
        bucket = "Near match"
    else:
        bucket = "Strong match"
    if not matched and not compensating and criteria.required_skills:
        bucket = "Not recommended"

    selected_evidence = select_evidence(evidence, criteria, all_terms)
    recommendation = recommendation_text(profile, criteria, bucket, matched, missing, uncertain, compensating)
    return CandidateMatch(
        resume_id=profile["resume_id"],
        domain=profile["domain"],
        bucket=bucket,
        score=round(score, 3),
        matched_requirements=matched,
        missing_requirements=missing,
        uncertain_requirements=uncertain,
        compensating_evidence=compensating,
        evidence=selected_evidence,
        recommendation=recommendation,
        profile=profile,
    )


def compensating_signals(profile: dict, terms: set[str], evidence_text: str, criteria: QueryCriteria) -> list[str]:
    signals = []
    role = profile.get("current_or_target_role") or ""
    if criteria.target_role and criteria.target_role in terms:
        signals.append(f"Role signal aligns with {criteria.target_role}")
    if len(set(criteria.required_skills).intersection(terms)) >= max(1, len(criteria.required_skills) - 1):
        signals.append("Most required skills are present")
    if profile.get("certifications"):
        signals.append("Certifications are present")
    if re.search(r"\b(lead|managed|manager|senior|architect|director|supervised|implemented|designed|project)\b", evidence_text, flags=re.I):
        signals.append("Experience shows project scope, ownership, or seniority")
    if role and re.search(r"\b(manager|senior|lead|architect|analyst|engineer)\b", role, flags=re.I):
        signals.append(f"Role/title signal: {role}")
    return signals[:5]


def _missing_core(criteria: QueryCriteria, missing: list[str]) -> bool:
    if criteria.min_years and any("Experience below" in item for item in missing) and not criteria.required_skills:
        return False
    if len(missing) >= max(2, len(criteria.required_skills)):
        return True
    return False


def select_evidence(evidence: list[dict], criteria: QueryCriteria, terms: set[str], max_items: int = 3) -> list[dict]:
    wanted = set(criteria.required_skills + criteria.nice_to_have_skills)
    scored = []
    for row in evidence:
        text = row.get("text", "")
        row_terms = set(find_terms(text))
        score = len(wanted.intersection(row_terms)) * 2 + len(terms.intersection(row_terms))
        if row.get("section") in {"Skills", "Experience", "Summary"}:
            score += 1
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, row in scored[:max_items]:
        selected.append(
            {
                "section": row.get("section"),
                "page": row.get("page"),
                "text": snippet(row.get("text", "")),
                "block_id": row.get("block_id"),
            }
        )
    if not selected:
        for row in evidence[:max_items]:
            selected.append(
                {
                    "section": row.get("section"),
                    "page": row.get("page"),
                    "text": snippet(row.get("text", "")),
                    "block_id": row.get("block_id"),
                }
            )
    return selected


def snippet(text: str, max_chars: int = 420) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def recommendation_text(
    profile: dict,
    criteria: QueryCriteria,
    bucket: str,
    matched: list[str],
    missing: list[str],
    uncertain: list[str],
    compensating: list[str],
) -> str:
    role = profile.get("current_or_target_role") or "candidate"
    if bucket == "Strong match":
        return f"{role} is a strong match because the CV supports the main requested criteria."
    if bucket == "Near match":
        gap = f" Gap: {missing[0]}." if missing else ""
        comp = f" Compensating evidence: {compensating[0]}." if compensating else ""
        return f"{role} is a near match, not a full match.{gap}{comp}"
    reason = missing[0] if missing else (uncertain[0] if uncertain else "limited relevant evidence found")
    return f"{role} is not recommended for this query because {reason}."


def bucket_rank(bucket: str) -> int:
    return {"Strong match": 3, "Near match": 2, "Not recommended": 1}.get(bucket, 0)
