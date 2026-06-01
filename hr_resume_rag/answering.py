from __future__ import annotations

from .models import CandidateMatch, QueryCriteria


def compose_answer(criteria: QueryCriteria, matches: list[CandidateMatch]) -> str:
    if not matches:
        mode = "strict" if criteria.strict else "flexible"
        return f"No candidates were found for this query in {mode} mode. Try relaxing the domain, years, or required skills."

    lines = []
    if criteria.warnings:
        lines.append("Clarification note: " + " ".join(criteria.warnings))
        lines.append("")

    mode = "strict" if criteria.strict else "flexible"
    lines.append(f"Screening mode: {mode}. Found {len(matches)} candidate(s).")
    for idx, match in enumerate(matches, start=1):
        profile = match.profile
        years = profile.get("years_experience_estimate")
        years_text = f"{years:g} years" if isinstance(years, (int, float)) else "years unclear"
        lines.append("")
        lines.append(f"{idx}. Candidate {match.resume_id} - {match.bucket} ({match.domain}, {years_text})")
        if profile.get("current_or_target_role"):
            lines.append(f"Role signal: {profile['current_or_target_role']}")
        if match.matched_requirements:
            lines.append("Matched: " + ", ".join(match.matched_requirements[:8]))
        if match.missing_requirements:
            lines.append("Missing/gap: " + ", ".join(match.missing_requirements[:5]))
        if match.uncertain_requirements:
            lines.append("Uncertain: " + ", ".join(match.uncertain_requirements[:4]))
        if match.compensating_evidence:
            lines.append("Compensating evidence: " + ", ".join(match.compensating_evidence[:3]))
        lines.append("Recommendation: " + match.recommendation)
        if match.evidence:
            ev = match.evidence[0]
            lines.append(f"Evidence: [{ev['section']} p.{ev['page']}] {ev['text']}")
    return "\n".join(lines)
