from __future__ import annotations

import re
from datetime import datetime

from .glossary import ROLE_TERMS, find_terms, normalize_terms
from .llm import CompatibleLLM, parse_json_object
from .models import CandidateProfile, EvidenceBlock, ParsedResume


DEGREE_RE = re.compile(
    r"\b(Ph\.?D\.?|Doctorate|Master(?:'s)?|MBA|M\.S\.|MS|Bachelor(?:'s)?|B\.S\.|BS|B\.A\.|BA|B\.E\.|BE|Associate)\b[^.\n]{0,90}",
    flags=re.I,
)
CERT_RE = re.compile(
    r"\b(Certified|Certification|Certificate|CISSP|PMP|CCNA|CPA|Series\s+\d+|Six Sigma|ITIL|Scrum)\b[^.\n]{0,90}",
    flags=re.I,
)


def extract_profile(
    resume: ParsedResume,
    evidence_blocks: list[EvidenceBlock],
    use_llm: bool = False,
) -> CandidateProfile:
    heuristic = heuristic_profile(resume, evidence_blocks)
    if not use_llm:
        return heuristic

    llm_profile = llm_extract_profile(resume, evidence_blocks)
    if not llm_profile:
        return heuristic
    return merge_profiles(heuristic, llm_profile)


def heuristic_profile(resume: ParsedResume, evidence_blocks: list[EvidenceBlock]) -> CandidateProfile:
    text = resume.text
    terms = find_terms(text)
    role_signals = [term for term in terms if term in ROLE_TERMS]
    skills = [term for term in terms if term not in role_signals]
    tools = [term for term in skills if term in {"SQL", "Python", "Java", "JavaScript", "C++", "C#", "Linux", "Windows Server", "AWS", "Azure", "Oracle", "Power BI", "Excel"}]
    education = _extract_section_lines(evidence_blocks, "Education", DEGREE_RE)
    certifications = _extract_section_lines(evidence_blocks, "Certifications", CERT_RE)
    if not certifications:
        certifications = _regex_lines(text, CERT_RE)
    years = estimate_years_experience(text)
    uncertain = []
    if years is None:
        uncertain.append("years_experience_estimate")
    if not role_signals:
        uncertain.append("role_signals")

    evidence_refs = []
    for block in evidence_blocks:
        block_terms = set(find_terms(block.text))
        if block_terms.intersection(terms):
            evidence_refs.append(block.block_id)
    evidence_refs = evidence_refs[:12]

    return CandidateProfile(
        resume_id=resume.resume_id,
        domain=resume.domain,
        source_pdf=resume.source_pdf,
        current_or_target_role=_guess_role(text, role_signals),
        role_signals=role_signals,
        skills=skills,
        tools=tools,
        education=education[:6],
        certifications=certifications[:8],
        years_experience_estimate=years,
        experience_summary=_experience_summary(evidence_blocks),
        evidence_refs=evidence_refs,
        uncertain_fields=uncertain,
        extraction_method="heuristic",
    )


def llm_extract_profile(resume: ParsedResume, evidence_blocks: list[EvidenceBlock]) -> dict | None:
    llm = CompatibleLLM()
    if not llm.available:
        return None
    sample = resume.text[:12000]
    system = (
        "You extract structured HR screening information from resumes. "
        "Return only valid JSON. Be conservative: if a field is not supported by evidence, put it in uncertain_fields."
    )
    user = f"""
Extract a candidate profile from this resume.

Required JSON keys:
resume_id, domain, current_or_target_role, role_signals, skills, tools, education,
certifications, years_experience_estimate, experience_summary, evidence_refs, uncertain_fields.

Resume id: {resume.resume_id}
Domain label from folder: {resume.domain}
Resume text:
{sample}
"""
    parsed = parse_json_object(llm.chat(system, user, json_mode=True))
    return parsed if isinstance(parsed, dict) else None


def merge_profiles(heuristic: CandidateProfile, llm_profile: dict) -> CandidateProfile:
    def list_value(key: str, fallback: list[str]) -> list[str]:
        value = llm_profile.get(key, fallback)
        if isinstance(value, list):
            return normalize_terms([str(item) for item in value if str(item).strip()])
        return fallback

    years = llm_profile.get("years_experience_estimate", heuristic.years_experience_estimate)
    try:
        years = float(years) if years is not None else None
    except (TypeError, ValueError):
        years = heuristic.years_experience_estimate

    return CandidateProfile(
        resume_id=heuristic.resume_id,
        domain=heuristic.domain,
        source_pdf=heuristic.source_pdf,
        current_or_target_role=str(llm_profile.get("current_or_target_role") or heuristic.current_or_target_role),
        role_signals=list_value("role_signals", heuristic.role_signals),
        skills=list_value("skills", heuristic.skills),
        tools=list_value("tools", heuristic.tools),
        education=[str(x) for x in llm_profile.get("education", heuristic.education) if str(x).strip()]
        if isinstance(llm_profile.get("education", heuristic.education), list)
        else heuristic.education,
        certifications=[str(x) for x in llm_profile.get("certifications", heuristic.certifications) if str(x).strip()]
        if isinstance(llm_profile.get("certifications", heuristic.certifications), list)
        else heuristic.certifications,
        years_experience_estimate=years,
        experience_summary=str(llm_profile.get("experience_summary") or heuristic.experience_summary),
        evidence_refs=heuristic.evidence_refs,
        uncertain_fields=[str(x) for x in llm_profile.get("uncertain_fields", heuristic.uncertain_fields)]
        if isinstance(llm_profile.get("uncertain_fields", heuristic.uncertain_fields), list)
        else heuristic.uncertain_fields,
        extraction_method="llm+heuristic",
    )


def estimate_years_experience(text: str) -> float | None:
    text = text or ""
    explicit = []
    for match in re.finditer(r"\b(\d{1,2})(?:\+| plus)?\s+(?:years?|yrs?)\b", text, flags=re.I):
        value = int(match.group(1))
        if 0 < value <= 45:
            explicit.append(value)
    if explicit:
        return float(max(explicit))

    ranges = _date_ranges(text)
    if not ranges:
        return None
    months = 0
    for start, end in ranges:
        if end < start:
            continue
        months += min((end.year - start.year) * 12 + end.month - start.month, 600)
    years = round(months / 12, 1)
    if years <= 0:
        return None
    return min(years, 45.0)


def _date_ranges(text: str) -> list[tuple[datetime, datetime]]:
    months = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
    range_re = re.compile(
        rf"((?:{months})[a-z]*\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})\s*(?:-|to|–|—)\s*((?:{months})[a-z]*\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}|current|present)",
        flags=re.I,
    )
    ranges = []
    for start_raw, end_raw in range_re.findall(text):
        start = _parse_date(start_raw)
        end = _parse_date(end_raw)
        if start and end:
            ranges.append((start, end))
    return ranges[:20]


def _parse_date(raw: str) -> datetime | None:
    raw = raw.strip()
    if re.match(r"current|present", raw, flags=re.I):
        return datetime.now()
    for fmt in ("%m/%Y", "%b %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _extract_section_lines(blocks: list[EvidenceBlock], section: str, regex: re.Pattern) -> list[str]:
    values = []
    for block in blocks:
        if block.section == section:
            values.extend(_regex_lines(block.text, regex))
    return _dedupe(values)


def _regex_lines(text: str, regex: re.Pattern) -> list[str]:
    values = []
    for match in regex.finditer(text or ""):
        values.append(re.sub(r"\s+", " ", match.group(0)).strip(" ,;:-"))
    return _dedupe(values)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = value.lower()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
    return out


def _guess_role(text: str, role_signals: list[str]) -> str:
    if role_signals:
        return role_signals[0]
    for line in text.splitlines()[:12]:
        line = re.sub(r"\s+", " ", line).strip(" ,")
        if 4 <= len(line) <= 80 and not re.match(r"^(summary|skills|experience|education)$", line, flags=re.I):
            return line.title()
    return ""


def _experience_summary(blocks: list[EvidenceBlock]) -> str:
    preferred = [b.text for b in blocks if b.section in {"Summary", "Experience"}]
    text = "\n".join(preferred) if preferred else "\n".join(b.text for b in blocks[:2])
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:3])[:700]
