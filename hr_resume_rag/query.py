from __future__ import annotations

import re

from .glossary import detect_abbreviation_warnings, query_terms_with_context
from .models import QueryCriteria


def interpret_query(query: str, strict: bool = False) -> QueryCriteria:
    lowered = query.lower()
    domain = None
    if re.search(r"\b(banking|bank|loan|credit|kyc|aml|mortgage)\b", lowered):
        domain = "BANKING"
    if re.search(r"\b(it|information technology|software|backend|frontend|developer|network|cyber|cloud|ai|ml)\b", lowered):
        domain = "INFORMATION-TECHNOLOGY"

    min_years = None
    year_match = re.search(r"\b(?:at least|min(?:imum)?|>=|over)?\s*(\d{1,2})(?:\+)?\s+(?:years?|yrs?)\b", lowered)
    if year_match:
        min_years = float(year_match.group(1))

    candidate_ids = re.findall(r"\b\d{6,10}\b", query)
    intent = "shortlist"
    if re.search(r"\b(compare|versus|vs)\b", lowered):
        intent = "compare"
    elif re.search(r"\b(why|weak|not suitable|reject|lacking|missing)\b", lowered):
        intent = "explain"
    elif re.search(r"\b(interview|questions)\b", lowered):
        intent = "interview"
    elif re.search(r"\b(upload|classify|closer to)\b", lowered):
        intent = "classify"

    query_strict = strict or bool(re.search(r"\b(strict|hard requirement|must have|exclude|do not show)\b", lowered))
    terms = query_terms_with_context(query)
    if not domain and any(
        term in terms
        for term in {
            "Backend Engineering",
            "Frontend Engineering",
            "Full Stack",
            "Software Engineering",
            "Data Engineering",
            "Data Science",
            "Artificial Intelligence",
            "Machine Learning",
            "DevOps",
            "SRE",
            "Quality Assurance",
            "SQL",
            "Python",
            "Java",
            "JavaScript",
            "C++",
            "C#",
            "Linux",
            "Windows Server",
            "AWS",
            "Azure",
            "Oracle",
        }
    ):
        domain = "INFORMATION-TECHNOLOGY"
    if not domain and any(
        term in terms
        for term in {
            "Loan Operations",
            "Credit Analysis",
            "Mortgage",
            "Risk Management",
            "Compliance",
            "KYC",
            "AML",
            "Relationship Management",
            "Branch Banking",
            "Cash Handling",
            "Financial Analysis",
            "Investment",
        }
    ):
        domain = "BANKING"
    nice_terms = _extract_nice_to_have(query, terms)
    required = [term for term in terms if term not in nice_terms]

    return QueryCriteria(
        raw_query=query,
        domain=domain,
        target_role=_target_role(required),
        required_skills=required,
        nice_to_have_skills=nice_terms,
        min_years=min_years,
        strict=query_strict,
        candidate_ids=candidate_ids,
        intent=intent,
        warnings=detect_abbreviation_warnings(query),
    )


def _extract_nice_to_have(query: str, terms: list[str]) -> list[str]:
    lowered = query.lower()
    if not re.search(r"\b(optional|nice to have|plus|bonus|preferred)\b", lowered):
        return []
    nice = []
    optional_slice = re.split(r"\b(optional|nice to have|plus|bonus|preferred)\b", query, flags=re.I)[-1]
    optional_terms = set(query_terms_with_context(optional_slice))
    for term in terms:
        if term in optional_terms:
            nice.append(term)
    return nice


def _target_role(terms: list[str]) -> str | None:
    for role in [
        "Backend Engineering",
        "Frontend Engineering",
        "Full Stack",
        "Artificial Intelligence",
        "Machine Learning",
        "Data Science",
        "Data Engineering",
        "Software Engineering",
        "Quality Assurance",
        "Business Analyst",
        "DevOps",
        "Project Management",
        "Relationship Management",
        "Branch Banking",
        "Financial Analysis",
    ]:
        if role in terms:
            return role
    return None
