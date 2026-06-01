from __future__ import annotations

import re
from collections import defaultdict


CANONICAL_ALIASES: dict[str, list[str]] = {
    "Backend Engineering": ["backend", "back-end", "back end", "backend engineer", "server-side", "server side"],
    "Frontend Engineering": ["frontend", "front-end", "front end", "frontend engineer", "ui engineer"],
    "Full Stack": ["full stack", "full-stack", "fullstack"],
    "Artificial Intelligence": ["artificial intelligence", "ai engineer", "ai"],
    "Machine Learning": ["machine learning", "ml engineer", "ml"],
    "Data Science": ["data science", "data scientist", "ds"],
    "Data Engineering": ["data engineering", "data engineer", "de"],
    "Software Engineering": ["software engineer", "software developer", "developer", "programmer"],
    "Quality Assurance": ["quality assurance", "qa", "tester", "testing"],
    "Business Analyst": ["business analyst", "ba"],
    "DevOps": ["devops", "ci/cd", "cicd"],
    "SRE": ["site reliability", "sre"],
    "Project Management": ["project management", "project manager", "pm", "scrum master"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sql server", "t-sql", "pl/sql"],
    "Python": ["python"],
    "Java": ["java", "spring", "spring boot"],
    "JavaScript": ["javascript", "js", "node.js", "nodejs", "react", "angular", "vue"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", ".net", "dotnet"],
    "Linux": ["linux", "unix"],
    "Windows Server": ["windows server", "active directory"],
    "Networking": ["network", "networking", "tcp/ip", "lan", "wan", "router", "switching"],
    "Cybersecurity": ["cybersecurity", "security", "information security", "infosec"],
    "Cloud": ["cloud", "aws", "azure", "gcp"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Oracle": ["oracle"],
    "Power BI": ["powerbi", "power bi", "business intelligence", "bi", "dashboard", "dashboards", "reporting"],
    "Excel": ["excel", "microsoft excel", "ms excel"],
    "Loan Operations": ["loan", "loans", "lending", "loan officer"],
    "Credit Analysis": ["credit", "credit analysis", "credit analyst"],
    "Mortgage": ["mortgage"],
    "Risk Management": ["risk", "risk management", "risk analyst"],
    "Compliance": ["compliance", "regulatory"],
    "KYC": ["kyc", "know your customer"],
    "AML": ["aml", "anti-money laundering", "anti money laundering"],
    "Relationship Management": ["relationship manager", "rm", "client relationship"],
    "Branch Banking": ["branch banking", "branch manager", "bank teller", "teller"],
    "Cash Handling": ["cash handling", "cash management"],
    "Financial Analysis": ["financial analysis", "financial analyst", "finance analysis"],
    "Investment": ["investment", "portfolio"],
    "Foreign Exchange": ["foreign exchange", "fx"],
    "CASA": ["casa", "current account savings account"],
    "Letter of Credit": ["letter of credit", "lc"],
}

AMBIGUOUS_ABBREVIATIONS = {
    "BE": ["Backend Engineering", "Bachelor of Engineering"],
    "BA": ["Business Analyst", "Bachelor of Arts"],
    "PM": ["Project Management", "Product Manager"],
    "DE": ["Data Engineering", "Diploma/Degree context"],
    "DS": ["Data Science", "Distributed Systems"],
}

ROLE_TERMS = [
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
    "SRE",
    "Project Management",
    "Loan Operations",
    "Credit Analysis",
    "Risk Management",
    "Compliance",
    "Relationship Management",
    "Branch Banking",
    "Financial Analysis",
]


def _alias_index() -> dict[str, str]:
    index = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        index[canonical.lower()] = canonical
        for alias in aliases:
            index[alias.lower()] = canonical
    return index


ALIAS_INDEX = _alias_index()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    if re.search(r"[+#./-]", alias):
        return alias.lower() in text.lower()
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text, flags=re.I) is not None


def find_terms(text: str) -> list[str]:
    found: dict[str, int] = {}
    lowered = text.lower()
    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in [canonical, *aliases]:
            if _contains_alias(lowered, alias.lower()):
                found[canonical] = min(found.get(canonical, 999), len(alias))
                break
    return sorted(found, key=lambda term: (term not in ROLE_TERMS, term.lower()))


def normalize_terms(raw_terms: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for term in raw_terms:
        key = normalize_text(term).lower()
        canonical = ALIAS_INDEX.get(key, normalize_text(term))
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def detect_abbreviation_warnings(query: str) -> list[str]:
    warnings = []
    for abbr, meanings in AMBIGUOUS_ABBREVIATIONS.items():
        if re.search(rf"(?<![A-Za-z0-9]){abbr}(?![A-Za-z0-9])", query):
            if abbr == "BE" and re.search(r"(?<![A-Za-z0-9])B\.E\.?(?![A-Za-z0-9])", query):
                warnings.append("Interpreted B.E./BE in education context as Bachelor of Engineering; verify if you meant backend engineering.")
            else:
                warnings.append(f"{abbr} is ambiguous: {', '.join(meanings)}. The assistant uses context but HR should confirm if ranking depends on it.")
    return warnings


def query_terms_with_context(query: str) -> list[str]:
    terms = find_terms(query)
    if re.search(r"(?<![A-Za-z0-9])BE(?![A-Za-z0-9])", query):
        education_context = re.search(r"\b(bachelor|degree|education|university|college)\b", query, flags=re.I) or re.search(
            r"(?<![A-Za-z0-9])B\.E\.?(?![A-Za-z0-9])", query
        )
        if education_context:
            terms = [term for term in terms if term != "Backend Engineering"]
        elif "Backend Engineering" not in terms:
            terms.append("Backend Engineering")
    if re.search(r"(?<![A-Za-z0-9])FE(?![A-Za-z0-9])", query) and "Frontend Engineering" not in terms:
        terms.append("Frontend Engineering")
    return normalize_terms(terms)


def group_terms_by_domain(terms: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    banking = {
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
        "Foreign Exchange",
        "CASA",
        "Letter of Credit",
    }
    for term in terms:
        buckets["BANKING" if term in banking else "INFORMATION-TECHNOLOGY"].append(term)
    return dict(buckets)
