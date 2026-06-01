from __future__ import annotations

import json
from typing import Any

from .config import llm_config


class CompatibleLLM:
    def __init__(self) -> None:
        cfg = llm_config()
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]
        self.available = bool(self.api_key)

    def chat(self, system: str, user: str, json_mode: bool = False) -> str | None:
        if not self.available:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "stream": False,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                return None
        return response.choices[0].message.content


def parse_json_object(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def llm_status() -> dict[str, str | bool | None]:
    cfg = llm_config()
    return {
        "available": bool(cfg["api_key"]),
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "api_key_env": "DEEPSEEK_API_KEY or OPENAI_API_KEY",
    }


def generate_hr_screening_answer(criteria, matches) -> str | None:
    """Use the configured OpenAI-compatible LLM to turn grounded matches into an HR answer."""
    llm = CompatibleLLM()
    if not llm.available:
        return None

    payload = {
        "query": criteria.raw_query,
        "interpreted_criteria": {
            "domain": criteria.domain,
            "target_role": criteria.target_role,
            "required_skills": criteria.required_skills,
            "nice_to_have_skills": criteria.nice_to_have_skills,
            "min_years": criteria.min_years,
            "strict": criteria.strict,
            "warnings": criteria.warnings,
        },
        "matches": [
            {
                "resume_id": match.resume_id,
                "domain": match.domain,
                "bucket": match.bucket,
                "score": match.score,
                "matched_requirements": match.matched_requirements,
                "missing_requirements": match.missing_requirements,
                "uncertain_requirements": match.uncertain_requirements,
                "compensating_evidence": match.compensating_evidence,
                "recommendation": match.recommendation,
                "profile": {
                    "current_or_target_role": match.profile.get("current_or_target_role"),
                    "years_experience_estimate": match.profile.get("years_experience_estimate"),
                    "skills": match.profile.get("skills", [])[:20],
                    "tools": match.profile.get("tools", [])[:20],
                    "certifications": match.profile.get("certifications", [])[:10],
                },
                "evidence": match.evidence,
            }
            for match in matches
        ],
    }
    system = (
        "You are an HR resume screening assistant. Use only the provided structured matches and evidence. "
        "Do not invent facts, employers, degrees, years, skills, regulations, market context, or external examples. "
        "Preserve match buckets exactly. "
        "When a candidate is a near match, explicitly state the gap and compensating evidence. "
        "When evidence is uncertain, say what HR should verify using only the uncertainty already present in the payload. "
        "Do not add interview topics that are not supported by the payload. Write concise Markdown."
    )
    user = (
        "Create a recruiter-friendly answer for this resume RAG result. "
        "Include ranked candidates, why each fits or does not fit, missing skills, and evidence references.\n\n"
        + json.dumps(payload, ensure_ascii=True, indent=2)
    )
    return llm.chat(system, user, json_mode=False)
