from __future__ import annotations

import re

from .models import EvidenceBlock, ParsedResume


SECTION_MAP = {
    "summary": "Summary",
    "professional summary": "Summary",
    "executive profile": "Summary",
    "profile": "Summary",
    "objective": "Summary",
    "highlights": "Skills",
    "skill highlights": "Skills",
    "skills": "Skills",
    "technical skills": "Skills",
    "core competencies": "Skills",
    "experience": "Experience",
    "professional experience": "Experience",
    "employment history": "Experience",
    "work experience": "Experience",
    "career history": "Experience",
    "education": "Education",
    "academic background": "Education",
    "certifications": "Certifications",
    "certification": "Certifications",
    "licenses": "Certifications",
    "professional affiliations": "Affiliations",
    "accomplishments": "Accomplishments",
    "awards": "Accomplishments",
}


def _canonical_heading(line: str) -> str | None:
    clean = re.sub(r"[^A-Za-z &/-]", " ", line).strip().lower()
    clean = re.sub(r"\s+", " ", clean)
    if clean in SECTION_MAP:
        return SECTION_MAP[clean]
    return None


def split_page_into_sections(text: str) -> list[tuple[str, str]]:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    sections: list[tuple[str, list[str]]] = [("Page", [])]
    current = 0
    for line in lines:
        heading = _canonical_heading(line)
        if heading and len(line.strip()) <= 40:
            sections.append((heading, []))
            current = len(sections) - 1
            continue
        sections[current][1].append(line)

    blocks = []
    for section, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            blocks.append((section, body))
    return blocks


def make_evidence_blocks(resume: ParsedResume) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    for page_num, page_text in enumerate(resume.pages, start=1):
        page_sections = split_page_into_sections(page_text)
        if not page_sections and page_text.strip():
            page_sections = [("Page", page_text.strip())]
        for section, body in page_sections:
            for idx, chunk in enumerate(_split_long_block(body), start=1):
                block_id = f"{resume.resume_id}:{page_num}:{section}:{idx}"
                blocks.append(
                    EvidenceBlock(
                        block_id=block_id,
                        resume_id=resume.resume_id,
                        domain=resume.domain,
                        source_pdf=resume.source_pdf,
                        section=section,
                        page=page_num,
                        text=chunk,
                    )
                )
    if not blocks and resume.text.strip():
        blocks.append(
            EvidenceBlock(
                block_id=f"{resume.resume_id}:0:Full Text:1",
                resume_id=resume.resume_id,
                domain=resume.domain,
                source_pdf=resume.source_pdf,
                section="Full Text",
                page=0,
                text=resume.text.strip(),
            )
        )
    return blocks


def _split_long_block(text: str, max_chars: int = 2200) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    if chunks:
        return chunks
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
