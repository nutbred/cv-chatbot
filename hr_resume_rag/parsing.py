from __future__ import annotations

import os
from pathlib import Path

from .config import ALLOWED_DOMAINS, LOW_TEXT_THRESHOLD
from .models import ParsedResume


def discover_pdfs(data_dir: Path) -> list[tuple[Path, str]]:
    pdfs: list[tuple[Path, str]] = []
    for domain in ALLOWED_DOMAINS:
        domain_dir = data_dir / domain
        if not domain_dir.exists():
            continue
        for path in sorted(domain_dir.glob("*.pdf")):
            pdfs.append((path, domain))
    return pdfs


def parse_pdf(path: Path, domain: str, use_llama_parse: bool = False) -> ParsedResume:
    warnings: list[str] = []
    pages = _parse_with_pypdf(path, warnings)
    parser = "pypdf"
    text = "\n\n".join(page for page in pages if page).strip()
    low_text = len(text) < LOW_TEXT_THRESHOLD

    if low_text and use_llama_parse:
        llama_pages = _parse_with_llama_parse(path, warnings)
        llama_text = "\n\n".join(page for page in llama_pages if page).strip()
        if len(llama_text) > len(text):
            pages = llama_pages
            text = llama_text
            parser = "llamaparse"
            low_text = len(text) < LOW_TEXT_THRESHOLD

    if low_text:
        warnings.append(f"Low parsed text length ({len(text)} chars); scanned or complex PDF possible.")

    return ParsedResume(
        resume_id=path.stem,
        domain=domain,
        source_pdf=str(path),
        parser=parser,
        pages=pages,
        text=text,
        low_text=low_text,
        parse_warnings=warnings,
    )


def _parse_with_pypdf(path: Path, warnings: list[str]) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for local PDF parsing. Install with: python -m pip install pypdf") from exc

    pages: list[str] = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                text = page.extract_text() or ""
            except Exception as exc:
                warnings.append(f"Page extraction failed with pypdf: {exc}")
                text = ""
            pages.append(text.strip())
    except Exception as exc:
        warnings.append(f"pypdf failed for {path.name}: {exc}")
    return pages


def _parse_with_llama_parse(path: Path, warnings: list[str]) -> list[str]:
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        warnings.append("LlamaParse requested but LLAMA_CLOUD_API_KEY is not set.")
        return []
    try:
        from llama_parse import LlamaParse
    except ImportError:
        warnings.append("LlamaParse requested but llama-parse is not installed.")
        return []

    try:
        parser = LlamaParse(result_type="text", split_by_page=True, verbose=False)
        docs = parser.load_data(str(path))
        return [(getattr(doc, "text", "") or "").strip() for doc in docs if getattr(doc, "text", "")]
    except Exception as exc:
        warnings.append(f"LlamaParse failed for {path.name}: {exc}")
        return []
