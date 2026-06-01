from __future__ import annotations

import time
from pathlib import Path

from .config import ARTIFACTS_DIR, DATA_DIR, ensure_artifacts_dir
from .extraction import extract_profile
from .indexing import build_faiss_index
from .io_utils import write_json, write_jsonl
from .parsing import discover_pdfs, parse_pdf
from .sections import make_evidence_blocks


def build_artifacts(
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    use_llama_parse: bool = False,
    use_llm_profile: bool = False,
    limit: int | None = None,
    build_index: bool = True,
) -> dict:
    artifacts_dir = ensure_artifacts_dir(artifacts_dir)
    started = time.time()
    pdfs = discover_pdfs(data_dir)
    if limit:
        pdfs = pdfs[:limit]

    parsed = []
    evidence = []
    profiles = []
    low_text_count = 0
    warnings: list[dict] = []

    for path, domain in pdfs:
        resume = parse_pdf(path, domain, use_llama_parse=use_llama_parse)
        blocks = make_evidence_blocks(resume)
        profile = extract_profile(resume, blocks, use_llm=use_llm_profile)
        parsed.append(resume)
        evidence.extend(blocks)
        profiles.append(profile)
        low_text_count += int(resume.low_text)
        if resume.parse_warnings:
            warnings.append({"resume_id": resume.resume_id, "warnings": resume.parse_warnings})

    write_jsonl(artifacts_dir / "documents.jsonl", parsed)
    write_jsonl(artifacts_dir / "evidence.jsonl", evidence)
    write_jsonl(artifacts_dir / "profiles.jsonl", profiles)

    manifest = {
        "pdf_count": len(pdfs),
        "profile_count": len(profiles),
        "evidence_count": len(evidence),
        "low_text_count": low_text_count,
        "domains": {
            "BANKING": sum(1 for _, domain in pdfs if domain == "BANKING"),
            "INFORMATION-TECHNOLOGY": sum(1 for _, domain in pdfs if domain == "INFORMATION-TECHNOLOGY"),
        },
        "use_llama_parse": use_llama_parse,
        "use_llm_profile": use_llm_profile,
        "warnings": warnings[:50],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    write_json(artifacts_dir / "manifest.json", manifest)

    if build_index:
        manifest["index"] = build_faiss_index(artifacts_dir / "evidence.jsonl", artifacts_dir)
        write_json(artifacts_dir / "manifest.json", manifest)
    return manifest
