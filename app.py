from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from hr_resume_rag.answering import compose_answer
from hr_resume_rag.config import ARTIFACTS_DIR
from hr_resume_rag.extraction import heuristic_profile
from hr_resume_rag.glossary import group_terms_by_domain
from hr_resume_rag.io_utils import read_json
from hr_resume_rag.llm import generate_hr_screening_answer, llm_status
from hr_resume_rag.parsing import parse_pdf
from hr_resume_rag.query import interpret_query
from hr_resume_rag.retrieval import ResumeSearchEngine
from hr_resume_rag.sections import make_evidence_blocks


st.set_page_config(page_title="HR Resume RAG Assistant", layout="wide")
st.title("HR Resume RAG Assistant")


@st.cache_resource(show_spinner=False)
def load_engine() -> ResumeSearchEngine:
    return ResumeSearchEngine(ARTIFACTS_DIR)


def artifacts_ready() -> bool:
    return (ARTIFACTS_DIR / "profiles.jsonl").exists() and (ARTIFACTS_DIR / "faiss.index").exists()


with st.sidebar:
    status = llm_status()
    st.header("Runtime status")
    st.caption(f"LLM: {'configured' if status['available'] else 'not configured'}")
    st.caption(f"LLM model: {status['model']}")
    st.caption(f"LlamaParse: {'configured' if os.getenv('LLAMA_CLOUD_API_KEY') else 'not configured'}")
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        parser_note = "pypdf + optional LlamaParse" if manifest.get("use_llama_parse") else "pypdf"
        profile_note = "LLM + heuristic" if manifest.get("use_llm_profile") else "heuristic"
        bm25_note = manifest.get("index", {}).get("bm25", {}).get("enabled", False)
        st.caption(f"Indexed parser: {parser_note}")
        st.caption(f"Profile extraction: {profile_note}")
        st.caption(f"Retrieval: FAISS + {'BM25' if bm25_note else 'no BM25'}")
    use_llm_answer = st.toggle(
        "Use LLM for final answer",
        value=bool(status["available"]),
        disabled=not bool(status["available"]),
        help="Requires DEEPSEEK_API_KEY or OPENAI_API_KEY. Retrieval and evidence remain deterministic.",
    )
    use_llama_upload = st.toggle(
        "Use LlamaParse for upload fallback",
        value=False,
        disabled=not bool(os.getenv("LLAMA_CLOUD_API_KEY")),
        help="Only affects uploaded PDFs in this UI. Rebuild artifacts with --use-llama-parse for the corpus.",
    )

    st.divider()
    st.header("Screening criteria")
    domain = st.selectbox("Domain", ["Auto", "INFORMATION-TECHNOLOGY", "BANKING"])
    min_years = st.number_input("Minimum years", min_value=0.0, max_value=40.0, value=0.0, step=0.5)
    strict = st.toggle("Strict mode", value=False, help="Exclude candidates who miss hard criteria.")
    required = st.text_input("Required skills", placeholder="SQL, backend, KYC")
    nice = st.text_input("Nice-to-have skills", placeholder="Power BI, AWS")

    st.divider()
    st.subheader("Upload CV")
    uploaded = st.file_uploader("Parse one PDF for quick inspection", type=["pdf"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        parsed = parse_pdf(tmp_path, "UPLOADED", use_llama_parse=use_llama_upload)
        blocks = make_evidence_blocks(parsed)
        profile = heuristic_profile(parsed, blocks)
        grouped = group_terms_by_domain(profile.skills + profile.role_signals)
        st.caption("Quick classification")
        if len(grouped.get("BANKING", [])) > len(grouped.get("INFORMATION-TECHNOLOGY", [])):
            st.write("Closer to: BANKING")
        elif len(grouped.get("INFORMATION-TECHNOLOGY", [])) > 0:
            st.write("Closer to: INFORMATION-TECHNOLOGY")
        else:
            st.write("Closer to: neither/unclear")
        with st.expander("Uploaded profile JSON"):
            st.json(profile.__dict__)


if not artifacts_ready():
    st.warning("Artifacts are not built yet. Run `python -m hr_resume_rag.cli build` first.")
    st.stop()


engine = load_engine()

sample_prompts = [
    "Find IT backend candidates with SQL and at least 1 year experience",
    "Find Banking candidates with loan, credit, KYC, or AML experience",
    "Why is this candidate weak for a PowerBI-heavy role?",
    "Find near-match candidates for 5 years SQL experience",
]

cols = st.columns(len(sample_prompts))
for col, prompt in zip(cols, sample_prompts):
    if col.button(prompt, use_container_width=True):
        st.session_state.pending_prompt = prompt

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask HR screening questions across Banking and IT resumes")
if st.session_state.get("pending_prompt"):
    query = st.session_state.pop("pending_prompt")

if query:
    augmented = query
    criteria_parts = []
    if domain != "Auto":
        criteria_parts.append(f"domain {domain}")
    if min_years:
        criteria_parts.append(f"at least {min_years:g} years")
    if required.strip():
        criteria_parts.append(f"required skills {required}")
    if nice.strip():
        criteria_parts.append(f"nice to have {nice}")
    if criteria_parts:
        augmented = f"{query}. Additional criteria: {', '.join(criteria_parts)}."

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Screening candidates..."):
            criteria = interpret_query(augmented, strict=strict)
            _, matches = engine.search(augmented, strict=strict, top_k=5, criteria=criteria)
            answer = generate_hr_screening_answer(criteria, matches) if use_llm_answer else None
            answer = answer or compose_answer(criteria, matches)
            st.markdown(answer)
            for match in matches:
                with st.expander(f"Candidate {match.resume_id} evidence and profile"):
                    st.write(match.recommendation)
                    st.json(
                        {
                            "bucket": match.bucket,
                            "score": match.score,
                            "matched": match.matched_requirements,
                            "missing": match.missing_requirements,
                            "uncertain": match.uncertain_requirements,
                            "compensating_evidence": match.compensating_evidence,
                            "profile": match.profile,
                            "evidence": match.evidence,
                        }
                    )
    st.session_state.messages.append({"role": "assistant", "content": answer})
