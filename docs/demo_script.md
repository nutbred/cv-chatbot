# Short Video Demo Script

Target length: 2-3 minutes.

## 0:00-0:20 - Project Intro

Show `README.md`.

Script:

> This is a chat-first HR resume RAG assistant for Banking and Information Technology resumes. It parses CV PDFs, extracts structured candidate profiles, retrieves evidence, and explains strong matches, near matches, and missing requirements.

## 0:20-0:40 - Runtime Status

Open Streamlit:

```powershell
streamlit run app.py
```

Show the sidebar.

Mention:

- LLM status
- parser mode
- FAISS + BM25 hybrid retrieval
- strict/flexible mode

## 0:40-1:10 - Banking Query

Prompt:

```text
Find banking candidates with KYC and AML experience
```

Show:

- ranked candidates
- matched skills
- evidence snippet
- candidate expander

## 1:10-1:40 - Abbreviation Query

Prompt:

```text
Find BE candidates with Java and SQL
```

Show:

- `BE` ambiguity warning
- IT backend interpretation
- Java and SQL evidence

## 1:40-2:15 - Near Match and Strict Mode

Prompt:

```text
Find banking candidates with KYC and AML and at least 20 years experience
```

Show flexible mode first:

- near-match bucket
- missing KYC or experience gap
- compensating evidence

Then enable strict mode and rerun:

- explain that strict mode excludes candidates who cannot prove hard requirements

## 2:15-2:40 - Evaluation and Close

Show README evaluation table.

Mention:

- 235 PDFs indexed
- 115 Banking, 120 IT
- 1,490 evidence blocks
- domain precision@5 = 1.0
- citation coverage = 1.0

Close:

> The key product idea is that HR does not only need search results. They need explainable, evidence-backed screening decisions, including when a candidate is close but not a perfect match.
