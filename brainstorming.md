# HR Resume RAG Brainstorming

This document preserves the product thinking behind the HR resume assistant so future implementation work does not lose the reasoning context.

## Product Direction

- Build a chat-first HR assistant, not a technical search dashboard.
- Use structured candidate profiles plus section/page evidence retrieval.
- Avoid generic semantic chunking as the primary approach because resumes are already short and semi-structured.
- Use hybrid retrieval: embeddings for broad phrasing and BM25 for exact acronyms/skills like SQL, KYC, AML, FE, BE, and Power BI.
- Always explain recommendations with CV evidence, matched requirements, missing requirements, and uncertainty.
- Default to flexible screening because HR often wants promising stretch candidates, but provide strict mode for hard requirements.

## Use Cases Considered

- Skill and domain search: "Find IT candidates with backend, SQL, and 1+ year experience."
- Banking screening: "Find candidates with loan, credit, KYC, AML, or compliance experience."
- Missing-skill explanation: "Why is this candidate weak for a PowerBI-heavy role?"
- Near-match recommendation: "Show candidates below 5 years if their projects are highly aligned."
- Candidate comparison: "Compare candidates 123, 456, and 789 for a junior backend role."
- Interview question generation: "Generate questions to verify SQL and dashboarding experience."
- Uploaded-CV classification: "Is this uploaded CV closer to Banking, IT, or neither?"
- Transferable fit: "Which banking candidates could transfer into fintech operations or risk analyst roles?"
- HR-friendly summarization: "Summarize this CV without technical jargon."
- Shortlist generation: "Give me 5 candidates and explain the tradeoffs."

## Edge Cases Considered

- Candidate has 3 years of experience but HR asks for 5 years; system should label as near-match only if there is strong compensating evidence.
- Skill appears only in education/coursework, not work experience; system should reduce confidence and explain that distinction.
- Candidate has the right tools but the wrong domain.
- Candidate has the right domain but misses core tools.
- Candidate has strong aligned projects but unclear dates or missing years of experience.
- Resume has missing dates, overlapping dates, or "Current" roles that make years approximate.
- Resume has scanned or low-text pages; local parsing may fail and LlamaParse can be used as optional enhancement.
- Company names are often anonymized as "Company Name" in this dataset, so company prestige should not be a primary signal.
- PowerBI, Power BI, business intelligence, BI, reporting, and dashboarding should be treated as related but not identical.
- Lowercase "be" should not count as backend evidence.
- Abbreviations may be ambiguous across sections and domains.

## Abbreviation Handling

Maintain a small domain glossary and normalize aliases into canonical terms. Use section context to disambiguate when possible and surface clarification when ambiguity affects ranking.

Important examples:

- `FE`: frontend engineer, front-end, frontend.
- `BE`: backend engineer/back-end in skills or role context, but `B.E.` can mean Bachelor of Engineering in education.
- `AI`: artificial intelligence, AI engineer.
- `ML`: machine learning.
- `DS`: data science or data scientist.
- `DE`: data engineer.
- `QA`: quality assurance.
- `BA`: business analyst, but can also mean Bachelor of Arts in education.
- `RM`: relationship manager in banking.
- `KYC`: know your customer.
- `AML`: anti-money laundering.
- `NPA` / `NPL`: non-performing assets/loans.
- `CASA`: current account savings account.
- `FX`: foreign exchange.
- `SME`: small and medium enterprise.
- `LC`: letter of credit.

## Match Buckets

- Strong match: meets domain, required skills, and experience threshold with clear evidence.
- Near match: misses a measurable requirement but has compensating evidence such as aligned projects, higher-responsibility roles, certifications, leadership/scope, or strong tool overlap.
- Not recommended: misses core requirements and lacks meaningful compensating evidence.

## Evaluation Ideas

- Data checks: exactly 235 PDFs, only Banking and IT, no other categories.
- Parsing checks: count low-text PDFs and compare parse coverage against the CSV reference text.
- Extraction checks: JSON validity, required field coverage, manual review of 10 profiles.
- Retrieval checks: domain filter precision, skill evidence coverage, citation coverage, latency.
- Hybrid retrieval checks: acronym-heavy queries should preserve exact matches and return the expected domain.
- Answer quality checks: manual review that every recommendation is backed by evidence.
