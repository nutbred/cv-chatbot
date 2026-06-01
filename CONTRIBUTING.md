# Contributing

## Local Setup

```powershell
python -m pip install -r requirements.txt
copy .env-example .env
```

Fill `.env` with real API keys only on your local machine.

## Common Commands

```powershell
python -m hr_resume_rag.cli build
python -m hr_resume_rag.cli ask "Find banking candidates with KYC and AML experience"
python -m hr_resume_rag.cli evaluate
streamlit run app.py
```

## Data Policy

Do not commit the Kaggle dataset, generated artifacts, `.env`, model indexes, or PDF files. They are intentionally ignored by `.gitignore`.
