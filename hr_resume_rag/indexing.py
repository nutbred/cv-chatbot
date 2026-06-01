from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from .config import EMBEDDING_MODEL
from .io_utils import read_jsonl, write_json
from .models import EvidenceBlock


def build_faiss_index(evidence_path: Path, artifacts_dir: Path, model_name: str = EMBEDDING_MODEL) -> dict:
    evidence = list(read_jsonl(evidence_path))
    texts = [_evidence_text(row) for row in evidence]
    embeddings, backend = embed_texts(texts, model_name)

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required. Install with: python -m pip install faiss-cpu") from exc

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / "faiss.index"))
    bm25_meta = build_bm25_index(evidence, artifacts_dir)

    metadata = {
        "embedding_model": model_name,
        "embedding_backend": backend,
        "evidence_count": len(evidence),
        "embedding_dim": int(embeddings.shape[1]),
        "bm25": bm25_meta,
    }
    write_json(artifacts_dir / "index_manifest.json", metadata)
    return metadata


def load_faiss_index(artifacts_dir: Path):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required. Install with: python -m pip install faiss-cpu") from exc
    return faiss.read_index(str(artifacts_dir / "faiss.index"))


def build_bm25_index(evidence: list[dict], artifacts_dir: Path) -> dict:
    from rank_bm25 import BM25Okapi

    texts = [_evidence_text(row) for row in evidence]
    tokenized = [tokenize_for_bm25(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    with (artifacts_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "doc_count": len(evidence)}, f)
    return {"enabled": True, "doc_count": len(evidence)}


def load_bm25_index(artifacts_dir: Path):
    path = artifacts_dir / "bm25.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f).get("bm25")


def tokenize_for_bm25(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9+#./-]*", text)
    aliases = []
    joined = " ".join(tokens)
    if "power bi" in joined or "powerbi" in joined:
        aliases.append("powerbi")
    if "anti money laundering" in joined or "anti-money laundering" in text:
        aliases.append("aml")
    if "know your customer" in joined:
        aliases.append("kyc")
    if "back end" in joined or "back-end" in text:
        aliases.append("backend")
    if "front end" in joined or "front-end" in text:
        aliases.append("frontend")
    return tokens + aliases


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL) -> tuple[np.ndarray, str]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(embeddings), "sentence-transformers"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=2048, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        artifacts = Path("artifacts")
        artifacts.mkdir(exist_ok=True)
        with (artifacts / "tfidf_vectorizer.pkl").open("wb") as f:
            pickle.dump(vectorizer, f)
        return matrix.toarray().astype("float32"), "tfidf-fallback"


def embed_query(query: str, artifacts_dir: Path, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    manifest_path = artifacts_dir / "index_manifest.json"
    backend = "sentence-transformers"
    if manifest_path.exists():
        import json

        backend = json.loads(manifest_path.read_text(encoding="utf-8")).get("embedding_backend", backend)
    if backend == "tfidf-fallback":
        with (artifacts_dir / "tfidf_vectorizer.pkl").open("rb") as f:
            vectorizer = pickle.load(f)
        return vectorizer.transform([query]).toarray().astype("float32")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode([query], convert_to_numpy=True, normalize_embeddings=False)).astype("float32")


def _evidence_text(row: dict | EvidenceBlock) -> str:
    if isinstance(row, EvidenceBlock):
        return f"{row.section}\n{row.text}"
    return f"{row.get('section', '')}\n{row.get('text', '')}"
