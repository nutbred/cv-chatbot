from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .answering import compose_answer
from .config import ARTIFACTS_DIR, DATA_DIR
from .evaluation import evaluate
from .indexing import build_faiss_index
from .io_utils import read_json, write_json
from .llm import generate_hr_screening_answer
from .pipeline import build_artifacts
from .retrieval import ResumeSearchEngine


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="HR Resume RAG Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Parse PDFs, extract profiles, and build the vector index")
    build.add_argument("--data-dir", type=Path, default=DATA_DIR)
    build.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    build.add_argument("--use-llama-parse", action="store_true")
    build.add_argument("--use-llm-profile", action="store_true")
    build.add_argument("--limit", type=int)
    build.add_argument("--no-index", action="store_true")

    ingest = sub.add_parser("ingest", help="Parse PDFs and extract profiles without building FAISS")
    ingest.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ingest.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ingest.add_argument("--use-llama-parse", action="store_true")
    ingest.add_argument("--use-llm-profile", action="store_true")
    ingest.add_argument("--limit", type=int)

    index = sub.add_parser("index", help="Build FAISS index from existing evidence artifacts")
    index.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)

    ask = sub.add_parser("ask", help="Ask an HR screening question")
    ask.add_argument("query")
    ask.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ask.add_argument("--strict", action="store_true")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--use-llm", action="store_true", help="Use configured DeepSeek/OpenAI-compatible LLM to write the final answer")

    stats = sub.add_parser("stats", help="Print artifact manifest")
    stats.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)

    eval_cmd = sub.add_parser("evaluate", help="Run built-in evaluation queries")
    eval_cmd.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)

    serve = sub.add_parser("serve", help="Launch Streamlit app")
    serve.add_argument("--port", type=int, default=8501)

    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_artifacts(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            use_llama_parse=args.use_llama_parse,
            use_llm_profile=args.use_llm_profile,
            limit=args.limit,
            build_index=not args.no_index,
        )
        print_json(manifest)
        return 0
    if args.command == "ingest":
        manifest = build_artifacts(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            use_llama_parse=args.use_llama_parse,
            use_llm_profile=args.use_llm_profile,
            limit=args.limit,
            build_index=False,
        )
        print_json(manifest)
        return 0
    if args.command == "index":
        meta = build_faiss_index(args.artifacts_dir / "evidence.jsonl", args.artifacts_dir)
        manifest_path = args.artifacts_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest["index"] = meta
            write_json(manifest_path, manifest)
        print_json(meta)
        return 0
    if args.command == "ask":
        engine = ResumeSearchEngine(args.artifacts_dir)
        criteria, matches = engine.search(args.query, strict=args.strict, top_k=args.top_k)
        answer = generate_hr_screening_answer(criteria, matches) if args.use_llm else None
        print(answer or compose_answer(criteria, matches))
        return 0
    if args.command == "stats":
        print_json(read_json(args.artifacts_dir / "manifest.json"))
        return 0
    if args.command == "evaluate":
        print_json(evaluate(args.artifacts_dir))
        return 0
    if args.command == "serve":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(args.port)])
    return 1


def print_json(data: object) -> None:
    import json

    print(json.dumps(data, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    raise SystemExit(main())
