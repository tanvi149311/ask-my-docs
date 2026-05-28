"""Command-line interface: `ingest` and `ask`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ask_my_docs.pipeline import ask, ingest


def _cmd_ingest(args: argparse.Namespace) -> int:
    n = ingest(Path(args.docs_dir))
    print(f"Ingested {n} chunks from {args.docs_dir}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    answer = ask(args.question)
    print("\n=== Answer ===")
    print(answer.text)
    print("\n=== Citations ===")
    print(", ".join(answer.citations) if answer.citations else "(none)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ask-my-docs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index documents from a directory")
    p_ingest.add_argument("docs_dir", help="Directory containing .txt/.md/.pdf files")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against indexed documents")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
