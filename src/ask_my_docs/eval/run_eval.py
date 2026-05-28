"""RAGAS evaluation script.

Usage:
    python -m ask_my_docs.eval.run_eval
    python -m ask_my_docs.eval.run_eval --golden eval/golden_set.json --output eval/results.json
    python -m ask_my_docs.eval.run_eval --assert-thresholds   # CI mode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Thresholds used by --assert-thresholds (CI gate).
THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.80,
}


def _load_golden(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _run_pipeline(question: str) -> tuple[str, list[str]]:
    from ask_my_docs.pipeline import ask

    answer = ask(question)
    contexts = [r.chunk.text for r in answer.contexts]
    return answer.text, contexts


def evaluate(golden_path: Path, output_path: Path | None = None) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print(
            "RAGAS not installed. Run: pip install 'ask-my-docs[eval]'",
            file=sys.stderr,
        )
        sys.exit(1)

    golden = _load_golden(golden_path)

    questions, answers, ground_truths, contexts_list = [], [], [], []

    for item in golden:
        q = item["question"]
        print(f"  evaluating: {q[:70]}...")
        answer_text, ctxs = _run_pipeline(q)
        questions.append(q)
        answers.append(answer_text)
        ground_truths.append(item["ground_truth"])
        contexts_list.append(ctxs if ctxs else [""])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": contexts_list,
        }
    )

    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    df = result.to_pandas()
    scores = (
        df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
        .mean()
        .to_dict()
    )

    report = {"scores": {k: round(float(v), 4) for k, v in scores.items()}, "n": len(golden)}
    print(json.dumps(report, indent=2))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against the golden set")
    parser.add_argument("--golden", default="eval/golden_set.json", help="Path to golden Q&A JSON")
    parser.add_argument("--output", default=None, help="Path to write results JSON")
    parser.add_argument(
        "--assert-thresholds",
        action="store_true",
        help="Exit non-zero if any metric falls below its threshold (CI gate)",
    )
    args = parser.parse_args()

    report = evaluate(
        golden_path=Path(args.golden),
        output_path=Path(args.output) if args.output else None,
    )

    if args.assert_thresholds:
        scores = report["scores"]
        failures = [
            f"{m}={scores.get(m, 0):.3f} < {t}"
            for m, t in THRESHOLDS.items()
            if scores.get(m, 0) < t
        ]
        if failures:
            print(f"\nEVAL CI GATE FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
            sys.exit(1)
        print("\nAll eval thresholds passed ✓")


if __name__ == "__main__":
    main()
