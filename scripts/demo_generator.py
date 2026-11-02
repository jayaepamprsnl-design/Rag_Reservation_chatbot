#!/usr/bin/env python3
"""Run three sample questions through the RAG pipeline and print results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.generator import answer_question

QUESTIONS = [
    ("Where are the parking facilities located and how do I get there by public transport?", []),
    (
        "Do you have EV charging? How much does it cost?",
        [
            {"role": "user", "content": "I drive an electric car."},
            {"role": "assistant", "content": "Great — we have EV charging bays at all three facilities."},
        ],
    ),
    ("Can I get a refund if I cancel my booking last minute?", []),
]


def main() -> None:
    for i, (question, history) in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 70}")
        print(f"Question {i}: {question}")
        if history:
            print(f"  (with {len(history)}-turn chat history)")
        print("-" * 70)

        result = answer_question(question, history)

        print(f"Answer:\n{result['answer']}")
        print(f"\nSources : {', '.join(result['sources']) or '—'}")
        print(f"Chunks  : {len(result['retrieved_chunks'])} retrieved  "
              f"(top score: {result['retrieved_chunks'][0]['score'] if result['retrieved_chunks'] else 'n/a'})")

    print(f"\n{'=' * 70}\nDone.\n")


if __name__ == "__main__":
    main()
