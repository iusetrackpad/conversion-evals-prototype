"""Score worker outputs against ground truth using exact match and an LLM judge."""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

TEST_DATA_PATH = Path("data/test_data.csv")
WORKER_RESULTS_PATH = Path("data/worker_results.csv")
SCORED_PATH = Path("data/scored_results.csv")
SUMMARY_PATH = Path("data/summary.csv")

JUDGE_MODEL = "claude-sonnet-4-5"
CHECKPOINT_EVERY = 10
INVALID = "invalid_output"

JUDGE_PROMPT = """You are evaluating a job-title classifier.

Title: {title}
Ground truth: seniority={true_seniority}, function={true_function}
Worker prediction: seniority={pred_seniority}, function={pred_function}

Score the prediction on a 0-1 scale:
- 1.0 = perfect match
- 0.7-0.9 = defensible alternative interpretation
- 0.3-0.6 = wrong but understandable
- 0.0-0.2 = clearly wrong

Respond with ONLY a JSON object — no markdown fences, no preamble.
Format: {{"score": <float 0-1>, "reason": "<one sentence>"}}
"""


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def prompt_overwrite(path: Path) -> str:
    response = input(f"{path} already exists. [o]verwrite / [s]kip / [a]bort: ").strip().lower()
    if response in ("o", "overwrite"):
        return "overwrite"
    if response in ("s", "skip"):
        return "skip"
    return "abort"


def call_judge(client: Anthropic, title: str, true_sen: str, true_fn: str, pred_sen: str, pred_fn: str):
    """Returns (score, reason). score is None on failure."""
    prompt = JUDGE_PROMPT.format(
        title=title,
        true_seniority=true_sen,
        true_function=true_fn,
        pred_seniority=pred_sen,
        pred_function=pred_fn,
    )
    try:
        msg = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return None, f"judge api error: {e}"

    raw = strip_fences(msg.content[0].text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"judge parse error: {e}"

    score = data.get("score")
    reason = data.get("reason", "")
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None, f"judge score not numeric: {score!r}"
    score = max(0.0, min(1.0, score))
    return score, reason


def save_scored(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "title",
            "worker_model",
            "true_seniority",
            "true_function",
            "predicted_seniority",
            "predicted_function",
            "exact_match",
            "judge_score",
            "judge_reason",
            "latency_ms",
            "cost_usd",
        ],
    ).to_csv(path, index=False)


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    for path in (TEST_DATA_PATH, WORKER_RESULTS_PATH):
        if not path.exists():
            print(f"ERROR: Required input not found: {path}", file=sys.stderr)
            sys.exit(1)

    if SCORED_PATH.exists():
        choice = prompt_overwrite(SCORED_PATH)
        if choice == "abort":
            print("Aborted.")
            sys.exit(0)
        if choice == "skip":
            print(f"Skipping. Existing {SCORED_PATH} kept.")
            sys.exit(0)

    test_df = pd.read_csv(TEST_DATA_PATH)
    workers_df = pd.read_csv(WORKER_RESULTS_PATH)
    merged = workers_df.merge(test_df, on="title", how="left")

    client = Anthropic(api_key=api_key)

    rows: list[dict] = []
    total = len(merged)

    for i, r in enumerate(merged.itertuples(index=False), start=1):
        true_sen = r.true_seniority
        true_fn = r.true_function
        pred_sen = r.predicted_seniority
        pred_fn = r.predicted_function

        exact_match = int(pred_sen == true_sen and pred_fn == true_fn)

        if pred_sen == INVALID or pred_fn == INVALID:
            judge_score = None
            judge_reason = "skipped - invalid worker output"
        else:
            judge_score, judge_reason = call_judge(client, r.title, true_sen, true_fn, pred_sen, pred_fn)

        rows.append(
            {
                "title": r.title,
                "worker_model": r.worker_model,
                "true_seniority": true_sen,
                "true_function": true_fn,
                "predicted_seniority": pred_sen,
                "predicted_function": pred_fn,
                "exact_match": exact_match,
                "judge_score": judge_score,
                "judge_reason": judge_reason,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
            }
        )

        judge_display = f"{judge_score:.2f}" if judge_score is not None else "N/A"
        print(
            f"[{i}/{total}] {r.worker_model}: '{r.title}' → "
            f"exact={exact_match}, judge={judge_display}"
        )

        if i % CHECKPOINT_EVERY == 0:
            save_scored(rows, SCORED_PATH)

    save_scored(rows, SCORED_PATH)

    # Per-worker aggregates
    scored_df = pd.DataFrame(rows)
    summary_rows = []
    for worker_name, group in scored_df.groupby("worker_model"):
        num_calls = len(group)
        accuracy = group["exact_match"].mean() * 100
        judge_scores = group["judge_score"].dropna()
        avg_judge = judge_scores.mean() if len(judge_scores) else float("nan")
        avg_latency = group["latency_ms"].mean()
        total_cost = group["cost_usd"].sum()
        cost_per_1k = (total_cost / num_calls) * 1000 if num_calls else 0.0
        summary_rows.append(
            {
                "worker_model": worker_name,
                "num_calls": num_calls,
                "accuracy_pct": accuracy,
                "avg_judge_score": avg_judge,
                "avg_latency_ms": avg_latency,
                "total_cost_usd": total_cost,
                "cost_per_1k_usd": cost_per_1k,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    # Pretty comparison table
    print()
    header = f"{'Model':<18} | {'Accuracy':<8} | {'Avg Judge Score':<15} | {'Avg Latency':<11} | {'Cost per 1K':<12}"
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        judge_str = f"{s['avg_judge_score']:.2f}" if pd.notna(s["avg_judge_score"]) else "N/A"
        print(
            f"{s['worker_model']:<18} | "
            f"{s['accuracy_pct']:>6.0f}%  | "
            f"{judge_str:<15} | "
            f"{s['avg_latency_ms']:>7.0f}ms   | "
            f"${s['cost_per_1k_usd']:.4f}"
        )

    print()
    print(f"Row-level results: {SCORED_PATH}")
    print(f"Summary:           {SUMMARY_PATH}")


if __name__ == "__main__":
    main()