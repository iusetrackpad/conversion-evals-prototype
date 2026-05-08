"""Run three worker models against the synthetic job titles and record predictions."""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

SENIORITY_LEVELS = ["IC", "Senior IC", "Manager", "Director", "VP", "C-level", "Founder"]
FUNCTIONS = [
    "Engineering",
    "Marketing",
    "Sales",
    "Product",
    "Design",
    "Operations",
    "Finance",
    "HR",
    "Legal",
    "Customer Success",
]

INPUT_PATH = Path("data/test_data.csv")
OUTPUT_PATH = Path("data/worker_results.csv")
CHECKPOINT_EVERY = 10
INVALID = "invalid_output"

# (input_per_million_usd, output_per_million_usd)
WORKERS = [
    {
        "name": "claude-haiku",
        "model": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "price_in": 0.80,
        "price_out": 4.00,
    },
    {
        "name": "gpt-4o-mini",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "price_in": 0.15,
        "price_out": 0.60,
    },
    {
        "name": "claude-sonnet",
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "price_in": 3.00,
        "price_out": 15.00,
    },
]

CLASSIFICATION_PROMPT = """Classify the following job title into seniority and function.

Title: {title}

seniority must be exactly one of: {seniority_levels}
function must be exactly one of: {functions}

Respond with ONLY a JSON object — no markdown code fences, no preamble, no commentary.
Format: {{"seniority": "...", "function": "..."}}
"""


def build_prompt(title: str) -> str:
    return CLASSIFICATION_PROMPT.format(
        title=title,
        seniority_levels=SENIORITY_LEVELS,
        functions=FUNCTIONS,
    )


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def call_anthropic(client: Anthropic, model: str, prompt: str):
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text, msg.usage.input_tokens, msg.usage.output_tokens


def call_openai(client: OpenAI, model: str, prompt: str):
    resp = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


def compute_cost(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def parse_prediction(raw: str):
    """Return (seniority, function). Either may be INVALID if parsing/validation fails."""
    cleaned = strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return INVALID, INVALID
    if not isinstance(data, dict):
        return INVALID, INVALID
    seniority = data.get("seniority", INVALID)
    function = data.get("function", INVALID)
    if seniority not in SENIORITY_LEVELS:
        seniority = INVALID
    if function not in FUNCTIONS:
        function = INVALID
    return seniority, function


def prompt_overwrite(path: Path) -> str:
    """Returns 'overwrite', 'skip', or 'abort'."""
    response = input(f"{path} already exists. [o]verwrite / [s]kip / [a]bort: ").strip().lower()
    if response in ("o", "overwrite"):
        return "overwrite"
    if response in ("s", "skip"):
        return "skip"
    return "abort"


def save_results(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "title",
            "worker_model",
            "predicted_seniority",
            "predicted_function",
            "latency_ms",
            "cost_usd",
            "raw_response",
        ],
    ).to_csv(path, index=False)


def main() -> None:
    load_dotenv()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    missing = [k for k, v in [("ANTHROPIC_API_KEY", anthropic_key), ("OPENAI_API_KEY", openai_key)] if not v]
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}. Run 01_generate_data.py first.", file=sys.stderr)
        sys.exit(1)

    if OUTPUT_PATH.exists():
        choice = prompt_overwrite(OUTPUT_PATH)
        if choice == "abort":
            print("Aborted.")
            sys.exit(0)
        if choice == "skip":
            print(f"Skipping. Existing {OUTPUT_PATH} kept.")
            sys.exit(0)

    df = pd.read_csv(INPUT_PATH)
    titles = df["title"].tolist()

    anthropic_client = Anthropic(api_key=anthropic_key)
    openai_client = OpenAI(api_key=openai_key)

    rows: list[dict] = []
    total_calls = len(titles) * len(WORKERS)
    call_idx = 0

    for title in titles:
        prompt = build_prompt(title)
        for worker in WORKERS:
            call_idx += 1
            start = time.time()
            try:
                if worker["provider"] == "anthropic":
                    raw, in_tok, out_tok = call_anthropic(anthropic_client, worker["model"], prompt)
                else:
                    raw, in_tok, out_tok = call_openai(openai_client, worker["model"], prompt)
                latency_ms = int((time.time() - start) * 1000)
                seniority, function = parse_prediction(raw)
                cost = compute_cost(in_tok, out_tok, worker["price_in"], worker["price_out"])
                rows.append(
                    {
                        "title": title,
                        "worker_model": worker["name"],
                        "predicted_seniority": seniority,
                        "predicted_function": function,
                        "latency_ms": latency_ms,
                        "cost_usd": cost,
                        "raw_response": raw,
                    }
                )
                print(
                    f"[{call_idx}/{total_calls}] {worker['name']}: '{title}' → "
                    f"{seniority}/{function} ({latency_ms}ms, ${cost:.4f})"
                )
            except Exception as e:
                latency_ms = int((time.time() - start) * 1000)
                print(f"[{call_idx}/{total_calls}] {worker['name']}: '{title}' → ERROR: {e}", file=sys.stderr)
                rows.append(
                    {
                        "title": title,
                        "worker_model": worker["name"],
                        "predicted_seniority": INVALID,
                        "predicted_function": INVALID,
                        "latency_ms": latency_ms,
                        "cost_usd": 0.0,
                        "raw_response": f"ERROR: {e}",
                    }
                )

            if call_idx % CHECKPOINT_EVERY == 0:
                save_results(rows, OUTPUT_PATH)

    save_results(rows, OUTPUT_PATH)

    # Summary
    print()
    print(f"Total calls:  {len(rows)}")
    total_cost = sum(r["cost_usd"] for r in rows)
    print(f"Total cost:   ${total_cost:.4f}")
    print("Average latency per worker:")
    for worker in WORKERS:
        worker_rows = [r for r in rows if r["worker_model"] == worker["name"]]
        if worker_rows:
            avg = sum(r["latency_ms"] for r in worker_rows) / len(worker_rows)
            print(f"  {worker['name']}: {avg:.0f}ms ({len(worker_rows)} calls)")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
