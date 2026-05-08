"""Generate synthetic job title test data using the Anthropic API."""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from anthropic import Anthropic, APIError
from dotenv import load_dotenv

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

OUTPUT_PATH = Path("data/test_data.csv")
MODEL = "claude-sonnet-4-5"
NUM_TITLES = 50

PROMPT = f"""Generate exactly {NUM_TITLES} realistic job titles as test data for an evaluation.

Aim for a diverse mix:
- Easy/clear titles (e.g., "Senior Software Engineer", "Marketing Manager")
- Tricky titles where seniority/function are ambiguous (e.g., "Founding Engineer", "Player-Coach", "Head of Growth")
- Unusual phrasings and creative titles (e.g., "Chief of Staff", "Evangelist", "Ninja", "Wizard")
- Regional variations (e.g., UK "Managing Director", "Solicitor"; "Directeur", "Geschäftsführer")
- A spread across seniority levels and functions

For each title, label it with the most appropriate seniority and function from these fixed lists:
- seniority must be one of: {SENIORITY_LEVELS}
- function must be one of: {FUNCTIONS}

Return ONLY valid JSON — no markdown fences, no preamble, no commentary.
Format: a JSON array of objects, each with keys "title", "seniority", "function".

Example of the exact output shape (do not include this example in your response):
[{{"title": "Senior Software Engineer", "seniority": "Senior IC", "function": "Engineering"}}]
"""


def prompt_overwrite(path: Path) -> bool:
    response = input(f"{path} already exists. Overwrite? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    if OUTPUT_PATH.exists() and not prompt_overwrite(OUTPUT_PATH):
        print("Aborted. Existing file kept.")
        sys.exit(0)

    client = Anthropic(api_key=api_key)

    print(f"Calling {MODEL} for {NUM_TITLES} job titles...")
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": PROMPT}],
        )
    except APIError as e:
        print(f"ERROR: Anthropic API call failed: {e}", file=sys.stderr)
        sys.exit(1)

    raw_text = message.content[0].text.strip()

    # Strip markdown code fences if the model wrapped the JSON despite instructions.
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # Drop opening fence (handles ``` and ```json/```JSON/etc.)
        lines = lines[1:]
        # Drop closing fence if present.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    try:
        entries = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON response: {e}", file=sys.stderr)
        print("--- Raw response ---", file=sys.stderr)
        print(raw_text, file=sys.stderr)
        sys.exit(1)

    if not isinstance(entries, list):
        print(f"ERROR: Expected a JSON array, got {type(entries).__name__}.", file=sys.stderr)
        print(raw_text, file=sys.stderr)
        sys.exit(1)

    valid_entries = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            print(f"WARN: Entry {i} is not an object, skipping: {entry!r}")
            continue
        missing = {"title", "seniority", "function"} - entry.keys()
        if missing:
            print(f"WARN: Entry {i} missing keys {missing}, skipping: {entry!r}")
            continue
        if entry["seniority"] not in SENIORITY_LEVELS:
            print(f"WARN: Entry {i} has invalid seniority {entry['seniority']!r}, skipping.")
            continue
        if entry["function"] not in FUNCTIONS:
            print(f"WARN: Entry {i} has invalid function {entry['function']!r}, skipping.")
            continue
        valid_entries.append(entry)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "title": e["title"],
                "true_seniority": e["seniority"],
                "true_function": e["function"],
            }
            for e in valid_entries
        ]
    )
    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Generated: {len(entries)}")
    print(f"Valid:     {len(valid_entries)}")
    print(f"Output:    {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
