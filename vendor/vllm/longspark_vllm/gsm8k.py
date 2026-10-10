"""GSM8K test fixtures and strict final-answer scoring, using only the stdlib.

The tokenizer and inference backends deliberately live outside this module.
Fixtures contain a user-message body; callers apply the Qwen3 chat template
with enable_thinking=False and add_generation_prompt=True before inference.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


METHODS = ("vanilla", "longspark")


USER_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Your final line must be four hashes, a space, and the numeric answer. "
    "Do not put units, XML tags, or Markdown formatting on the final line.\n"
    "For example:\n#### 42"
)
SCORING_POLICY = (
    "Use only the first nonempty line after the last #### marker. The whole "
    "line must be a signed base-10 integer or decimal, optionally with correctly "
    "grouped thousands commas and one leading dollar symbol. Compare exact "
    "Decimal values; missing/invalid final answers count as incorrect."
)
_NUMBER = re.compile(
    r"(?P<prefix>[+\-]?\s*\$?|\$\s*[+\-]?)\s*"
    r"(?P<number>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)"
)
_REVISION = re.compile(r"[0-9a-f]{40}")


def canonical_prompt(question: str) -> str:
    """Return only the user message, with no reference answer or chat template."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    return f"{USER_INSTRUCTION}\n\n{question}"


def normalize_numeric_answer(text: str) -> str | None:
    """Canonicalize a complete numeric string without float conversion or eval.

    Decimal precision is retained even beyond the current decimal context's
    precision. Units, percentages, expressions, fractions, NaN and infinity
    are rejected rather than silently reduced to a numeric prefix.
    """
    if not isinstance(text, str):
        return None
    match = _NUMBER.fullmatch(text.strip().replace("\N{MINUS SIGN}", "-"))
    if match is None:
        return None
    prefix = match["prefix"]
    sign = "-" if "-" in prefix else ""
    try:
        value = Decimal(sign + match["number"].replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def parse_answer(text: str) -> dict[str, Any]:
    """Report the final marked line and its parse status, retaining the raw line."""
    if not isinstance(text, str):
        raise ValueError("answer text must be a string")
    if "####" not in text:
        return dict(answer=None, final_answer_line=None, status="missing_marker")
    suffix = text.rsplit("####", 1)[1].lstrip()
    line = suffix.splitlines()[0].strip() if suffix else ""
    answer = normalize_numeric_answer(line)
    return dict(answer=answer, final_answer_line=line,
                status="parsed" if answer is not None else "invalid_final_answer")


def extract_answer(text: str) -> str | None:
    """Extract only the final #### answer; never fall back to reasoning numbers."""
    return parse_answer(text)["answer"]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = []
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            raise ValueError(f"{path}: blank JSONL row {index}; original order must be preserved")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be an object")
        rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest()


def load_examples(path: str | Path, *, limit: int = 100, source_url: str,
                  revision: str, split: str = "test") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the first test rows, preserving zero-based official JSONL indices.

    The caller downloads the pinned official file. This helper records its
    complete byte-level SHA256; it does not fetch data or certify authenticity
    merely from a URL string.
    """
    if split != "test":
        raise ValueError("Only the GSM8K test split is supported")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not _REVISION.fullmatch(revision):
        raise ValueError("revision must be a full lowercase 40-character Git commit SHA")
    expected_url = (
        f"https://raw.githubusercontent.com/openai/grade-school-math/{revision}/"
        "grade_school_math/data/test.jsonl"
    )
    if source_url != expected_url:
        raise ValueError("source_url must identify the pinned official GSM8K test JSONL")
    path = Path(path)
    source, sha256 = _read_jsonl(path)
    if len(source) < limit:
        raise ValueError(f"Requested {limit} test examples, but the file contains {len(source)}")
    examples = []
    for index, row in enumerate(source[:limit]):
        question, reference = row.get("question"), row.get("answer")
        prompt = canonical_prompt(question)
        if not isinstance(reference, str) or (gold := extract_answer(reference)) is None:
            raise ValueError(f"Official test row {index} has no valid marked gold answer")
        examples.append(dict(source_id=f"gsm8k-test-{index:04d}", dataset_index=index,
                             question=question, reference_answer=reference,
                             gold_answer=gold, prompt=prompt))
    provenance = dict(
        dataset="openai/grade-school-math", split="test", source_url=source_url,
        revision=revision, source_sha256=sha256, source_bytes=path.stat().st_size,
        source_rows=len(source), selected_count=len(examples),
        selected_indices=list(range(limit)), selection="first rows in original order; no shuffle",
        prompt_instruction=USER_INSTRUCTION,
        chat_template=dict(enable_thinking=False, add_generation_prompt=True),
        scoring_policy=SCORING_POLICY,
    )
    return examples, provenance


def _token_ids(value: Any, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in value):
        raise ValueError(f"{field} must be a sequence of nonnegative integer token IDs")
    return list(value)


def validate_example(example: dict[str, Any]) -> None:
    """Reject reference/prompt drift before scoring a saved fixture."""
    index = example.get("dataset_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("dataset_index must be a nonnegative integer")
    if example.get("source_id") != f"gsm8k-test-{index:04d}":
        raise ValueError("source_id does not match the original test index")
    if example.get("prompt") != canonical_prompt(example.get("question")):
        raise ValueError("prompt differs from the canonical user instruction and question")
    reference = example.get("reference_answer")
    if not isinstance(reference, str) or (gold := extract_answer(reference)) is None:
        raise ValueError("reference_answer must contain a valid marked gold answer")
    if example.get("gold_answer") != gold:
        raise ValueError("gold_answer differs from the canonical reference answer")
    if "input_ids" in example:
        if not _token_ids(example["input_ids"], "input_ids"):
            raise ValueError("input_ids must not be empty")


def score_record(example: dict[str, Any], record: dict[str, Any], *,
                 eos_token_ids: Iterable[int] = (), max_new_tokens: int | None = None) -> dict[str, Any]:
    """Score one generated response, preserving its text and backend fields.

    EOS detection uses actual output IDs. An EOS at the token cap is a natural
    termination, not a truncation. Tokens after EOS are counted explicitly so
    accidental ignore_eos generation remains visible in the result.
    """
    validate_example(example)
    if record.get("source_id", example["source_id"]) != example["source_id"]:
        raise ValueError("record source_id does not match the fixture")
    if not isinstance(record.get("output_text"), str):
        raise ValueError("record must contain output_text")
    output_ids = _token_ids(record.get("output_ids"), "output_ids")
    eos_ids = set(_token_ids(list(eos_token_ids), "eos_token_ids"))
    if max_new_tokens is not None:
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if len(output_ids) > max_new_tokens:
            raise ValueError("output_ids exceeds max_new_tokens")
    if "output_tokens" in record and record["output_tokens"] != len(output_ids):
        raise ValueError("output_tokens differs from the actual output_ids length")
    parsed = parse_answer(record["output_text"])
    first_eos = next((index for index, token in enumerate(output_ids) if token in eos_ids), None)
    eos_seen = first_eos is not None
    at_cap = max_new_tokens is not None and len(output_ids) == max_new_tokens
    truncated = not eos_seen and (at_cap or record.get("finish_reason") in {"length", "max_new_tokens"})
    result = dict(record)
    result.update({key: example[key] for key in (
        "source_id", "dataset_index", "question", "reference_answer", "gold_answer")})
    result.update(
        parsed_answer=parsed["answer"], final_answer_line=parsed["final_answer_line"],
        parse_status=parsed["status"], correct=parsed["answer"] == example["gold_answer"],
        output_tokens=len(output_ids), eos_token_ids=sorted(eos_ids), eos_seen=eos_seen,
        eos_at_end=bool(output_ids) and output_ids[-1] in eos_ids, first_eos_index=first_eos,
        tokens_after_eos=len(output_ids) - first_eos - 1 if eos_seen else 0,
        max_new_tokens=max_new_tokens, token_cap_reached=at_cap, truncated=bool(truncated),
    )
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    unparsed = sum(row["parsed_answer"] is None for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    return dict(
        total=total, correct=correct, incorrect=total - correct, unparsed=unparsed,
        accuracy=correct / total if total else None,
        parsed_accuracy=correct / (total - unparsed) if total > unparsed else None,
        parse_status_counts=dict(sorted(Counter(row["parse_status"] for row in rows).items())),
        eos_count=sum(row["eos_seen"] for row in rows),
        truncated_count=sum(row["truncated"] for row in rows),
        token_cap_count=sum(row["token_cap_reached"] for row in rows),
        outputs_with_tokens_after_eos=sum(row["tokens_after_eos"] > 0 for row in rows),
        output_tokens=output_tokens, mean_output_tokens=output_tokens / total if total else None,
    )


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize already-scored records overall and by method; do not drop failures."""
    rows = list(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        method = row.get("method", "unspecified")
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a nonempty string")
        key = (method, row["source_id"])
        if key in seen:
            raise ValueError(f"Duplicate scored example for {method}: {row['source_id']}")
        seen.add(key)
        if type(row["correct"]) is not bool or row["correct"] != (row["parsed_answer"] == row["gold_answer"]):
            raise ValueError("correct flag disagrees with the exact parsed/gold answer comparison")
        grouped[method].append(row)
    return dict(scoring_policy=SCORING_POLICY, **_aggregate(rows),
                by_method={method: _aggregate(group) for method, group in sorted(grouped.items())})


def load_inputs(path, *, limit, max_new_tokens, max_model_len):
    raw = Path(path).read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Blank fixture rows are forbidden; preserve the official test indices")
    if len(lines) < limit:
        raise ValueError(f"Fixture has {len(lines)} rows, fewer than --limit {limit}")
    examples = []
    for index, line in enumerate(lines[:limit]):
        example = json.loads(line)
        validate_example(example)
        if example["dataset_index"] != index:
            raise ValueError("Fixture must contain the original test prefix in order")
        ids = example.get("input_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("Each fixture row needs pretokenized input_ids; this runner never retokenizes")
        if len(ids) + max_new_tokens > max_model_len:
            raise ValueError(f"{example['source_id']}: prompt plus generation cap exceeds max_model_len")
        examples.append(example)
    return examples, hashlib.sha256(raw).hexdigest()


def resolve_eos(target_path, tokenizer, hf_config):
    """Use generation config, then model config, then tokenizer EOS IDs.

    Preserve each declared source for auditing, while using only the selected
    source's full list so both evaluation backends share stopping semantics.
    """
    generation_path = Path(target_path) / "generation_config.json"
    generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
    raw = {
        "generation_config.json:eos_token_id": generation.get("eos_token_id"),
        "config.json:eos_token_id": hf_config.get("eos_token_id"),
        "tokenizer:eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "tokenizer:eos_token_ids": getattr(tokenizer, "eos_token_ids", None),
    }
    sources = {}
    for name, value in raw.items():
        values = [] if value is None else value if isinstance(value, (list, tuple)) else [value]
        if any(type(token) is not int or token < 0 for token in values):
            raise ValueError(f"Invalid EOS token IDs in {name}: {value!r}")
        sources[name] = sorted(set(values))
    selected = next((name for name, value in raw.items() if value is not None), None)
    ids = sources[selected] if selected is not None else []
    if not ids:
        raise ValueError("No EOS token IDs found; natural-EOS evaluation cannot proceed")
    vocab_size = hf_config.get("vocab_size")
    if vocab_size is not None and any(token >= vocab_size for token in ids):
        raise ValueError("Configured EOS token ID is outside the target vocabulary")
    return dict(effective_token_ids=ids, selected_source=selected, sources=sources,
                generation_config_present=generation_path.is_file())


def compare_outputs(records):
    pairs = {}
    for row in records:
        pair = pairs.setdefault(row["source_id"], {})
        if row["method"] in pair:
            raise ValueError("Duplicate method/source_id when comparing outputs")
        pair[row["method"]] = row
    checks = []
    for source_id, pair in pairs.items():
        if set(pair) != set(METHODS):
            raise ValueError(f"Missing AR or LongSpark output for {source_id}")
        left, right = (pair[method]["output_ids"] for method in METHODS)
        mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        if mismatch is None and len(left) != len(right):
            mismatch = min(len(left), len(right))
        checks.append(dict(source_id=source_id, exact_match=left == right,
                           first_mismatch_index=mismatch, vanilla_tokens=len(left), longspark_tokens=len(right)))
    return dict(passed=all(row["exact_match"] for row in checks), matches=sum(row["exact_match"] for row in checks),
                comparisons=len(checks), checks=checks)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture", help="Create untokenized first-N test examples and provenance")
    fixture.add_argument("--dataset", type=Path, required=True)
    fixture.add_argument("--source-url", required=True)
    fixture.add_argument("--revision", required=True)
    fixture.add_argument("--limit", type=int, default=100)
    fixture.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("summarize", help="Score JSONL records against a saved fixture")
    score.add_argument("--fixture", type=Path, required=True)
    score.add_argument("--records", type=Path, required=True, help="JSONL with source_id, output_text and output_ids")
    score.add_argument("--eos-token-id", action="append", type=int, default=[])
    score.add_argument("--max-new-tokens", type=int, default=1024)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fixture":
        rows, provenance = load_examples(args.dataset, limit=args.limit,
                                         source_url=args.source_url, revision=args.revision)
        sidecar = args.output.with_suffix(args.output.suffix + ".metadata.json")
        if args.output.exists() or sidecar.exists():
            raise FileExistsError("Fixture or metadata already exists; choose a new output path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        _write_json(sidecar, provenance)
        print(json.dumps(dict(output=str(args.output), metadata=str(sidecar), count=len(rows))))
        return
    fixture_rows, fixture_sha = _read_jsonl(args.fixture)
    if not fixture_rows:
        raise ValueError("fixture must not be empty")
    for index, example in enumerate(fixture_rows):
        validate_example(example)
        if example["dataset_index"] != index:
            raise ValueError("fixture must contain the original test prefix in order")
    examples = {row["source_id"]: row for row in fixture_rows}
    records, records_sha = _read_jsonl(args.records)
    if not records:
        raise ValueError("records must not be empty")
    scored = [score_record(examples[row["source_id"]], row, eos_token_ids=args.eos_token_id,
                           max_new_tokens=args.max_new_tokens) for row in records]
    summary = summarize_records(scored)
    if any(group["total"] != len(examples) for group in summary["by_method"].values()):
        raise ValueError("Every method must cover the complete fixture exactly once")
    _write_json(args.output, dict(schema_version=1, dataset="gsm8k", split="test",
                                 fixture_sha256=fixture_sha, records_sha256=records_sha,
                                 summary=summary, records=scored))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
