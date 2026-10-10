"""GSM8K fixture integrity and strict scoring contracts, with no model runtime."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longspark_vllm import gsm8k


REVISION = "0123456789abcdef0123456789abcdef01234567"
SOURCE_URL = (f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/"
              "grade_school_math/data/test.jsonl")


class AnswerParsingTests(unittest.TestCase):
    def test_exact_decimal_normalization_and_currency(self):
        accepted = {
            "1,234,567.8900": "1234567.89", " -$1,200.50 ": "-1200.5",
            "$-1,200.50": "-1200.5", "+ $ 00012.000": "12", "$ + 12.00": "12",
            "−.50": "-0.5", ".5000": "0.5", "42.": "42", "-0.000": "0",
            "1000": "1000", "001000": "1000",
            "123456789012345678901234567890.123456789012345678900":
                "123456789012345678901234567890.1234567890123456789",
        }
        for text, expected in accepted.items():
            with self.subTest(text=text):
                self.assertEqual(gsm8k.normalize_numeric_answer(text), expected)

    def test_numeric_prefixes_expressions_units_and_malformed_grouping_are_rejected(self):
        rejected = ("12,34", "1,23,456", "1,2345", "1,,000", "1,000,", "1,000.2.3",
                    "42 dollars", "42%", "1/2", "1 + 2", "1e3", "NaN", "Infinity",
                    "--2", "$$3", "-$-3", "(12)", "", "**42**", "42 or 43",
                    "__import__('os').system('false')")
        for text in rejected:
            with self.subTest(text=text):
                self.assertIsNone(gsm8k.normalize_numeric_answer(text))

    def test_last_marker_wins_even_when_it_is_invalid(self):
        self.assertEqual(gsm8k.extract_answer("Earlier #### 10\nFinal #### $1,200.00"), "1200")
        self.assertIsNone(gsm8k.extract_answer("Earlier #### 10\nFinal #### no answer"))
        self.assertIsNone(gsm8k.extract_answer("Earlier #### 10\nFinal ####"))

    def test_no_fallback_to_reasoning_numbers(self):
        parsed = gsm8k.parse_answer("We compute 6 * 7 = 42. The answer is 42.")
        self.assertEqual(parsed["status"], "missing_marker")
        self.assertIsNone(parsed["answer"])
        self.assertIsNone(gsm8k.extract_answer("#### Therefore 42"))
        self.assertIsNone(gsm8k.extract_answer("#### 42 because 6 times 7 is 42"))
        self.assertIsNone(gsm8k.extract_answer("#### Reasoning\n42"))

    def test_first_nonempty_marked_line_and_original_text(self):
        text = "Work: 6 * 7.\n#### \n\n -$42.000\nAdditional note: 999"
        self.assertEqual(gsm8k.parse_answer(text), dict(
            answer="-42", final_answer_line="-$42.000", status="parsed"))

    def test_xml_placeholder_responses_remain_unparsed(self):
        for text in ("#### <answer>18</answer>", "#### <answer>\n3", "#### <answer>\n70000\n</answer>"):
            with self.subTest(text=text):
                self.assertEqual(gsm8k.parse_answer(text)["status"], "invalid_final_answer")
                self.assertIsNone(gsm8k.extract_answer(text))


class FixtureAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gsm8k scoring ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "test.jsonl"
        self.source = [
            dict(question="There are 2 apples and 3 pears. How many fruits?",
                 answer="Private reference reasoning: 2 + 3 = 5.\n#### 5"),
            dict(question="What is half of one dollar?",
                 answer="Private reference reasoning: 1 / 2 = 0.50.\n#### $0.50"),
            dict(question="Balance starts at -10 and loses 2. What is it?",
                 answer="Private reference reasoning: -10 - 2 = -12.\n#### -12"),
        ]
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in self.source))
        self.examples, self.provenance = self.load()

    def load(self, **kwargs):
        options = dict(limit=3, source_url=SOURCE_URL, revision=REVISION)
        options.update(kwargs)
        return gsm8k.load_examples(self.dataset, **options)

    def record(self, *, index=0, text="#### 5", ids=(10, 99), **extra):
        return gsm8k.score_record(self.examples[index], dict(
            method="vanilla", source_id=self.examples[index]["source_id"],
            output_text=text, output_ids=list(ids), **extra), eos_token_ids=[99], max_new_tokens=4)

    def test_prefix_order_source_hash_and_reference_preservation(self):
        selected, provenance = self.load(limit=2)
        self.assertEqual([row["source_id"] for row in selected], ["gsm8k-test-0000", "gsm8k-test-0001"])
        self.assertEqual([row["dataset_index"] for row in selected], [0, 1])
        self.assertEqual(selected[1]["reference_answer"], self.source[1]["answer"])
        self.assertEqual(selected[1]["gold_answer"], "0.5")
        self.assertEqual(provenance["source_sha256"], hashlib.sha256(self.dataset.read_bytes()).hexdigest())
        self.assertEqual(provenance["source_rows"], 3)
        self.assertEqual(provenance["selected_indices"], [0, 1])
        self.assertEqual(provenance["source_url"], SOURCE_URL)
        self.assertEqual(provenance["revision"], REVISION)
        self.assertEqual(provenance["split"], "test")
        self.assertEqual(self.load(limit=2), (selected, provenance))

    def test_prompt_is_exact_user_instruction_without_reference_or_template(self):
        self.assertNotIn("<answer>", gsm8k.USER_INSTRUCTION)
        self.assertEqual(gsm8k.USER_INSTRUCTION.splitlines()[-1], "#### 42")
        self.assertIn("Do not put units, XML tags, or Markdown formatting", gsm8k.USER_INSTRUCTION)
        for example in self.examples:
            self.assertEqual(example["prompt"], gsm8k.USER_INSTRUCTION + "\n\n" + example["question"])
            self.assertNotIn("Private reference reasoning", example["prompt"])
            self.assertNotIn("<|im_start|>", example["prompt"])
            self.assertNotIn("input_ids", example)
        self.assertEqual(self.provenance["chat_template"], dict(
            enable_thinking=False, add_generation_prompt=True))

    def test_only_pinned_official_test_source_is_allowed(self):
        for options in (dict(split="train"), dict(source_url=SOURCE_URL.replace("test.jsonl", "train.jsonl")),
                        dict(revision="main"), dict(source_url=SOURCE_URL.replace(REVISION, "main")),
                        dict(source_url=SOURCE_URL.replace("openai/", "someone/")), dict(limit=4), dict(limit=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load(**options)

    def test_blank_rows_or_unmarked_gold_fail_instead_of_reindexing(self):
        content = self.dataset.read_text()
        self.dataset.write_text("\n" + content)
        with self.assertRaisesRegex(ValueError, "blank JSONL"):
            self.load()
        row = dict(self.source[0], answer="The answer is 5")
        self.dataset.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "gold answer"):
            self.load(limit=1)

    def test_scoring_preserves_generated_text_and_counts_unparsed_as_wrong(self):
        raw = "The answer follows.\n#### $5.000\n"
        correct = self.record(text=raw, seconds=1.5)
        missing = self.record(index=1, text="We calculate 0.5", ids=[1, 2, 3, 4])
        wrong = self.record(index=2, text="#### 12")
        self.assertEqual(correct["output_text"], raw)
        self.assertEqual(correct["reference_answer"], self.source[0]["answer"])
        self.assertEqual(correct["seconds"], 1.5)
        self.assertTrue(correct["correct"])
        self.assertEqual(missing["parse_status"], "missing_marker")
        self.assertFalse(missing["correct"])
        summary = gsm8k.summarize_records([correct, missing, wrong])
        self.assertEqual((summary["total"], summary["correct"], summary["unparsed"]), (3, 1, 1))
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["parsed_accuracy"], 1 / 2)
        self.assertEqual(summary["eos_count"], 2)
        self.assertEqual(summary["truncated_count"], 1)

    def test_eos_at_cap_is_not_truncation_and_post_eos_tokens_are_visible(self):
        natural = self.record(ids=[1, 2, 3, 99])
        self.assertTrue(natural["token_cap_reached"])
        self.assertTrue(natural["eos_at_end"])
        self.assertFalse(natural["truncated"])
        truncated = self.record(ids=[1, 2, 3, 4])
        self.assertTrue(truncated["truncated"])
        self.assertFalse(truncated["eos_seen"])
        accidental = self.record(ids=[1, 99, 2, 3])
        self.assertEqual(accidental["tokens_after_eos"], 2)
        self.assertFalse(accidental["eos_at_end"])
        self.assertEqual(gsm8k.summarize_records([accidental])["outputs_with_tokens_after_eos"], 1)
        self.assertTrue(self.record(ids=[1, 2], finish_reason="length")["truncated"])

    def test_scoring_rejects_fixture_and_token_count_drift(self):
        for change in (dict(prompt="includes gold"), dict(gold_answer="6"), dict(source_id="gsm8k-train-0000")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                gsm8k.score_record(dict(self.examples[0], **change), dict(output_text="#### 5", output_ids=[1]))
        with self.assertRaises(ValueError):
            self.record(output_tokens=100)
        with self.assertRaises(ValueError):
            self.record(ids=[1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            self.record(ids=[True])

    def test_separate_methods_duplicate_detection_and_empty_denominator(self):
        ar = self.record()
        spec = dict(ar, method="longspark")
        summary = gsm8k.summarize_records([ar, spec])
        self.assertEqual(summary["by_method"]["vanilla"]["total"], 1)
        self.assertEqual(summary["by_method"]["longspark"]["correct"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            gsm8k.summarize_records([ar, ar])
        with self.assertRaisesRegex(ValueError, "correct flag"):
            gsm8k.summarize_records([dict(ar, correct=False)])
        self.assertIsNone(gsm8k.summarize_records([])["accuracy"])

    def test_cli_creates_fixture_and_complete_scores_without_overwriting(self):
        fixture = self.root / "fixtures" / "gsm8k.jsonl"
        options = ["fixture", "--dataset", str(self.dataset), "--source-url", SOURCE_URL,
                   "--revision", REVISION, "--limit", "3", "--output", str(fixture)]
        with redirect_stdout(io.StringIO()):
            gsm8k.main(options)
        self.assertEqual(json.loads(fixture.with_suffix(".jsonl.metadata.json").read_text()), self.provenance)
        saved = fixture.read_bytes()
        with self.assertRaises(FileExistsError):
            gsm8k.main(options)
        self.assertEqual(fixture.read_bytes(), saved)
        records = self.root / "responses.jsonl"
        records.write_text("".join(json.dumps(dict(source_id=example["source_id"], method=method,
                           output_text="#### " + example["gold_answer"], output_ids=[1, 99])) + "\n"
                           for method in ("vanilla", "longspark") for example in self.examples))
        report = self.root / "summary.json"
        args = ["summarize", "--fixture", str(fixture), "--records", str(records),
                "--eos-token-id", "99", "--max-new-tokens", "4", "--output", str(report)]
        with redirect_stdout(io.StringIO()):
            gsm8k.main(args)
        result = json.loads(report.read_text())
        self.assertEqual(result["summary"]["correct"], 6)
        self.assertEqual(result["summary"]["eos_count"], 6)
        self.assertEqual(len(result["records"]), 6)
        with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
            gsm8k.main(args)
        records.write_text(records.read_text().splitlines()[0] + "\n")
        args[-1] = str(self.root / "incomplete.json")
        with self.assertRaisesRegex(ValueError, "complete fixture"), redirect_stdout(io.StringIO()):
            gsm8k.main(args)
        self.assertFalse(Path(args[-1]).exists())

    def test_module_cli_imports_no_model_runtime(self):
        script = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'transformers', 'specforge', 'sglang'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
sys.argv = ['gsm8k', '--help']
runpy.run_module('longspark_vllm.gsm8k', run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-S", "-c", script,
                                 str(Path(gsm8k.__file__).resolve().parents[1])],
                                capture_output=True, text=True, check=True)
        self.assertIn("fixture", result.stdout)
        self.assertIn("summarize", result.stdout)


class RuntimeFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gsm8k runtime fixture ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = self.root / "input.jsonl"
        self.examples = [dict(source_id=f"gsm8k-test-{index:04d}", dataset_index=index,
                              question=f"What is {index} plus one?",
                              reference_answer=f"#### {index + 1}", gold_answer=str(index + 1),
                              prompt=gsm8k.canonical_prompt(f"What is {index} plus one?"),
                              input_ids=[index + 1, 42]) for index in range(2)]
        self.fixture.write_text("".join(json.dumps(row) + "\n" for row in self.examples))

    def load(self, **options):
        defaults = dict(limit=2, max_new_tokens=4, max_model_len=6)
        return gsm8k.load_inputs(self.fixture, **(defaults | options))

    def test_tokenized_prefix_preserves_ids_hash_and_context_bounds(self):
        loaded, digest = self.load()
        self.assertEqual(loaded, self.examples)
        self.assertEqual(digest, hashlib.sha256(self.fixture.read_bytes()).hexdigest())
        for options in (dict(limit=3), dict(max_model_len=5)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load(**options)

    def test_tokenized_fixture_rejects_reordering_missing_ids_and_blank_rows(self):
        invalid = (
            list(reversed(self.examples)),
            [dict(row, input_ids=[]) for row in self.examples],
            [{key: value for key, value in row.items() if key != "input_ids"}
             for row in self.examples],
        )
        for rows in invalid:
            self.fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load()
        self.fixture.write_text("\n" + self.fixture.read_text())
        with self.assertRaisesRegex(ValueError, "Blank fixture"):
            self.load()

    def test_eos_precedence_retains_all_sources_without_union(self):
        generation = self.root / "generation_config.json"
        generation.write_text(json.dumps(dict(eos_token_id=[98, 99])))
        tokenizer = SimpleNamespace(eos_token_id=97, eos_token_ids=[96, 97])
        result = gsm8k.resolve_eos(self.root, tokenizer, dict(eos_token_id=99, vocab_size=200))
        self.assertEqual(result["effective_token_ids"], [98, 99])
        self.assertEqual(result["selected_source"], "generation_config.json:eos_token_id")
        self.assertEqual(result["sources"]["tokenizer:eos_token_ids"], [96, 97])
        generation.unlink()
        self.assertEqual(gsm8k.resolve_eos(self.root, tokenizer, dict(eos_token_id=99))
                         ["effective_token_ids"], [99])
        self.assertEqual(gsm8k.resolve_eos(self.root, tokenizer, {})["effective_token_ids"], [97])
        self.assertEqual(gsm8k.resolve_eos(self.root, SimpleNamespace(eos_token_ids=[96, 97]), {})
                         ["effective_token_ids"], [96, 97])

    def test_empty_invalid_and_out_of_vocabulary_eos_fail(self):
        for value in ([], [-1], [True], "99", [201]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gsm8k.resolve_eos(self.root, SimpleNamespace(eos_token_id=value), dict(vocab_size=200))
        with self.assertRaisesRegex(ValueError, "No EOS"):
            gsm8k.resolve_eos(self.root, SimpleNamespace(), {})

    def test_exact_output_comparison_catches_length_and_token_mismatches(self):
        ar = dict(source_id="one", method="vanilla", output_ids=[1, 2])
        for ids, mismatch in (([1, 2, 3], 2), ([1, 3], 1), ([1], 1)):
            spec = dict(ar, method="longspark", output_ids=ids)
            result = gsm8k.compare_outputs([ar, spec])
            self.assertFalse(result["passed"])
            self.assertEqual(result["checks"][0]["first_mismatch_index"], mismatch)
        self.assertTrue(gsm8k.compare_outputs([ar, dict(ar, method="longspark")])["passed"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            gsm8k.compare_outputs([ar, ar])
        with self.assertRaisesRegex(ValueError, "Missing"):
            gsm8k.compare_outputs([ar])


if __name__ == "__main__":
    unittest.main()
