"""No-CUDA checks for native graph coverage and prefill memory accounting."""

import ast
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from longspark_vllm.native_v2.startup import (
    prefill_workspace_budget, require_graph_coverage,
)

Desc = namedtuple("Desc", "cg_mode num_reqs")


class Manager:
    def __init__(self, batches=(), missing=()):
        self.graphs = {Desc("FULL", batch): object() for batch in batches}
        self.missing = set(missing)
        self.compilation_config = NS(cudagraph_capture_sizes=[8 * batch for batch in batches])

    def dispatch(self, batch, tokens, *, uniform_token_count, num_active_loras):
        assert tokens == batch * 7 and uniform_token_count == 7 and num_active_loras == 0
        candidates = [desc for desc in self.graphs if desc.num_reqs >= batch]
        if not candidates or batch in self.missing:
            return Desc("NONE", batch)
        return min(candidates, key=lambda item: item.num_reqs)


def test_padded_graph_covers_smaller_batches_without_requiring_exact_shapes():
    assert require_graph_coverage(Manager([4]), 4, full_mode="FULL") == [1, 2, 3, 4]


def test_any_missing_batch_fails_at_startup_instead_of_at_first_request():
    with pytest.raises(RuntimeError, match=r"missing_batches=\[3\]"):
        require_graph_coverage(Manager([4], missing=[3]), 4, full_mode="FULL")


def test_empty_graphs_report_invalid_capture_sizes():
    manager = Manager()
    manager.compilation_config.cudagraph_capture_sizes = [8]
    with pytest.raises(RuntimeError, match=r"captures=0.*capture_sizes=\[8\].*draft_query_len=7"):
        require_graph_coverage(manager, 1, full_mode="FULL")


def budget(**overrides):
    options = dict(max_num_reqs=2, max_num_tokens=512, max_model_len=4096,
                   num_layers=5, kv_heads=8, query_heads=32, head_dim=128,
                   hidden_size=2560)
    options.update(overrides)
    return prefill_workspace_budget(**options)


def test_prefill_budget_covers_unbalanced_requests_and_explains_all_terms():
    result = budget()
    # One 511-token prefill mixed with one 1-token decode gathers 2*511 rows.
    assert result["max_batch"] == 2 and result["padded_width"] == 511
    assert result["update_bytes"] == sum(result["update_terms"].values())
    assert result["bytes"] == (max(result["gather_bytes"], result["update_bytes"])
                                + result["allocator_library_margin_bytes"])
    assert result["bytes"] > 2 * 2 * 5 * 511 * 8 * 128 * 2  # two packed K/V pairs


def test_budget_respects_token_and_context_caps_and_grows_with_prompt_limit():
    small = budget(max_num_tokens=8)
    large = budget(max_num_tokens=512)
    assert large["bytes"] > small["bytes"]
    assert budget(max_num_reqs=4)["bytes"] > large["bytes"]
    capped = budget(max_num_reqs=32, max_num_tokens=7, max_model_len=2)
    assert capped["max_batch"] <= 7 and capped["padded_width"] <= 2


@pytest.mark.parametrize("options", [dict(max_num_tokens=0), dict(max_num_reqs=True),
                                      dict(query_heads=31)])
def test_invalid_budget_geometry_is_rejected(options):
    with pytest.raises(ValueError):
        budget(**options)


def dummy_propose_function(torch):
    # Exercise the actual dummy branch without importing vLLM or torch. The
    # method is parsed from the implementation; capture decorators are removed.
    path = Path(__file__).resolve().parents[1] / "longspark_vllm/native_v2/speculator.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "propose")
    method.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace["propose"]


@pytest.mark.parametrize("is_profile", [False, True])
def test_dummy_profile_reservation_overlaps_draft_and_is_released(is_profile):
    live, events = [], []

    class Scratch:
        def __init__(self):
            live.append(True)

        def __del__(self):
            live.pop()
            events.append("released")

    def empty(count, *, dtype, device):
        assert count == 12345 and dtype == "uint8" and device == "test-device"
        events.append("reserved")
        return Scratch()

    class Buffer:
        def copy_(self, _): pass
        def fill_(self, _): pass
        def zero_(self): pass

    def generate(batch):
        assert batch == 2
        assert bool(live) == is_profile
        events.append("draft")

    owner = NS(_idx=Buffer(), _padding_slots=object(), _starts=Buffer(),
               _counts=Buffer(), _anchors=Buffer(), _dummy_calls=0,
               _prefill_workspace_budget={"bytes": 12345}, _profile_budget_checks=0,
               _generate_draft=generate, draft_tokens=[1, 2, 3], device="test-device")
    propose = dummy_propose_function(NS(empty=empty, uint8="uint8"))
    result = propose(owner, NS(num_reqs=2), None, None, None, None, None, None,
                     None, None, None, None, dummy_run=True, is_profile=is_profile)
    assert result == [1, 2] and not live
    assert owner._profile_budget_checks == int(is_profile)
    assert events == (["reserved", "draft", "released"] if is_profile else ["draft"])
