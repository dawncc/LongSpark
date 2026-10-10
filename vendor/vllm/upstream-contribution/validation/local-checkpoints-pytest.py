import sys
from pathlib import Path

import pytest


class LocalCheckpoints:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            module = item.module
            if not module.__name__.endswith("test_longspark"):
                continue
            module.MODEL = "/data/lpf/models/Qwen3-4B"
            module.DRAFT = "/data/chenchen/codex/LongSpark/models/4B/longspark"
            module.SPECULATIVE_CONFIG["model"] = module.DRAFT
            module.ENV["VLLM_USE_RUST_FRONTEND"] = "0"


if __name__ == "__main__":
    destination = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/api-e2e")
    destination.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        pytest.main(
            [
                "-o", "addopts=", "-vv", "-s",
                "tests/v1/e2e/spec_decode/longspark/test_longspark.py",
                "-k", sys.argv[1] if len(sys.argv) > 1 else "native_api",
                f"--junitxml={destination / 'junit.xml'}",
            ],
            plugins=[LocalCheckpoints()],
        )
    )
