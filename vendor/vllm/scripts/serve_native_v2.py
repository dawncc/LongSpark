#!/usr/bin/env python3
"""Launch the native vLLM OpenAI-compatible server with LongSpark V2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "vendor/vllm"), str(ROOT / "vendor")]
from longspark_vllm.serving import make_plan, parser, write_json


def main():
    args = parser().parse_args()
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    for path in (args.target, args.draft):
        if not path.is_dir():
            raise ValueError(f"Model directory does not exist: {path}")
    if args.audit_dir.exists():
        raise FileExistsError(f"Choose a new audit directory: {args.audit_dir}")
    os.environ.update(plan["environment"])
    paths = [str(ROOT / "vendor/vllm"), str(ROOT / "vendor")]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    # Check exact installed upstream source hashes before loading any worker.
    checked = subprocess.run([sys.executable, str(ROOT / "vendor/vllm/scripts/install_native_v2_patch.py"),
                              "--check"], check=True, text=True, capture_output=True)
    patch = json.loads(checked.stdout)
    if patch["state"] != "patched":
        raise ValueError("Apply install_native_v2_patch.py --apply before serving LongSpark V2")
    args.audit_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.audit_dir / "launch.json", plan)
    write_json(args.audit_dir / "patch-check.json", patch)
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm", *plan["argv"]]
    vllm_main()


if __name__ == "__main__":
    main()
