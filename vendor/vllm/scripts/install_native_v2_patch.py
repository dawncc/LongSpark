#!/usr/bin/env python3
"""Install the reviewed vLLM V2 extension with exact hashes and rollback.

Run with the isolated vLLM environment's Python; importing vLLM (and CUDA)
is unnecessary. The default operation only checks the installed source.
No fuzzy patching, dependency installation, or process restarts are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "native-v2-custom-class.manifest.json"
)
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "vllm":
        raise ValueError(f"Unsafe patch path: {value!r}")
    return Path(*path.parts)


def apply_unified_diff(patch: str, originals: dict[str, bytes]) -> dict[str, bytes]:
    """Apply only exact text hunks to the supplied, already hash-checked files."""
    lines = patch.splitlines(keepends=True)
    results: dict[str, bytes] = {}
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- a/"):
            raise ValueError(f"Unexpected diff header: {lines[index]!r}")
        name = lines[index][6:].rstrip("\n")
        checked_relative_path(name)
        index += 1
        if index >= len(lines) or lines[index] != f"+++ b/{name}\n":
            raise ValueError(f"Mismatched diff path: {name}")
        if name not in originals or name in results:
            raise ValueError(f"Unknown or duplicate diff path: {name}")
        index += 1
        old_lines = originals[name].decode("utf-8").splitlines(keepends=True)
        new_lines: list[str] = []
        cursor = 0
        while index < len(lines) and lines[index].startswith("@@ "):
            match = HUNK.match(lines[index])
            if match is None:
                raise ValueError(f"Invalid hunk: {lines[index]!r}")
            old_start, old_count, new_start, new_count = (
                int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1)
            )
            start = old_start - 1 if old_count else old_start
            if start < cursor or start > len(old_lines):
                raise ValueError(f"Invalid hunk offset in {name}")
            new_lines.extend(old_lines[cursor:start])
            cursor = start
            expected_new_start = new_start - 1 if new_count else new_start
            if len(new_lines) != expected_new_start:
                raise ValueError(f"Invalid new-file hunk offset in {name}")
            index += 1
            read_count = write_count = 0
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- a/")):
                line = lines[index]
                if not line or line[0] not in " +-":
                    raise ValueError(f"Unsupported diff line in {name}: {line!r}")
                marker, content = line[0], line[1:]
                if marker in " -":
                    if cursor >= len(old_lines) or old_lines[cursor] != content:
                        raise ValueError(f"Exact diff context mismatch in {name}:{cursor + 1}")
                    cursor += 1
                    read_count += 1
                if marker in " +":
                    new_lines.append(content)
                    write_count += 1
                index += 1
            if (read_count, write_count) != (old_count, new_count):
                raise ValueError(f"Hunk line counts do not match in {name}")
        new_lines.extend(old_lines[cursor:])
        results[name] = "".join(new_lines).encode("utf-8")
    if results.keys() != originals.keys():
        raise ValueError("Patch paths do not match the manifest")
    return results


def load_manifest(path: Path) -> tuple[dict, bytes]:
    manifest = json.loads(path.read_text())
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", manifest["patch_id"]):
        raise ValueError("Unsafe patch identifier")
    patch_path = path.parent / manifest["patch_file"]
    if patch_path.parent.resolve() != path.parent.resolve():
        raise ValueError("Patch file must be next to the manifest")
    patch = patch_path.read_bytes()
    if sha256(patch) != manifest["patch_sha256"]:
        raise ValueError("Patch SHA-256 does not match its manifest")
    names = [row["path"] for row in manifest["files"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Manifest must have unique nonempty file paths")
    for name in names:
        checked_relative_path(name)
    return manifest, patch


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.longspark-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process(root: Path, version: str, action: str, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    manifest, patch = load_manifest(manifest_path)
    if version not in manifest["supported_versions"]:
        raise ValueError(f"Unsupported vLLM {version!r}; expected {manifest['supported_versions']}")
    if action not in {"check", "apply", "rollback"}:
        raise ValueError(f"Unknown action: {action}")
    root = root.resolve()
    rows = manifest["files"]
    current, modes, states = {}, {}, {}
    for row in rows:
        name = row["path"]
        path = root / checked_relative_path(name)
        # Do not follow a symlink into a different installation or checkout.
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Expected a regular, non-symlink source file: {path}")
        current[name] = path.read_bytes()
        modes[name] = stat.S_IMODE(path.stat().st_mode)
        digest = sha256(current[name])
        if digest == row["before_sha256"]:
            states[name] = "original"
        elif digest == row["after_sha256"]:
            states[name] = "patched"
        else:
            raise ValueError(
                f"Refusing to change unrecognized source {path}: SHA-256={digest}. "
                "Review the installed source against v0.30.0; no force mode is available."
            )
    if len(set(states.values())) != 1:
        raise ValueError(f"Mixed original/patched installation; preserve it for manual review: {states}")
    state = next(iter(states.values()))
    backup = root / ".longspark-native-v2-patch" / manifest["patch_id"]
    report = dict(patch_id=manifest["patch_id"], version=version, root=str(root),
                  action=action, state=state, backup=str(backup), files=states,
                  patch_sha256=manifest["patch_sha256"], changed=False)
    if action == "check" or (action == "apply" and state == "patched"):
        return report
    if action == "rollback" and state == "original":
        return report

    if action == "apply":
        desired = apply_unified_diff(patch.decode("utf-8"), current)
        for row in rows:
            if sha256(desired[row["path"]]) != row["after_sha256"]:
                raise ValueError(f"Patched output hash mismatch: {row['path']}")
        # Complete and verify every backup before changing any installed file.
        for row in rows:
            saved = backup / "original" / row["path"]
            saved.parent.mkdir(parents=True, exist_ok=True)
            if saved.exists():
                if saved.read_bytes() != current[row["path"]]:
                    raise ValueError(f"Existing rollback backup does not match: {saved}")
            else:
                with saved.open("xb") as handle:
                    handle.write(current[row["path"]])
        saved_manifest = backup / "manifest.json"
        expected_manifest = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        if saved_manifest.exists() and saved_manifest.read_bytes() != expected_manifest:
            raise ValueError(f"Existing backup manifest differs: {saved_manifest}")
        if not saved_manifest.exists():
            saved_manifest.write_bytes(expected_manifest)
    else:
        desired = {}
        for row in rows:
            saved = backup / "original" / row["path"]
            data = saved.read_bytes()
            if sha256(data) != row["before_sha256"]:
                raise ValueError(f"Rollback backup hash mismatch: {saved}")
            desired[row["path"]] = data

    written = []
    try:
        for row in rows:
            name = row["path"]
            path = root / name
            # Catch concurrent edits between preflight and mutation.
            if path.read_bytes() != current[name]:
                raise ValueError(f"File changed after preflight: {path}")
            atomic_write(path, desired[name], modes[name])
            written.append(name)
    except BaseException:
        for name in reversed(written):
            if (root / name).read_bytes() == desired[name]:
                atomic_write(root / name, current[name], modes[name])
        raise
    report.update(changed=True, state="patched" if action == "apply" else "original")
    report["files"] = {name: report["state"] for name in current}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_const", dest="action", const="check")
    actions.add_argument("--apply", action="store_const", dest="action", const="apply")
    actions.add_argument("--rollback", action="store_const", dest="action", const="rollback")
    parser.set_defaults(action="check")
    parser.add_argument("--root", type=Path, help="For source review: directory containing vllm/")
    parser.add_argument("--version", help="Required with --root; does not bypass file hashes")
    args = parser.parse_args(argv)
    if (args.root is None) != (args.version is None):
        parser.error("--root and --version must be provided together")
    try:
        if args.root is None:
            distribution = importlib.metadata.distribution("vllm")
            root, version = Path(distribution.locate_file("")), distribution.version
        else:
            root, version = args.root, args.version
        report = process(root, version, args.action)
    except (OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Native V2 patch stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
