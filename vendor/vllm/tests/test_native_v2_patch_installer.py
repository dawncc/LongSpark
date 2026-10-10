"""CPU-only protection checks for the exact-source, reversible vLLM patch."""

import difflib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import install_native_v2_patch as installer


@pytest.fixture
def package(tmp_path):
    root, bundle = tmp_path / "site-packages", tmp_path / "patch"
    bundle.mkdir()
    files, patch = [], []
    before = {"vllm/a.py": b"# original\nVALUE = 1\n",
              "vllm/b.py": b"# original\nVALUE = 2\n"}
    after = {name: data.replace(b"original", b"patched") for name, data in before.items()}
    for name, data in before.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append(dict(path=name, before_sha256=installer.sha256(data),
                          after_sha256=installer.sha256(after[name])))
        patch.extend(difflib.unified_diff(data.decode().splitlines(keepends=True),
                                         after[name].decode().splitlines(keepends=True),
                                         fromfile=f"a/{name}", tofile=f"b/{name}"))
    patch = "".join(patch).encode()
    (bundle / "test.patch").write_bytes(patch)
    manifest = bundle / "manifest.json"
    manifest.write_text(json.dumps(dict(patch_id="test-v1", supported_versions=["0.30.0+cu129"],
                                        patch_file="test.patch", patch_sha256=installer.sha256(patch),
                                        files=files)))
    return root, manifest, before, after


def run(package, action):
    root, manifest, _, _ = package
    return installer.process(root, "0.30.0+cu129", action, manifest)


def test_apply_idempotency_and_verified_rollback(package):
    root, _, before, after = package
    assert run(package, "check")["state"] == "original"
    assert not (root / ".longspark-native-v2-patch").exists()
    assert run(package, "apply")["changed"]
    assert {name: (root / name).read_bytes() for name in after} == after
    assert not run(package, "apply")["changed"]
    assert run(package, "rollback")["changed"]
    assert {name: (root / name).read_bytes() for name in before} == before
    assert not run(package, "rollback")["changed"]


def test_source_mismatch_aborts_before_any_write(package):
    root, _, before, _ = package
    (root / "vllm/b.py").write_bytes(b"# user change\n")
    with pytest.raises(ValueError, match="unrecognized source"):
        run(package, "apply")
    assert (root / "vllm/a.py").read_bytes() == before["vllm/a.py"]
    assert not (root / ".longspark-native-v2-patch").exists()


def test_rollback_preserves_edits_made_after_install(package):
    root, _, _, after = package
    run(package, "apply")
    (root / "vllm/b.py").write_bytes(b"# later user edit\n")
    with pytest.raises(ValueError, match="unrecognized source"):
        run(package, "rollback")
    assert (root / "vllm/a.py").read_bytes() == after["vllm/a.py"]
    assert (root / "vllm/b.py").read_bytes() == b"# later user edit\n"


def test_tampered_backup_prevents_all_restores(package):
    root, _, _, after = package
    report = run(package, "apply")
    saved = Path(report["backup"]) / "original/vllm/b.py"
    saved.write_bytes(b"# broken backup\n")
    with pytest.raises(ValueError, match="backup hash mismatch"):
        run(package, "rollback")
    assert {name: (root / name).read_bytes() for name in after} == after


def test_interrupted_apply_restores_already_written_files(package, monkeypatch):
    root, _, before, _ = package
    write = installer.atomic_write
    calls = 0

    def interrupted(path, data, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated filesystem failure")
        return write(path, data, mode)

    monkeypatch.setattr(installer, "atomic_write", interrupted)
    with pytest.raises(OSError, match="simulated filesystem failure"):
        run(package, "apply")
    assert {name: (root / name).read_bytes() for name in before} == before


def test_changed_patch_or_version_is_rejected(package):
    root, manifest, _, _ = package
    with pytest.raises(ValueError, match="Unsupported vLLM"):
        installer.process(root, "0.31.0", "apply", manifest)
    patch = manifest.parent / "test.patch"
    patch.write_bytes(patch.read_bytes() + b"# extra\n")
    with pytest.raises(ValueError, match="Patch SHA-256"):
        run(package, "apply")


def test_symlink_to_another_installation_is_rejected(package):
    root, _, before, _ = package
    external = root.parent / "different.py"
    external.write_bytes(before["vllm/a.py"])
    path = root / "vllm/a.py"
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match="non-symlink"):
        run(package, "apply")
    assert external.read_bytes() == before["vllm/a.py"]


def test_exact_context_is_required_even_for_parseable_diff():
    patch = "--- a/vllm/a.py\n+++ b/vllm/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    with pytest.raises(ValueError, match="context mismatch"):
        installer.apply_unified_diff(patch, {"vllm/a.py": b"different\n"})


def test_reviewed_bundle_hashes_are_consistent():
    manifest, _ = installer.load_manifest(installer.DEFAULT_MANIFEST)
    assert manifest["upstream_tag"] == "v0.30.0"
    assert len(manifest["files"]) == 5
