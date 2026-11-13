#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the files a run of wheels.yml would upload, as one bundle.

Usage: python tools/check-dist.py --version VERSION [--release]
                                  [--write-sums | --verify-sums]
                                  [--max-mib N] [--skip-twine] BUNDLE

BUNDLE/dist/ must hold exactly the files of one version: for each of
manylinux_2_28 x86_64, manylinux_2_28 aarch64 and macOS 11 arm64, a
cudaq-pulse cp312-abi3 wheel, a cudaq-pulse cp314-cp314t wheel and a
cudaq-pulse-tools py3-none wheel, each with that one platform tag (such as
cudaq_pulse-0.1.0-cp312-abi3-manylinux_2_28_x86_64.whl), and one
cudaq-pulse sdist. Every file must
be named and carry metadata for exactly VERSION, a public PEP 440 version
(an index rejects a local version, X.Y.Z+...); with --release it must be a
release version (X.Y.Z, X.Y.Z.postN or X.Y.Z{a,b,rc}N, as release-meta.py
accepts for a tag), and otherwise a dev version (.devN). Each file must be at
most --max-mib MiB (default 90; PyPI's limit is 100), pass
tools/check-wheel-contents.py (license files, notices, layout and the
cudaq-pulse-tools pin) and `twine check --strict`.

--write-sums then writes BUNDLE/SHA256SUMS (`sha256sum --check` reads it in
BUNDLE) and BUNDLE/MANIFEST.json, one entry per file. --verify-sums instead
requires that SHA256SUMS lists exactly the files of dist/ with their digests.
Exits 1 listing every problem.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
_PUBLIC_VERSION = re.compile(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?")
_PLATFORMS = {
    "linux-x86_64": "manylinux_2_28_x86_64",
    "linux-aarch64": "manylinux_2_28_aarch64",
    "macos-arm64": "macosx_11_0_arm64",
}
# (distribution, python tag, abi tag) of the wheels each platform needs.
_WHEELS = [
    ("cudaq-pulse", "cp312", "abi3"),
    ("cudaq-pulse", "cp314", "cp314t"),
    ("cudaq-pulse-tools", "py3", "none"),
]
_SDIST = "cudaq-pulse"
SUMS = "SHA256SUMS"
MANIFEST = "MANIFEST.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), _TOOLS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(path: Path) -> dict[str, str]:
    """Name and Version from a wheel's METADATA or an sdist's PKG-INFO."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as wheel:
            item = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
            text = wheel.read(item).decode()
    else:
        with tarfile.open(path) as sdist:
            member = next(m for m in sdist.getmembers() if re.fullmatch(r"[^/]+/PKG-INFO", m.name))
            file = sdist.extractfile(member)
            assert file is not None
            text = file.read().decode()
    message = Parser().parsestr(text)
    return {"name": message.get("Name") or "", "version": message.get("Version") or ""}


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def describe(path: Path) -> dict[str, str] | None:
    """The distribution, version and tags in a file name, or ``None``."""
    if path.name.endswith(".tar.gz"):
        stem = path.name[: -len(".tar.gz")]
        name, _, version = stem.rpartition("-")
        return {"name": name, "version": version, "kind": "sdist"} if name else None
    if path.suffix == ".whl":
        parts = path.name[: -len(".whl")].split("-")
        if len(parts) != 5:
            return None
        name, version, python, abi, platform = parts
        return {
            "name": name,
            "version": version,
            "kind": "wheel",
            "python": python,
            "abi": abi,
            "platform": platform,
        }
    return None


def check_version(version: str, release: bool) -> list[str]:
    if "+" in version:
        return [f"version {version} has a local segment, which an index rejects"]
    if not _PUBLIC_VERSION.fullmatch(version):
        return [f"version {version} is not a normalized public PEP 440 version"]
    if release:
        try:
            _load("release-meta").release_version(f"v{version}")
        except ValueError as error:
            return [f"version {version} is not a release version: {error}"]
    elif ".dev" not in version:
        return [f"version {version} is not a dev version (.devN), but --release was not given"]
    return []


def check_files(files: list[Path], version: str, max_mib: float) -> list[str]:
    """The expected set of files, each named and built as ``version``."""
    problems = []
    found: dict[tuple[str, ...], list[str]] = {}
    for path in files:
        parts = describe(path)
        if parts is None:
            problems.append(f"{path.name}: not a wheel or sdist")
            continue
        if parts["version"] != version:
            local = " (a local version)" if "+" in parts["version"] else ""
            problems.append(
                f"{path.name}: file name version {parts['version']}{local}, expected {version}"
            )
        meta = _metadata(path)
        if meta["version"] != version:
            problems.append(f"{path.name}: metadata Version {meta['version']}, expected {version}")
        if _normalized(meta["name"]) != parts["name"]:
            problems.append(
                f"{path.name}: metadata Name {meta['name']} does not match the file name"
            )
        mib = path.stat().st_size / (1024 * 1024)
        if mib > max_mib:
            problems.append(f"{path.name}: {mib:.1f} MiB, over the {max_mib:g} MiB limit")
        if parts["kind"] == "sdist":
            key: tuple[str, ...] = ("sdist", parts["name"])
        else:
            if "." in parts["platform"]:
                problems.append(
                    f"{path.name}: platform tags {parts['platform']}, expected exactly one"
                )
            platform = next(
                (p for p, tag in _PLATFORMS.items() if tag == parts["platform"]),
                parts["platform"],
            )
            key = (parts["name"], parts["python"], parts["abi"], platform)
        found.setdefault(key, []).append(path.name)
    expected = {("sdist", _normalized(_SDIST))} | {
        (_normalized(dist), python, abi, platform)
        for platform in _PLATFORMS
        for dist, python, abi in _WHEELS
    }
    for key in sorted(expected - found.keys()):
        problems.append(f"missing {_label(key)}")
    for key in sorted(found.keys() - expected):
        problems += [f"{name}: not one of the expected files" for name in found[key]]
    for key in sorted(found.keys() & expected):
        if len(found[key]) > 1:
            problems.append(f"more than one {_label(key)}: {found[key]}")
    return problems


def _label(key: tuple[str, ...]) -> str:
    if key[0] == "sdist":
        return f"{key[1]} sdist"
    name, python, abi, platform = key
    return f"{name} {python}-{abi} wheel for {platform}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_sums(files: list[Path]) -> str:
    return "".join(f"{_sha256(path)}  dist/{path.name}\n" for path in sorted(files))


def verify_sums(bundle: Path, files: list[Path]) -> list[str]:
    sums = bundle / SUMS
    if not sums.is_file():
        return [f"missing {SUMS}"]
    listed = {}
    for line in sums.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  dist/([^/]+)", line)
        if match is None:
            return [f"{SUMS}: malformed line {line!r}"]
        listed[match.group(2)] = match.group(1)
    names = {path.name: path for path in files}
    problems = [
        f"{SUMS} lists {name}, which is not in dist/"
        for name in sorted(listed.keys() - names.keys())
    ]
    problems += [f"{SUMS} does not list {name}" for name in sorted(names.keys() - listed.keys())]
    problems += [
        f"{name}: SHA-256 differs from {SUMS}"
        for name in sorted(listed.keys() & names.keys())
        if _sha256(names[name]) != listed[name]
    ]
    return problems


def manifest(files: list[Path], version: str) -> str:
    entries = []
    for path in sorted(files):
        parts = describe(path) or {}
        meta = _metadata(path)
        entry = {
            "file": f"dist/{path.name}",
            "distribution": meta["name"],
            "version": version,
            "kind": parts.get("kind", ""),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if parts.get("kind") == "wheel":
            entry["tags"] = f"{parts['python']}-{parts['abi']}-{parts['platform']}"
        entries.append(entry)
    return json.dumps({"version": version, "files": entries}, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--release", action="store_true")
    sums = parser.add_mutually_exclusive_group()
    sums.add_argument("--write-sums", action="store_true")
    sums.add_argument("--verify-sums", action="store_true")
    parser.add_argument("--max-mib", type=float, default=90)
    parser.add_argument("--skip-twine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)

    dist = args.bundle / "dist"
    files = sorted(p for p in dist.iterdir() if p.is_file()) if dist.is_dir() else []
    problems = check_version(args.version, args.release)
    if not files:
        problems.append(f"no files in {dist}")
    else:
        problems += check_files(files, args.version, args.max_mib)
        problems += _load("check-wheel-contents").check(files)
        if not args.skip_twine:
            twine = subprocess.run(
                [sys.executable, "-m", "twine", "check", "--strict", *map(str, files)],
                capture_output=True,
                text=True,
            )
            if twine.returncode != 0:
                problems.append(f"twine check --strict failed:\n{twine.stdout}{twine.stderr}")
    if args.verify_sums:
        problems += verify_sums(args.bundle, files)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    if args.write_sums:
        (args.bundle / SUMS).write_text(render_sums(files))
        (args.bundle / MANIFEST).write_text(manifest(files, args.version))
    print("| file | MiB |\n|---|---:|")
    for path in files:
        print(f"| `{path.name}` | {path.stat().st_size / (1024 * 1024):.1f} |")
    print(f"\n{len(files)} files of version {args.version} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
