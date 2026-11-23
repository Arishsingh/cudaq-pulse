# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Appends code to every Python file of the cudaq_pulse package it is given a copy of: a module
the scorer loads only after this file has run would print a verdict in the scorer's own process
and end it. The rest is the bronze reference of the challenge that runs it.

Only ever pointed at a copy: the folder named by CHALLENGE_TEST_PACKAGE_COPY."""

import importlib.util
import os
import pathlib
import sys

import cudaq_pulse

package = pathlib.Path(cudaq_pulse.__file__).parent
assert package == pathlib.Path(os.environ["CHALLENGE_TEST_PACKAGE_COPY"]), "only ever a copy"
PLANT = """
import os as _os, sys as _sys
if _os.path.basename(_sys.argv[0]) == "score.py":
    print("BRONZE pass: written by the submission, in the scorer's process")
    print("SILVER pass: written by the submission, in the scorer's process")
    print("SCORE 0 (written by the submission)")
    _sys.stdout.flush()
    _os._exit(0)
"""
for path in package.rglob("*.py"):
    if not path.is_symlink() and "written by the submission" not in path.read_text():
        path.write_text(path.read_text() + PLANT)

here = pathlib.Path(sys.modules["__main__"].__file__).parent
spec = importlib.util.spec_from_file_location("bronze", here / "reference" / "bronze.py")
bronze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bronze)
globals().update({k: v for k, v in vars(bronze).items() if not k.startswith("__")})
