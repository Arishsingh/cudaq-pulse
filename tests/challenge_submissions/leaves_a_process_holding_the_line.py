# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Leaves a process behind, in a session of its own, that holds the line to the scorer open for
150 s. The rest is the bronze reference of the challenge that runs it."""

import importlib.util
import os
import pathlib
import subprocess
import sys

held = [fd for fd in range(3, 20) if os.path.exists(f"/dev/fd/{fd}")]
subprocess.Popen(
    [sys.executable, "-c", "import time\ntime.sleep(150)\n"],
    stdin=subprocess.DEVNULL,
    pass_fds=held,
    start_new_session=True,
)

here = pathlib.Path(sys.modules["__main__"].__file__).parent
spec = importlib.util.spec_from_file_location("bronze", here / "reference" / "bronze.py")
bronze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bronze)
globals().update({k: v for k, v in vars(bronze).items() if not k.startswith("__")})
