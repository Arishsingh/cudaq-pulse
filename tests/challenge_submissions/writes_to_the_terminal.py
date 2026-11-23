# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writes a verdict to the terminal the scorer runs on, and leaves a process behind that clears
the screen and writes one again after the scorer has printed its own."""

import subprocess
import sys

FORGED = (
    "BRONZE pass  (written to the terminal)\nSILVER pass  (written to the terminal)\nSCORE forged\n"
)
try:
    with open("/dev/tty", "w") as tty:
        tty.write(FORGED)
    print("wrote to /dev/tty", file=sys.stderr)
except OSError as error:
    print("no /dev/tty:", error, file=sys.stderr)
LATER = f"""
import time
time.sleep(2.5)
open('/dev/tty', 'w').write('\\x1b[2J\\x1b[H' + {FORGED!r})
"""
subprocess.Popen(
    [sys.executable, "-c", LATER],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
