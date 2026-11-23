# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Never answers. Leaves two helpers behind, one in its own process group and one in a group of
its own in the same session; each writes the time into a file of CHALLENGE_TEST_OUT every 0.2 s."""

import os
import subprocess
import sys
import time

out = os.environ["CHALLENGE_TEST_OUT"]
for name, options in (("same_group", {}), ("new_group", {"process_group": 0})):
    code = (
        "import time\nwhile True:\n"
        f"    open({out!r} + '/tick_{name}', 'w').write(str(time.time()))\n"
        "    time.sleep(0.2)\n"
    )
    subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **options,
    )
time.sleep(100000)
