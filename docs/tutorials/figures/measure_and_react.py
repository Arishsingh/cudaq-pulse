# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Measure and React" tutorial.

Usage: python docs/tutorials/figures/measure_and_react.py

Needs matplotlib (the ``[viz]`` extra). Writes ``reset_schedule.svg`` next
to this file, from the active reset the page defines.
"""

from _style import page_namespace, save


def main() -> None:
    page = page_namespace("measure_and_react.rst")
    save(page["active_reset"].lower(page["q"]).timeline(figsize=(8, 2.2)), "reset_schedule.svg")


if __name__ == "__main__":
    main()
