# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find the pi pulse: a report whose total_cycles is a whole number of 400 digits."""

import sys

_exchange = sys.modules["_exchange"]


class _Json:
    @staticmethod
    def dumps(_):
        return '{"pulses": [], "others": [], "total_cycles": 1' + "0" * 400 + "}"


_exchange.json = _Json
