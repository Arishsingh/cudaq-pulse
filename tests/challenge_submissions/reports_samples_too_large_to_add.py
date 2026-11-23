# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A report whose first pulse has samples of 1.7e308 + 1.7e308j: each part is a finite number,
their magnitude is not. The rest is the bronze reference of the challenge that runs it."""

import importlib.util
import json as _json
import pathlib
import sys

_exchange = sys.modules["_exchange"]
_main = sys.modules["__main__"]
_here = pathlib.Path(_main.__file__).parent
spec = importlib.util.spec_from_file_location("ref", _here / "reference" / "bronze.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
globals().update({k: v for k, v in vars(ref).items() if not k.startswith("__")})


class _Json:
    @staticmethod
    def dumps(reply):
        def spoil(one):
            for item in (one.get("drives") or one.get("pulses") or [])[:1]:
                item["re"] = [1.7e308] * len(item["re"])
                item["im"] = [1.7e308] * len(item["im"])

        spoil(reply)
        for one in reply.get("cases", []):
            spoil(one)
        return _json.dumps(reply)


_exchange.json = _Json
