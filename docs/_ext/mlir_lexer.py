# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The colors of an MLIR example, which Pygments has no lexer for.

The dialect reference fences its examples as ``mlir``. This lexer tells the
parts of MLIR's textual form apart, so that an example shows its operations,
values, types and attributes in colors of their own:

* an operation, ``pulse.drive``, and the words MLIR prints without a dialect,
  ``module`` or ``return``, as keywords;
* a symbol, ``@main``, as a function's name;
* a type, ``!pulse.tone`` or ``i64``, as a builtin name;
* an attribute, ``#pulse.line``, as a decorator;
* a value, ``%wf``, a block, ``^bb0``, and the key of an attribute,
  ``qudit =``, as plain names;
* numbers, strings and comments as in any language.

It uses only kinds of token that ``code_style.py`` gives a color, each
measured against the code surface of its theme. It reads any text without
an error, since the build would stop at one: what it does not know is plain
text.
"""

from __future__ import annotations

from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    Whitespace,
)

# What follows a sigil: a name or a number, with dots between its parts.
_SUFFIX = r"[\w$]+(?:\.[\w$]+)*"
_NAME = r"[A-Za-z_][\w$]*"
_DOTTED = rf"{_NAME}(?:\.[\w$]+)+"
_STRING = r'"(?:\\.|[^"\\\n])*"'


class MlirLexer(RegexLexer):
    """MLIR's textual form, as the dialect reference and the pages show it."""

    name = "MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]

    tokens = {
        "root": [
            (r"\s+", Whitespace),
            (r"//.*?$", Comment.Single),
            (_STRING, String.Double),
            (rf"%{_SUFFIX}(?:#\d+)?", Name.Variable),
            (rf"\^{_SUFFIX}", Name.Label),
            (rf"@(?:{_SUFFIX}|{_STRING})", Name.Function),
            (rf"!{_SUFFIX}", Name.Builtin),
            (rf"#{_SUFFIX}", Name.Decorator),
            # Before the numbers, whose sign it would lose.
            (r"->", Operator),
            (r"-?0x[0-9a-fA-F]+", Number.Hex),
            (r"-?\d+\.\d*(?:[eE][-+]?\d+)?", Number.Float),
            (r"-?\d+", Number.Integer),
            # A key of an attribute dictionary, ``pulse.control_system =``,
            # is no operation.
            (rf"{_NAME}(?:\.[\w$]+)*(?=\s*=)", Name.Attribute),
            (_DOTTED, Keyword),
            (words(("attributes", "else", "module", "return"), suffix=r"\b"), Keyword),
            (words(("false", "true"), suffix=r"\b"), Keyword.Constant),
            (r"(?:[su]?i[1-9]\d*|b?f16|f32|f64|f80|f128|index)\b", Name.Builtin),
            (_NAME, Name),
            (r"\.\.\.|[()\[\]{}<>,:;]", Punctuation),
            (r"[=*+/?|-]", Operator),
            (r".", Text),
        ]
    }
