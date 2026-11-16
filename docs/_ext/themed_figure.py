# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A figure drawn once per theme.

``.. themed-figure:: figures/name.svg`` takes the options of ``figure`` and
shows ``name.svg`` in the light theme and ``name-dark.svg`` in the dark one,
under one caption. The two images carry the theme's ``only-light`` and
``only-dark`` classes, which follow the reader's theme switch and keep the
dark image out of the theme's dimming of images in dark mode. They sit in a
container of their own, since the theme hides a caption that follows a
hidden image directly.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from docutils import nodes
from docutils.parsers.rst.directives.images import Figure


class ThemedFigure(Figure):
    def run(self) -> list[nodes.Node]:
        result = super().run()
        figure = result[0]
        if not isinstance(figure, nodes.figure):
            return result
        light = next(iter(figure.findall(nodes.image)))
        uri = PurePosixPath(light["uri"])
        dark = light.deepcopy()
        dark["uri"] = str(uri.with_name(f"{uri.stem}-dark{uri.suffix}"))
        light["classes"].append("only-light")
        dark["classes"].append("only-dark")
        holder = light.parent
        box = nodes.container(classes=["themed-figure"])
        holder.replace(light, box)
        box += [light, dark]
        return result


def setup(app):
    app.add_directive("themed-figure", ThemedFigure)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
