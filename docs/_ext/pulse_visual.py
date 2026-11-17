# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Animated figures that ``pulse-visual.js`` draws from a JSON file inlined in the page.

``.. pulse-visual:: figures/t_state_injection.json`` plays one scene of the
file. The figure also holds the scene's still frame, ``<scene>.svg`` and
``<scene>-dark.svg`` next to the JSON, as a ``themed-figure`` does: readers
without JavaScript see it, and the script hides it once it draws. The
directive's content is the caption::

    .. pulse-visual:: figures/t_state_injection.json
       :scene: herald_loop
       :alt: What the figure shows, the still's description.

       The caption.

Each JSON file goes into the page once, in a ``<script type="application/
json">``, so a page opened from a local copy works with no fetch. A page
loads the script and its stylesheet only when it uses them.
"""

from __future__ import annotations

import html
import json
from pathlib import Path, PurePosixPath

from docutils import nodes
from docutils.parsers.rst import Directive, directives

FORMAT = "pulse-visual/1"
_ROOT = "pulse-visual"


def _data_id(path: Path) -> str:
    return f"pulse-data-{path.stem.replace('_', '-')}"


class PulseVisual(Directive):
    required_arguments = 1
    has_content = True
    option_spec = {
        "scene": directives.unchanged_required,
        "alt": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        env = self.state.document.settings.env
        relative, path = env.relfn2path(self.arguments[0], env.docname)
        env.note_dependency(relative)
        data = json.loads(Path(path).read_text())
        if data.get("format") != FORMAT:
            raise self.error(f"{self.arguments[0]} is not {FORMAT} data")
        scene = self.options.get("scene", "")
        if scene not in {s["name"] for s in data["scenes"]}:
            raise self.error(f"{self.arguments[0]} has no scene {scene!r}")

        figure = nodes.figure(classes=["pulse-visual-figure"])
        data_id = _data_id(Path(path))
        inlined = env.temp_data.setdefault("pulse_visual_inlined", set())
        if data_id not in inlined:
            inlined.add(data_id)
            # "</" would end the script element early; JSON reads "<\/" as "</".
            body = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
            body = body.replace("</", "<\\/")
            script = f'<script type="application/json" id="{data_id}">{body}</script>'
            figure += nodes.raw("", script, format="html")
        root = (
            f'<div class="{_ROOT}" data-source="{data_id}" data-scene="{html.escape(scene)}"></div>'
        )
        figure += nodes.raw("", root, format="html")

        light = PurePosixPath(self.arguments[0]).with_name(f"{scene}.svg")
        dark = light.with_name(f"{scene}-dark.svg")
        alt = self.options.get("alt", "")
        box = nodes.container(classes=["themed-figure", "pulse-fallback"])
        box += nodes.image(uri=str(light), alt=alt, classes=["only-light"], width="100%")
        box += nodes.image(uri=str(dark), alt=alt, classes=["only-dark"], width="100%")
        figure += box

        if self.content:
            holder = nodes.Element()
            self.state.nested_parse(self.content, self.content_offset, holder)
            first = holder[0]
            if isinstance(first, nodes.paragraph):
                caption = nodes.caption(first.rawsource, "", *first.children)
                caption.source, caption.line = first.source, first.line
                figure += caption
        return [figure]


def _add_assets(app, pagename, templatename, context, doctree) -> None:
    # Files added here load on this page only: Sphinx resets them per page.
    if doctree is None:
        return
    if any(f'class="{_ROOT}"' in raw.astext() for raw in doctree.findall(nodes.raw)):
        app.add_css_file("pulse-visual.css")
        app.add_js_file("pulse-visual.js", defer="defer")


def setup(app):
    app.add_directive("pulse-visual", PulseVisual)
    app.connect("html-page-context", _add_assets)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
