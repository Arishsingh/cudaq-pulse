# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures that a script in ``_static`` draws from a JSON file, inlined in the page.

``.. toy-flow:: figures/toy_flow.json`` plays one scene of the file with
``toy-flow.js``. With ``:fallback: figures/name.svg``, the figure holds
``name.svg`` and ``name-dark.svg`` as a ``themed-figure`` does, and the
script hides them once it draws; its content is the caption::

    .. toy-flow:: figures/toy_flow.json
       :scene: flip_other
       :fallback: figures/isa_feedback.svg
       :alt: What the figure shows.

       The caption.

``.. memory-gauge:: figures/controller_memory.json`` draws the rows and
columns it names with ``mem-gauge.js``, over a table of the same numbers
for readers without JavaScript::

    .. memory-gauge:: figures/controller_memory.json
       :rows: train train_loop
       :columns: pulses

Each JSON file goes into the page once, in a ``<script type="application/
json">`` that the scripts read, so a page opened from a local copy of the
docs works with no fetch. A page loads a script and its stylesheet only
when it uses it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path, PurePosixPath

from docutils import nodes
from docutils.parsers.rst import Directive, directives

# The stylesheet and script each directive needs, by the class of its root.
_ASSETS = {"pulse-toy-flow": "toy-flow", "pulse-mem-gauge": "mem-gauge"}

# What the gauges' fallback table calls each column.
_COLUMNS = {
    "pulses": "pulses",
    "samples": "waveform samples",
    "frequencies": "frequencies on the line",
}


def _data_id(path: Path) -> str:
    return f"pulse-data-{path.stem.replace('_', '-')}"


class _DataFigure(Directive):
    required_arguments = 1
    has_content = True
    root_class = ""

    def data(self) -> tuple[str, dict, list[nodes.Node]]:
        """The JSON file's element id, its data, and the script that
        inlines it, the first time the page uses the file."""
        env = self.state.document.settings.env
        relative, path = env.relfn2path(self.arguments[0], env.docname)
        env.note_dependency(relative)
        text = Path(path).read_text()
        data_id = _data_id(Path(path))
        inlined = env.temp_data.setdefault("pulse_data_inlined", set())
        script = []
        if data_id not in inlined:
            inlined.add(data_id)
            # "</" would end the script element early; JSON reads "<\/" as "</".
            body = json.dumps(json.loads(text), separators=(",", ":")).replace("</", "<\\/")
            raw = f'<script type="application/json" id="{data_id}">{body}</script>'
            script.append(nodes.raw("", raw, format="html"))
        return data_id, json.loads(text), script

    def root(self, data_id: str, **attributes: str) -> nodes.raw:
        attrs = "".join(f' data-{k}="{html.escape(v)}"' for k, v in attributes.items())
        raw = f'<div class="{self.root_class}" data-source="{data_id}"{attrs}></div>'
        return nodes.raw("", raw, format="html")

    def caption(self) -> list[nodes.Node]:
        if not self.content:
            return []
        holder = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, holder)
        first = holder[0]
        if not isinstance(first, nodes.paragraph):
            return []
        caption = nodes.caption(first.rawsource, "", *first.children)
        caption.source, caption.line = first.source, first.line
        return [caption]


class ToyFlow(_DataFigure):
    option_spec = {
        "scene": directives.unchanged_required,
        "fallback": directives.unchanged,
        "alt": directives.unchanged,
    }
    root_class = "pulse-toy-flow"

    def run(self) -> list[nodes.Node]:
        data_id, data, script = self.data()
        scene = self.options["scene"]
        if scene not in {s["name"] for s in data["scenes"]}:
            raise self.error(f"{self.arguments[0]} has no scene {scene!r}")
        figure = nodes.figure(classes=["pulse-data-figure"])
        figure += script
        figure += self.root(data_id, scene=scene)
        if "fallback" in self.options:
            light = PurePosixPath(self.options["fallback"])
            dark = light.with_name(f"{light.stem}-dark{light.suffix}")
            box = nodes.container(classes=["themed-figure", "pulse-fallback"])
            alt = self.options.get("alt", "")
            box += nodes.image(uri=str(light), alt=alt, classes=["only-light"], width="100%")
            box += nodes.image(uri=str(dark), alt=alt, classes=["only-dark"], width="100%")
            figure += box
        figure += self.caption()
        return [figure]


class MemoryGauge(_DataFigure):
    option_spec = {
        "rows": directives.unchanged_required,
        "columns": directives.unchanged_required,
    }
    root_class = "pulse-mem-gauge"

    def run(self) -> list[nodes.Node]:
        data_id, data, script = self.data()
        rows = self.options["rows"].split()
        columns = self.options["columns"].split()
        by_name = {row["name"]: row for row in data["rows"]}
        unknown = [r for r in rows if r not in by_name] + [c for c in columns if c not in _COLUMNS]
        if unknown:
            raise self.error(f"{self.arguments[0]} has no row or column {', '.join(unknown)}")

        # The same numbers as a table, which the script hides once it draws.
        head = "".join(f"<th>{html.escape(_COLUMNS[c])}</th>" for c in columns)
        limits = "".join(f"<td>{data['limits'][c]:,}</td>" for c in columns)
        body = [f"<tr><th>the controller holds</th>{limits}</tr>"]
        for name in rows:
            row = by_name[name]
            cells = "".join(f"<td>{row[c]:,}</td>" for c in columns)
            label = html.escape(row["label"]) + (" (refused)" if row["refused"] else "")
            body.append(f"<tr><th>{label}</th>{cells}</tr>")
        table = (
            '<table class="pulse-fallback mg-table">'
            f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        )

        figure = nodes.figure(classes=["pulse-data-figure"])
        figure += script
        figure += self.root(data_id, rows=" ".join(rows), columns=" ".join(columns))
        figure += nodes.raw("", table, format="html")
        figure += self.caption()
        return [figure]


def _add_assets(app, pagename, templatename, context, doctree) -> None:
    # Files added here load on this page only: Sphinx resets them per page.
    if doctree is None:
        return
    shown = " ".join(raw.astext() for raw in doctree.findall(nodes.raw))
    for root, name in _ASSETS.items():
        if f'class="{root}"' in shown:
            app.add_css_file(f"{name}.css")
            app.add_js_file(f"{name}.js", defer="defer")


def setup(app):
    app.add_directive("toy-flow", ToyFlow)
    app.add_directive("memory-gauge", MemoryGauge)
    app.connect("html-page-context", _add_assets)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
