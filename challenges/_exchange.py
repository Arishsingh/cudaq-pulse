# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the scorers of the challenges share.

A scorer never loads a submission. It starts a second Python process (the challenge's _child.py),
which loads the file and compiles its kernels, and reads from it only JSON: numbers that say what
a kernel plays. This module starts that process (``start_child``, ``ask_child``) in a session of
its own, bounded in time and in the size of what it sends, and ends it with everything it started
(``stop_child``). Nothing it sends is trusted: ``read_driven`` checks every field before anything
is compared or replayed.

Text from the second process never stands in a line of the scorer's. What the file prints is
shown as it comes, and a message it sends is shown under the scorer's own words (``said``), each
line with MARK in front and short enough not to wrap.

A scorer loads and runs everything it will need before it starts the second process (each has a
``warm_up``); from then on it loads no module (``start_child`` sees to it), so a file that the
submission writes while it runs is not executed by the scorer in that run.
"""

import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

CHILD_TIMEOUT_S = 120  # for a process that compiles and answers once
LAB_TIMEOUT_S = 900  # for a process that plays through the scorer, run after run
MAX_REPLY_BYTES = 4_000_000  # an honest reply is a tenth of it
MAX_LINE_CHARS = 4_000_000  # one line of an exchange that goes back and forth
MAX_CYCLES = 200_000  # 100 us at the 2 GHz clock: nothing that starts or lasts longer is read
MAX_SAMPLES = 2048  # in one pulse or readout: 1024 ns at the 2 GHz clock
MAX_EVENTS = 512  # pulses, readouts or waits
MAX_SAMPLE = 1.0e6  # the real or the imaginary part of one sample, in magnitude
MAX_TONE = 1.0e12  # a frequency in Hz or a phase in rad, in magnitude
MAX_SHOWN_LINES = 200  # of what the submission prints
MAX_SAID_CHARS = 300  # of a message from the second process
MARK = "  | "
SAID_WIDTH = 72  # with MARK, under the 80 columns of a narrow terminal


class InvalidSubmissionError(Exception):
    """The submission cannot be scored; the message says why, in lines safe to print."""


class LateImportError(ImportError):
    """The scorer would have had to load a module after the second process started."""


def one_line(text):
    """*text* from the second process as one short line of printable ASCII characters."""
    plain = "".join(c if " " <= c <= "~" else " " if c.isspace() else "?" for c in str(text))
    return " ".join(plain.split())[:MAX_SAID_CHARS]


def said(text):
    """*text* from the second process, to print after the scorer's own words.

    It goes on lines of its own, each marked and at most SAID_WIDTH characters long: whatever
    the text holds, none of it starts a line or wraps onto one.
    """
    lines = textwrap.wrap(one_line(text), SAID_WIDTH, break_long_words=True) or ["(nothing)"]
    return "".join(f"\n{MARK}{line}" for line in lines)


def scorer_of(challenge):
    """The score.py of another challenge, as a module, for a function it shares."""
    path = Path(__file__).resolve().parent / challenge / "score.py"
    spec = importlib.util.spec_from_file_location(f"_{challenge}_score", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_scorer(main, usage, seeded=True):
    """Call main(path[, seed]) with the command line's arguments and exit with what it returns."""
    arguments = sys.argv[1:]
    if not 1 <= len(arguments) <= (2 if seeded else 1):
        sys.exit(usage)
    seed = []
    if len(arguments) == 2:
        if not re.fullmatch(r"[0-9]{1,18}", arguments[1]):
            sys.exit(f"{usage}\nthe seed is a whole number, 0 or more, of at most 18 digits")
        seed = [int(arguments[1])]
    try:
        code = main(arguments[0], *seed)
    except LateImportError as error:
        print(f"INVALID {error}; nothing is scored\nBRONZE fail\nSILVER fail\nSCORE none")
        code = 1
    except KeyboardInterrupt:
        sys.exit("interrupted: no verdict")
    sys.stdout.flush()
    sys.exit(code)


# The scorer's side.


class _NoLaterImports:
    """First on sys.meta_path once the second process runs: no module is loaded after that."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        raise LateImportError(
            f"the scorer does not load a module ({one_line(name)[:60]}) once the submission's process has started"
        )


def start_child(child, arguments, text=False, time_limit_s=None):
    """The second process, started in a session of its own, with no terminal.

    What it prints for a person (its standard error) is shown as it comes, each line marked as
    the submission's. From here on the scorer loads no module. With a time limit, the process
    is ended when the limit is over, and ``read_message`` then says so.
    """
    if not any(isinstance(finder, _NoLaterImports) for finder in sys.meta_path):
        sys.meta_path.insert(0, _NoLaterImports())
    process = subprocess.Popen(
        [sys.executable, str(child), *arguments],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        **({"text": True, "errors": "replace"} if text else {}),
    )

    def show():
        shown = 0
        while True:
            line = process.stderr.readline(4096)
            if not line:
                break
            shown += 1
            if shown <= MAX_SHOWN_LINES:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", "replace")
                print(f"{MARK}{one_line(line)[:SAID_WIDTH]}", file=sys.stderr)

    threading.Thread(target=show, daemon=True).start()
    process.late = False
    if time_limit_s is not None:

        def too_late():
            process.late = True
            stop_child(process)

        process.watch = threading.Timer(time_limit_s, too_late)
        process.watch.daemon = True
        process.watch.start()
        process.time_limit_s = time_limit_s
    return process


def _in_session(session):
    """The processes whose session is *session*, of those ps lists; none where there is no ps.

    The session is asked of the system for each process: the session column of ps is not the
    session's number on every system (macOS prints 0 in it).
    """
    try:
        listed = subprocess.run(
            ["ps", "-A", "-o", "pid="],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for word in listed.split():
        try:
            if os.getsid(int(word)) == session:
                found.append(int(word))
        except (OSError, ValueError):  # it ended since ps listed it, or may not be asked about
            pass
    return found


def stop_child(process):
    """End the second process and every process still in its session.

    Its process group first, then whatever else ps finds in the session (a process in a group
    of its own). A process that started a session of its own is not found.
    """
    if (
        getattr(process, "watch", None) is not None
        and threading.current_thread() is not process.watch
    ):
        process.watch.cancel()
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:  # the group is empty already
            pass
        for pid in _in_session(process.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    process.kill()
    process.wait(timeout=30)


def ask_child(child, arguments, message=None):
    """The reply of the second process: the last line it writes, a JSON object.

    The process has CHILD_TIMEOUT_S to end. A process it leaves behind is ended with it, and
    one that got away with the line to the scorer in its hands is not waited for.
    """
    process = start_child(child, arguments)
    chunks, size = [], 0

    def read():
        nonlocal size
        while size <= MAX_REPLY_BYTES:
            chunk = process.stdout.read1(65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        process.stdin.write(json.dumps(message).encode() + b"\n")
        process.stdin.close()
    except OSError:  # the process ended before it read
        pass
    deadline = time.monotonic() + CHILD_TIMEOUT_S
    while process.poll() is None and size <= MAX_REPLY_BYTES and time.monotonic() < deadline:
        time.sleep(0.02)
    late = process.poll() is None and size <= MAX_REPLY_BYTES
    stop_child(process)
    reader.join(2)
    if late:
        raise InvalidSubmissionError(f"it did not finish within {CHILD_TIMEOUT_S} s")
    if size > MAX_REPLY_BYTES:
        raise InvalidSubmissionError(f"its process wrote more than {MAX_REPLY_BYTES} bytes")
    lines = b"".join(list(chunks)).decode("utf-8", "replace").split("\n")
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise InvalidSubmissionError("its process ended without a result")
    try:
        reply = json.loads(lines[-1])
    except (ValueError, RecursionError):
        reply = None
    if not isinstance(reply, dict):
        raise InvalidSubmissionError("its result could not be read")
    return reply


def read_message(process):
    """One message of an exchange that goes back and forth: a line of JSON, an object."""
    line = process.stdout.readline(MAX_LINE_CHARS)
    if process.late:
        raise InvalidSubmissionError(f"it did not finish within {process.time_limit_s} s")
    if not line:
        raise InvalidSubmissionError("its process ended before it answered")
    try:
        message = json.loads(line) if line.endswith("\n") else None
    except (ValueError, RecursionError):
        message = None
    if not isinstance(message, dict):
        raise InvalidSubmissionError("its process sent something that is not a message")
    return message


def _whole(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(value)
    return value


def _bounded(value, limit, what):
    """A finite number of at most *limit* in magnitude; a larger one is refused by name."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(value)
    if abs(value) > limit:
        raise InvalidSubmissionError(
            f"{what} is over {limit:g} in magnitude; none that large is read"
        )
    return float(value)


def _samples(item):
    real, imag = item["re"], item["im"]
    if type(real) is not list or type(imag) is not list or len(real) != len(imag) or not real:
        raise ValueError("samples")
    if len(real) > MAX_SAMPLES:
        raise InvalidSubmissionError(
            f"a pulse or readout lasts {len(real)} clock cycles; none longer than {MAX_SAMPLES} ({MAX_SAMPLES // 2} ns) is read"
        )
    part = "a sample's real or imaginary part"
    return [
        complex(_bounded(a, MAX_SAMPLE, part), _bounded(b, MAX_SAMPLE, part))
        for a, b in zip(real, imag, strict=True)
    ]


def _tone(item):
    """(frequency in Hz, phase in rad) of a pulse or readout."""
    return (
        _bounded(item["frequency_hz"], MAX_TONE, "a frequency"),
        _bounded(item["phase_rad"], MAX_TONE, "a phase"),
    )


def _flag(value):
    if type(value) is not bool:
        raise ValueError(value)
    return value


def _items(value):
    if type(value) is not list:
        raise ValueError("list")
    if len(value) > MAX_EVENTS:
        raise InvalidSubmissionError(
            f"it has {len(value)} pulses, readouts or waits; no more than {MAX_EVENTS} of a kind are read"
        )
    return value


def _when(item, length):
    """(start, end) in clock cycles; an event that starts or lasts beyond MAX_CYCLES is refused."""
    start = item["start_cycles"]
    for value in (start, length):
        if type(value) is int and value > MAX_CYCLES:
            raise InvalidSubmissionError(
                f"a pulse, readout or wait starts at or lasts {min(value, 10**12)} clock cycles; nothing beyond {MAX_CYCLES} ({MAX_CYCLES // 2000} us) is read"
            )
    start = _whole(start, 0, MAX_CYCLES)
    return start, start + _whole(length, 0, MAX_CYCLES)


def _pulse(item, readouts_count=None):
    samples = _samples(item)
    start, end = _when(item, len(samples))
    frequency_hz, phase_rad = _tone(item)
    pulse = {
        "qudit": _whole(item["qudit"], 0, 10_000),
        "start": start,
        "end": end,
        "samples": samples,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
    }
    if readouts_count is None:  # a readout
        pulse["in_branch"] = _flag(item["in_branch"])
    else:  # a drive: the readouts it depends on, as (readout, arm)
        depends = _items(item["depends"])
        pulse["depends"] = [
            (_whole(readout, 0, readouts_count - 1), _flag(arm)) for readout, arm in depends
        ]
    return pulse


def read_driven(reply):
    """What the reply says a kernel plays, every field checked: (drives, readouts, waits, others).

    A drive or readout is a dict: qudit, start and end (clock cycles), samples (complex),
    frequency_hz, phase_rad; a drive also has depends, the (readout, arm) pairs of the branches it
    is in, and a readout in_branch. A wait is (qudit, start, end, whether it is in a branch,
    whether it is on a drive line). An other is the name of anything else the kernel has, one
    short printable line. A reply that cannot be read, or that breaks one of this module's
    limits, raises InvalidSubmissionError, which says which.
    """
    try:
        readouts = [_pulse(item) for item in _items(reply["readouts"])]
        drives = [_pulse(item, len(readouts)) for item in _items(reply["drives"])]
        waits = []
        for item in _items(reply["waits"]):
            start, end = _when(item, item["duration_cycles"])
            qudit, branch = _whole(item["qudit"], 0, 10_000), _flag(item["in_branch"])
            waits.append((qudit, start, end, branch, _flag(item["on_drive_line"])))
        others = [kind_name(kind) for kind in _items(reply["others"])]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise InvalidSubmissionError("what it plays could not be read") from None
    return drives, readouts, waits, others


def read_pulses(reply):
    """The reply of Find the pi pulse, every field checked: (pulses in time order, others, end).

    A pulse is a dict: start (clock cycles), samples (complex), frequency_hz, phase_rad. The
    end is that of the kernel's last event, a wait included, in clock cycles.
    """
    try:
        pulses = []
        for item in _items(reply["pulses"]):
            samples = _samples(item)
            start, _ = _when(item, len(samples))
            frequency_hz, phase_rad = _tone(item)
            pulses.append(
                {
                    "start": start,
                    "samples": samples,
                    "frequency_hz": frequency_hz,
                    "phase_rad": phase_rad,
                }
            )
        others = [kind_name(kind) for kind in _items(reply["others"])]
        _, end = _when({"start_cycles": 0}, reply["total_cycles"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise InvalidSubmissionError("its pulses could not be read") from None
    return sorted(pulses, key=lambda pulse: pulse["start"]), others, end


# What a kernel can have besides pulses, readouts and waits, by the name the second process
# gives it. A name that is not here is the submission's own text, and is not shown.
KINDS = {
    "frame change": "a frame change",
    "frame_update": "a frame change",
    "readout": "a readout",
    "sync": "a sync",
    "submit": "a call of classical code",
    "answer": "an answer of classical code",
    "drive whose frame depends on a measurement": "a pulse whose frame depends on a measurement",
    "readout whose frame depends on a measurement": "a readout whose frame depends on a measurement",
    "wait whose length depends on a measurement": "a wait whose length depends on a measurement",
    "branch on more than the outcome of one readout": "a branch on more than the outcome of one readout",
}


def kind_name(kind):
    """What the second process calls something else a kernel has, in the scorer's own words."""
    return KINDS.get(kind if type(kind) is str else "", "an operation the scorer does not know")


def digits(value, limit):
    """*value*, which is over *limit*, with enough digits to show that it is."""
    return f"{value:.4f}" if f"{value:.4f}" != f"{limit:.4f}" else f"{value:.12g}"


def counted(count, noun):
    """'1 pulse', '2 pulses'."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def same_pulse(one, other):
    """Whether two drives or readouts send the same samples on the same tone."""
    turn = (one["phase_rad"] - other["phase_rad"]) % (2.0 * math.pi)
    return (
        len(one["samples"]) == len(other["samples"])
        and all(abs(a - b) <= 1e-9 for a, b in zip(one["samples"], other["samples"], strict=True))
        and abs(one["frequency_hz"] - other["frequency_hz"]) <= 1e-3
        and min(turn, 2.0 * math.pi - turn) <= 1e-9
    )


# The second process's side.


def explain(error):
    """An exception as text: its type, its message and the kernel line the compiler names."""
    said = [line.strip() for line in str(error).splitlines() if line.strip()]
    text = f"{type(error).__name__}: {' '.join(said[:2])}"  # a list of limits: its first entry
    notes = "\n".join(str(note) for note in getattr(error, "__notes__", ()))
    where = re.search(r"line (\d+), in \w+\n\s*(.+)", notes)
    if where:
        text += f" (line {where.group(1)}: {where.group(2).strip()})"
    return text


def answer(make_reply):
    """Load the file named by the first argument and write make_reply(module) for the scorer.

    The scorer's line is taken away from the file first: what it prints goes to standard error.
    """
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
        if spec is None or spec.loader is None:
            raise ImportError(f"{sys.argv[1]} is not a Python file")
        submission = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submission)
        reply = make_reply(submission)
    except BaseException as error:
        reply = {"error": explain(error)}
    channel.write(json.dumps(reply) + "\n")
    channel.flush()


def driven(program):
    """What a lowered program plays, as plain numbers for ``read_driven``."""
    import numpy as np
    from cudaq_pulse.controller import sample_waveform

    export = program.export_to_controller()
    role = {line["line"]: line["role"] for line in export["lines"]}
    number = {}  # a readout's place among the readouts, by its event index
    drives, readouts, waits, others = [], [], [], []
    for event in export["events"]:
        kind, conditions = event["kind"], event.get("conditions") or []
        if kind in ("drive", "readout"):
            if event["frequency_hz"] is None or event["phase_rad"] is None:
                others.append(f"{kind} whose frame depends on a measurement")
                continue
            count = int(event["duration_cycles"])
            samples = np.asarray(sample_waveform(event["waveform"], count), dtype=complex)
            item = {
                "qudit": int(event["qudit"]),
                "start_cycles": int(event["start_cycles"]),
                "re": [float(x) for x in samples.real],
                "im": [float(x) for x in samples.imag],
                "frequency_hz": float(event["frequency_hz"]),
                "phase_rad": float(event["phase_rad"]),
            }
            if kind == "readout":
                number[event["index"]] = len(readouts)
                item["in_branch"] = bool(conditions)
                readouts.append(item)
                continue
            simple = all(
                c["test"] == "is_set" and not c["answers"] and c["readouts"][0] in number
                for c in conditions
            )
            if not simple:
                others.append("branch on more than the outcome of one readout")
                continue
            item["depends"] = [[number[c["readouts"][0]], bool(c["arm"])] for c in conditions]
            drives.append(item)
        elif kind == "wait":
            if "length" in event:
                others.append("wait whose length depends on a measurement")
                continue
            waits.append(
                {
                    "qudit": int(event["qudit"]),
                    "start_cycles": int(event["start_cycles"]),
                    "duration_cycles": int(event["duration_cycles"]),
                    "in_branch": bool(conditions),
                    "on_drive_line": role.get(event["line"]) == "drive",
                }
            )
        elif kind != "sync":
            others.append("frame change" if kind == "frame_update" else str(kind))
    return {"drives": drives, "readouts": readouts, "waits": waits, "others": others}
