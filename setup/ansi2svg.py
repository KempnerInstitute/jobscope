#!/usr/bin/env python3.12
"""Convert colored terminal output (ANSI) on stdin to an SVG, via rich.
Usage:  ... | python3.12 setup/ansi2svg.py OUT.svg ["title"]
Works for any plot (plotext line/hist + rich heat/bars) -- it just re-renders the
ANSI through a recording rich Console. Needs only rich (already a jobstats_plot dep)."""
import sys
from rich.console import Console
from rich.text import Text

out = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else ""
text = Text.from_ansi(sys.stdin.read().rstrip("\n"))
width = max((len(line) for line in text.plain.splitlines()), default=80)
con = Console(record=True, force_terminal=True, width=max(60, width + 1))
con.print(text)
con.save_svg(out, title=title)
print("  wrote %s (%d lines)" % (out, text.plain.count("\n") + 1), file=sys.stderr)
