#!/usr/bin/env python3
"""Convert colored terminal output (ANSI) on stdin to an SVG, via rich.

Usage:  ... | python scripts/ansi2svg.py OUT.svg ["title"]

Works for any jobscope plot (plotext line/hist + rich heat/bars): it re-renders
the ANSI through a recording rich Console. Needs only rich, which jobscope
already depends on.
"""

import sys

from rich.console import Console
from rich.text import Text

out = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else ""
text = Text.from_ansi(sys.stdin.read().rstrip("\n"))
width = max((len(line) for line in text.plain.splitlines()), default=80)
console = Console(record=True, force_terminal=True, width=max(60, width + 1))
console.print(text)
console.save_svg(out, title=title)
print("  wrote %s (%d lines)" % (out, text.plain.count("\n") + 1), file=sys.stderr)
