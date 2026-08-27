"""Read a completionist dataset interactively.

Usage:
    completionist inspect <dataset.parquet|dataset.jsonl|hf-dataset-name>

The format is auto-detected per sample (conversation, prompt/completion,
tools, or a raw JSON fallback). Local files, local directories following the
HF split layout, and remote datasets are all read via streaming.

Keys:
    left / right     previous / next sample
    tab / shift-tab  cycle view
    s                cycle split (when the dataset has several)
    up / down        scroll
    PgUp / PgDn      page up / down
    Home / End       jump to top / bottom
    q / Esc          quit
"""

import curses
import os
import sys
import textwrap
import time

import click

from completionist.dataset_stream import DatasetStream
from completionist.dataset_view import (
    ROLE_PREFIX,
    detect_format,
    render_sample_text,
    row_to_tabs,
    title,
)
from completionist.pager import Pager
from completionist.utils import handle_error


def _wrap_segments(segments, width):
    """Wrap (style, text) segments into display lines for the given width."""
    width = max(1, width)
    lines = []
    for style, text in segments:
        if style == "raw":
            lines.extend((style, ln) for ln in text.splitlines())
            continue
        prefix = ROLE_PREFIX.get(style, "")
        if prefix:
            wrapped = textwrap.wrap(text, width=max(1, width - len(prefix))) or [""]
            lines.append((style, prefix + wrapped[0]))
            for cont in wrapped[1:]:
                lines.append((style, " " * len(prefix) + cont))
        else:
            for chunk in textwrap.wrap(text, width=width) or [""]:
                lines.append((style, chunk))
    return lines


def _tui(stdscr, source, pager):
    curses.curs_set(0)
    stdscr.keypad(True)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)  # user
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # assistant
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # topic
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_WHITE)  # bars
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # tool

    attrs = {
        "user": curses.color_pair(1),
        "assistant": curses.color_pair(2),
        "topic": curses.color_pair(3) | curses.A_BOLD,
        "tool": curses.color_pair(5),
        "system": curses.A_BOLD,
        "raw": curses.A_NORMAL,
        "plain": curses.A_NORMAL,
    }

    tab_idx = 0
    scroll = 0

    while True:
        row = pager.current()
        fmt = detect_format(row)
        tabs = row_to_tabs(row, fmt)
        tab_idx %= len(tabs)
        _, segments = tabs[tab_idx]

        stdscr.erase()
        H, W = stdscr.getmaxyx()
        body_h = max(1, H - 3)

        total = pager.total
        split_label = f" [{source.split}]" if source.split else ""
        header = (
            f" [{pager.cursor + 1}/{total if total is not None else '?'}]"
            f"{split_label}  {title(row, fmt)}"
        )
        try:
            stdscr.addnstr(0, 0, header, W, curses.color_pair(4) | curses.A_BOLD)
        except curses.error:
            pass

        tabbar = "  ".join(
            f"[{name}]" if i == tab_idx else name for i, (name, _) in enumerate(tabs)
        )
        try:
            stdscr.addnstr(1, 0, tabbar, W, curses.color_pair(4))
        except curses.error:
            pass

        lines = _wrap_segments(segments, W)
        max_scroll = max(0, len(lines) - body_h)
        scroll = max(0, min(scroll, max_scroll))
        for i in range(body_h):
            li = scroll + i
            if li >= len(lines):
                break
            style, text = lines[li]
            try:
                stdscr.addnstr(2 + i, 0, text, W, attrs.get(style, curses.A_NORMAL))
            except curses.error:
                pass

        keys = ["←/→ sample", "Tab/Shift-Tab view"]
        if len(source.splits) > 1:
            keys.append("s split")
        keys += ["↑/↓ scroll", "PgUp/PgDn page", "q quit"]
        footer = " " + "   ".join(keys) + " "
        try:
            stdscr.addnstr(
                H - 1,
                0,
                footer.ljust(W)[:W],
                W,
                curses.color_pair(4) | curses.A_REVERSE,
            )
        except curses.error:
            pass

        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            break
        elif key == curses.KEY_LEFT:
            pager.prev()
            scroll = 0
        elif key == curses.KEY_RIGHT:
            pager.next()
            scroll = 0
        elif key == ord("s"):
            if source.switch_split():
                try:
                    first = source.next()
                except StopIteration:
                    continue
                pager.close()
                pager = Pager(source, first)
                tab_idx = 0
                scroll = 0
        elif key == ord("\t"):
            tab_idx += 1
            scroll = 0
        elif key == curses.KEY_BTAB:
            tab_idx -= 1
            scroll = 0
        elif key == curses.KEY_UP:
            scroll -= 1
        elif key == curses.KEY_DOWN:
            scroll += 1
        elif key == curses.KEY_PPAGE:
            scroll -= body_h
        elif key == curses.KEY_NPAGE:
            scroll += body_h
        elif key == curses.KEY_HOME:
            scroll = 0
        elif key == curses.KEY_END:
            scroll = max_scroll


@click.command("inspect")
@click.argument("dataset")
@click.option(
    "--split",
    default=None,
    help="Split to load (auto-detected if omitted).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Load at most this many samples.",
)
@click.option(
    "--index",
    type=int,
    default=0,
    show_default=True,
    help="Skip this many samples before reading.",
)
def inspect_cmd(dataset, split, limit, index):
    """Read a completionist dataset, auto-detecting its format.

    Interactive viewer (or a plain-text dump when stdout is not a TTY) with
    per-sample tabs for conversations, prompt/completion, and tools.
    """
    if index < 0:
        handle_error("--index must be >= 0.")
    if limit is not None and limit < 1:
        handle_error("--limit must be >= 1.")

    pager = None
    try:
        tty = sys.stdout.isatty()
        if tty:
            print("Loading dataset...", file=sys.stderr, flush=True)
            start = time.monotonic()
        source = DatasetStream(dataset, split, index, limit)
        try:
            first = source.next()
        except StopIteration:
            handle_error(f"No samples found in '{dataset}'.")
        if tty:
            print(
                f"done in {time.monotonic() - start:.1f}s", file=sys.stderr, flush=True
            )

        if not sys.stdout.isatty():
            total = source.total
            print(render_sample_text(first, detect_format(first), 0, total))
            n = 1
            try:
                while True:
                    row = source.next()
                    print(render_sample_text(row, detect_format(row), n, total))
                    n += 1
            except StopIteration:
                pass
            return

        pager = Pager(source, first)
        curses.wrapper(_tui, source, pager)
    except KeyboardInterrupt:
        if pager is not None:
            pager.close()
        # datasets/pyarrow leave background threads that abort the interpreter
        # during finalization on Ctrl-C; exit hard to skip the fatal error.
        os._exit(130)
    finally:
        if pager is not None:
            pager.close()
