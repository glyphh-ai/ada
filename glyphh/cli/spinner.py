"""
Braille spinner with prompt-style prefix.

Shows: glyphh> ⣾ thinking
Single braille character with one dot missing — the gap
orbits clockwise around the perimeter.

Usage:
    spinner = GridSpinner("thinking")
    spinner.start()
    # ... do work ...
    spinner.stop()
"""

from __future__ import annotations

import sys
import threading

# ANSI
_PURPLE = "\033[38;2;216;180;254m"  # bright visible purple
_RESET = "\033[0m"
_WHITE = "\033[37m"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE = "\033[2K"

# Braille dot layout:      Bit index:
#   1  4                    0  3
#   2  5                    1  4
#   3  6                    2  5
#   7  8                    6  7
#
# Gap orbits clockwise
_FULL = 0xFF
_ORBIT_BITS = [0, 3, 4, 5, 7, 6, 2, 1]
_FRAMES = [chr(0x2800 | (_FULL ^ (1 << b))) for b in _ORBIT_BITS]


class GridSpinner:
    """Single-line braille spinner with prompt prefix."""

    def __init__(self, label: str = "thinking", interval: float = 0.08):
        self.label = label
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame = 0

    def _animate(self):
        sys.stdout.write(_HIDE_CURSOR)
        sys.stdout.flush()

        while not self._stop_event.is_set():
            char = _FRAMES[self._frame % len(_FRAMES)]
            line = (
                f"\r{_CLEAR_LINE}"
                f"{_PURPLE}glyphh{_WHITE}> "
                f"{_PURPLE}{char}{_RESET}"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
            self._frame += 1
            self._stop_event.wait(self.interval)

    def start(self):
        self._stop_event.clear()
        self._frame = 0
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        sys.stdout.write(f"\r{_CLEAR_LINE}")
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()
