"""
Braille dot spinner — matches the Studio's Spinner.tsx.

Single braille character with one dot missing; the gap orbits
clockwise around the perimeter.  80 ms per frame.

Usage:
    from ..spinner import GridSpinner

    with GridSpinner():
        result = expensive_call()
"""

import sys
import threading
import time

# Braille U+2800 block — 8-dot grid.
# FULL = all 8 dots lit (⣿).  Each frame turns off one dot.
_FULL = 0xFF
_ORBIT = [0, 3, 4, 5, 7, 6, 2, 1]
_FRAMES = [chr(0x2800 | (_FULL ^ (1 << b))) for b in _ORBIT]
_INTERVAL = 0.08  # 80 ms, matches Studio

# ANSI bright-magenta — matches theme.ACCENT in dark mode
_CLR = "\033[95m"
_RST = "\033[0m"


class GridSpinner:
    """Context-manager braille spinner for the terminal."""

    def __init__(self, prefix: str = "  glyphh> "):
        self._prefix = prefix
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        # Clear the spinner line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self):
        i = 0
        while self._running:
            frame = _FRAMES[i % len(_FRAMES)]
            sys.stdout.write(f"\r{self._prefix}{_CLR}{frame}{_RST} ")
            sys.stdout.flush()
            i += 1
            time.sleep(_INTERVAL)
