"""Measure mountain bike suspension sag and rebound from a video."""

from __future__ import annotations

import sys


def main() -> None:
    from .cli import main as cli_main

    sys.exit(cli_main())


__all__ = ["main"]
