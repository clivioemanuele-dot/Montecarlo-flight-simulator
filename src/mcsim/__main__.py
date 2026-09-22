"""Permette ``python -m mcsim``. La guardia è obbligatoria con il contesto ``spawn`` (Windows)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
