"""PyInstaller entry point for the engine sidecar (meeting-engine)."""
import sys

from engine.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
