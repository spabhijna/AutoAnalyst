"""Thin wrapper kept for old habits: ``python run_demo.py`` is ``python main.py``."""

from main import main

if __name__ == "__main__":  # guard required: the sandbox re-imports __main__ in children
    main()
