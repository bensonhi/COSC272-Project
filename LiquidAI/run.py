"""Script entry point: train then test by default; --mode train/test runs one phase."""

if __package__:
    from .cli import main
else:
    # Allow an absolute script path or execution from inside LiquidAI/ while
    # keeping all functional modules under the same package namespace.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from LiquidAI.cli import main


if __name__ == "__main__":
    main()
