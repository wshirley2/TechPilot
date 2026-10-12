"""Output formatting helpers for the benchmark CLI."""


def export_text(items: list[str]) -> str:
    """Return the original text representation of exported items."""
    return "\n".join(items)
