"""Indexers for different ingestion modes."""

# Use lazy imports to avoid loading graph module when not needed
# (graph module depends on domain.mortgage.graph which requires running from rag-service dir)

def __getattr__(name: str):
    """Lazy import on access."""
    if name == "index_classic":
        from .classic import index_classic
        return index_classic
    elif name == "index_lod":
        from .lod import index_lod
        return index_lod
    elif name == "index_graph":
        from .graph import index_graph
        return index_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "index_classic",
    "index_lod",
    "index_graph",
]
