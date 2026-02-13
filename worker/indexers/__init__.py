"""Indexers for different ingestion modes."""

from .classic import index_classic
from .lod import index_lod
from .graph import index_graph

__all__ = [
    "index_classic",
    "index_lod",
    "index_graph",
]
