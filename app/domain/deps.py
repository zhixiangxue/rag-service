"""Dependency rule utilities and constants.

Centralized logic for dependency rule formatting, parsing, and validation
to ensure consistency across the codebase.
"""
from __future__ import annotations
from dataclasses import dataclass


class DependencySource:
    """Valid protocol prefixes for dependency rules."""

    DOC = "doc"
    LENDER = "lender"
    OVERLAY = "overlay"

    VALID_PREFIXES = {DOC, LENDER, OVERLAY}


@dataclass
class Rule:
    """Represents a parsed dependency rule with protocol and value.

    A rule is stored as a "protocol:value" string, e.g.:
      doc:abc123
      lender:Wells Fargo
      overlay:FreddieMac
    """

    protocol: str
    value: str

    def __str__(self) -> str:
        return f"{self.protocol}:{self.value}"

    @staticmethod
    def build(protocol: str, value: str) -> Rule:
        """Build a Rule from protocol and value.

        Args:
            protocol: One of DependencySource.VALID_PREFIXES
            value: The rule value (doc_id, lender name, overlay name)

        Returns:
            Rule instance

        Raises:
            ValueError: If protocol is not recognized
        """
        if protocol not in DependencySource.VALID_PREFIXES:
            raise ValueError(
                f"Invalid protocol '{protocol}'. "
                f"Must be one of: {sorted(DependencySource.VALID_PREFIXES)}"
            )
        return Rule(protocol=protocol, value=value)

    @staticmethod
    def parse(raw: str) -> Rule:
        """Parse a raw rule string into a Rule object.

        Args:
            raw: Rule string like "doc:abc123" or "lender:Wells Fargo"

        Returns:
            Rule instance

        Raises:
            ValueError: If format is invalid or protocol is not recognized
        """
        if ":" not in raw:
            raise ValueError(
                f"Invalid rule format, expected 'protocol:value': {raw}"
            )
        protocol, value = raw.split(":", 1)
        if protocol not in DependencySource.VALID_PREFIXES:
            raise ValueError(
                f"Invalid protocol '{protocol}'. "
                f"Must be one of: {sorted(DependencySource.VALID_PREFIXES)}"
            )
        return Rule(protocol=protocol, value=value)
