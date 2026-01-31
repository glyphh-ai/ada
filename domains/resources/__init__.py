"""
Resource Management Domain.

Handles resource quota tracking and enforcement for multi-tenancy.
"""

from domains.resources.manager import ResourceManager

__all__ = ["ResourceManager"]
