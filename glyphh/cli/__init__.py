"""
Glyphh CLI - Command Line Interface for model building, testing, and packaging

The CLI is organized into five categories:
- build: Model initialization and concept management
- test: Testing and validation commands
- package: Model packaging and deployment
- runtime: Runtime integration commands
- auth: Authentication and authorization
"""

from .main import cli

__all__ = ["cli"]
