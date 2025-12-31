"""Scholaris - Academic Research Automation Library.

A Python library for academic research automation including:
- Bibliography search and download
- BibTeX generation
- Literature review generation
- Academic writing assistance
"""

from .config import Config
from .exceptions import (
    ScholarisError,
    ConfigurationError,
    SearchError,
    DownloadError,
    BibTeXError,
    LLMError,
    RateLimitError,
    ConversionError,
    ValidationError,
)
from .core.models import Paper, Reference, Section, Review
from .scholaris import Scholaris

__version__ = "0.1.0"
__all__ = [
    "Scholaris",
    "Config",
    "Paper",
    "Reference",
    "Section",
    "Review",
    "ScholarisError",
    "ConfigurationError",
    "SearchError",
    "DownloadError",
    "BibTeXError",
    "LLMError",
    "RateLimitError",
    "ConversionError",
    "ValidationError",
]
