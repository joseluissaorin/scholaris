"""SPDF Reference Implementation - Minimal reader/writer for the .spdf format.

This module provides a minimal, dependency-light implementation of the SPDF
format that can be used as a reference for porting to other languages.

The only dependencies are:
- Python 3.9+ standard library (gzip, sqlite3, json, tempfile)
- numpy (for embedding vectors only)
"""

from .reader import SPDFReader, read_spdf
from .writer import SPDFWriter, write_spdf

__all__ = ["SPDFReader", "SPDFWriter", "read_spdf", "write_spdf"]
