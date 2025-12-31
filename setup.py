"""Setup configuration for Scholaris."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scholaris",
    version="1.0.0",
    author="José Luis Saorín Ferrer",
    author_email="jlsaorin@users.noreply.github.com",
    description="A Python library for academic research automation with AI-powered literature review generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joseluissaorin/scholaris",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        # Academic tools (Phase 1 & 2)
        "bibtexparser>=1.4.0",
        "PyPaperBot>=1.2.0",
        "selenium>=4.0.0",
        "undetected-chromedriver>=3.5.0",
        # LLM providers (Phase 3)
        "google-generativeai>=0.3.0",
        "google-api-python-client>=2.0.0",
        # Document processing (Phase 4)
        "python-docx>=0.8.11",
        "markdown>=3.4.0",
        "beautifulsoup4>=4.11.0",
    ],
    extras_require={
        "pdf": [
            # Optional: Better BibTeX extraction from PDFs
            "pdf2bib>=1.7",
        ],
        "dev": [
            # Development dependencies
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
            "flake8>=5.0.0",
        ],
        "all": [
            # All optional dependencies
            "pdf2bib>=1.7",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
            "flake8>=5.0.0",
        ],
    },
)
