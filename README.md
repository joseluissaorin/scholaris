# Scholaris

A Python library for automating academic research workflows, from paper discovery to literature review generation.

## Overview

Scholaris streamlines the academic research process by integrating paper search, PDF acquisition, bibliography management, and AI-assisted writing into a unified workflow. Built for researchers, students, and developers working with academic literature.

**Key Capabilities:**
- Search academic databases (Google Scholar)
- Download papers from open-access sources
- Extract and manage BibTeX citations
- Generate literature reviews using AI
- Export to multiple formats (Markdown, Word, HTML)

## Installation

```bash
# Standard installation
pip install git+https://github.com/joseluissaorin/scholaris.git

# With optional PDF metadata extraction
pip install git+https://github.com/joseluissaorin/scholaris.git[pdf]

# Development installation
git clone https://github.com/joseluissaorin/scholaris.git
cd scholaris
pip install -e .[dev]
```

**Requirements:**
- Python 3.9 or higher
- Google Gemini API key (for AI-powered features)

## Quick Start

```python
from scholaris import Scholaris

# Initialize
scholar = Scholaris(gemini_api_key="your-api-key")

# Search for papers
papers = scholar.search_papers(
    topic="transformer neural networks",
    max_papers=10,
    min_year=2020
)

# Download PDFs
pdf_paths = scholar.download_papers(papers, output_dir="./papers")

# Generate BibTeX
bibtex = scholar.generate_bibtex(pdf_paths, method="auto")

# Export bibliography
scholar.export_bibtex(bibtex, "references.bib")
```

## Core Features

### Paper Search and Acquisition

Search Google Scholar and download papers automatically:

```python
# Topic-based search
papers = scholar.search_papers(
    topic="machine learning",
    max_papers=20,
    min_year=2020
)

# Download PDFs via Sci-Hub
paths = scholar.download_papers(papers, output_dir="./papers")
```

### Bibliography Management

Extract and format citations from PDFs:

```python
# Extract BibTeX from PDFs
bibtex_entries = scholar.generate_bibtex(
    pdf_paths=["paper1.pdf", "paper2.pdf"],
    method="auto"  # Uses pdf2bib, falls back to AI extraction
)

# Format as APA 7th edition
formatted = scholar.format_references(bibtex_entries, style="APA7")

# Export to .bib file
scholar.export_bibtex(bibtex_entries, "my_bibliography.bib")
```

### Literature Review Generation

Generate comprehensive literature reviews with AI:

```python
review = scholar.generate_review(
    topic="Neural Machine Translation",
    papers=papers,
    bibtex_entries=bibtex_entries,
    sections=["Introduction", "Methods", "Discussion", "Conclusion"],
    min_words_per_section=500,
    language="English"
)

# Access generated content
print(review.markdown)
print(f"Word count: {review.word_count}")
```

### Multi-Format Export

Export reviews to various formats:

```python
scholar.export_markdown(review, "review.md")
scholar.export_docx(review, "review.docx")
scholar.export_html(review, "review.html")
```

## Complete Workflow

End-to-end research automation in a single call:

```python
review = scholar.complete_workflow(
    topic="Climate Change Mitigation Strategies",
    auto_search=True,
    max_papers=15,
    min_year=2019,
    sections=["Introduction", "Current Approaches", "Future Directions"],
    output_format="docx",
    output_path="./literature_review.docx"
)
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key
SCHOLARIS_SEARCH_PROVIDER=pypaperbot
SCHOLARIS_MAX_PAPERS_PER_KEYWORD=10
SCHOLARIS_MIN_PUBLICATION_YEAR=2020
SCHOLARIS_SCIHUB_MIRROR=https://www.sci-hub.ru
```

Load configuration:

```python
from scholaris import Config

config = Config.from_env()
scholar = Scholaris(config=config)
```

### Programmatic Configuration

```python
config = Config(
    search_provider="pypaperbot",
    max_papers_per_keyword=15,
    min_publication_year=2018,
    scihub_mirror="https://www.sci-hub.ru",
    citation_style="APA7"
)

scholar = Scholaris(config=config)
```

## API Reference

### Core Classes

**Scholaris** - Main interface for all operations
- `search_papers()` - Search academic databases
- `download_papers()` - Download PDFs
- `generate_bibtex()` - Extract citations
- `generate_review()` - Create literature reviews
- `complete_workflow()` - End-to-end automation

**Config** - Configuration management
- `from_env()` - Load from environment
- `from_dict()` - Load from dictionary

**Paper** - Paper metadata model
**Reference** - Citation data model
**Review** - Generated review model

See [Wiki](https://github.com/joseluissaorin/scholaris/wiki) for full API documentation.

## Architecture

Scholaris uses a modular provider-based architecture:

```
scholaris/
├── core/              # Data models and business logic
├── providers/         # Pluggable service backends
│   ├── search/       # Paper search (PyPaperBot)
│   ├── llm/          # AI providers (Gemini, DeepSeek, Perplexity)
│   └── bibtex/       # Citation extractors
├── converters/        # Format converters
└── utils/            # Helpers and utilities
```

## Testing

Scholaris has been extensively tested:

- Unit tests: 75% code coverage
- Integration tests: All features verified with real services
- Cross-platform: Linux, macOS, Windows

Run tests:

```bash
pytest tests/
```

## Examples

### Web Service Integration

```python
from flask import Flask, request, jsonify
from scholaris import Scholaris

app = Flask(__name__)
scholar = Scholaris()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    papers = scholar.search_papers(
        topic=data['topic'],
        max_papers=data.get('max_papers', 10)
    )
    return jsonify([p.to_dict() for p in papers])
```

### Command-Line Tool

```python
import click
from scholaris import Scholaris

@click.command()
@click.option('--topic', required=True)
@click.option('--max-papers', default=10)
def search(topic, max_papers):
    scholar = Scholaris()
    papers = scholar.search_papers(topic, max_papers=max_papers)
    for paper in papers:
        click.echo(f"{paper.title} ({paper.year})")

if __name__ == '__main__':
    search()
```

See the [examples/](examples/) directory for more use cases.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - Copyright (c) 2026 José Luis Saorín Ferrer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

If you use Scholaris in your research, please cite:

```bibtex
@software{scholaris2026,
  title={Scholaris: Academic Research Automation Library for Python},
  author={Saor\'{i}n Ferrer, Jos\'{e} Luis},
  year={2026},
  url={https://github.com/joseluissaorin/scholaris},
  version={1.0.0}
}
```

## Links

- [Documentation](https://github.com/joseluissaorin/scholaris/wiki)
- [Issue Tracker](https://github.com/joseluissaorin/scholaris/issues)
- [Changelog](CHANGELOG.md)

## Acknowledgments

Scholaris builds upon excellent open-source tools:
- PyPaperBot - Google Scholar integration
- pdf2bib - PDF metadata extraction
- Google Gemini - AI language model
- python-docx - Word document generation
