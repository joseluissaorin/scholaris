#!/usr/bin/env python3
"""Scholaris CLI - Academic citation automation from the command line.

Usage:
    scholaris cite <document> <spdf_dir> [options]
    scholaris process <pdf> [options]
    scholaris info <spdf_dir>
    scholaris install-skills [--global]
    scholaris --version
    scholaris --help

Commands:
    cite            Auto-cite a document using SPDF bibliography
    process         Process a PDF to SPDF format
    info            Show information about SPDF files in a directory
    install-skills  Install Claude Code skills and commands

Examples:
    # Cite a document using pre-processed SPDF files
    scholaris cite paper.md ./spdf -o paper_cited.md

    # Process a PDF to SPDF format
    scholaris process paper.pdf --key smith2024 --authors "John Smith" --year 2024

    # Show info about SPDF collection
    scholaris info ./spdf

    # Install Claude Code skills globally
    scholaris install-skills --global
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def get_version():
    """Get package version."""
    try:
        from scholaris import __version__
        return __version__
    except ImportError:
        return "0.1.0"


def cmd_cite(args):
    """Auto-cite a document using SPDF bibliography."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Use --api-key or set environment variable.")
        return 1

    from scholaris.auto_cite.citation_index import CitationIndex
    from scholaris.auto_cite.models import CitationStyle

    input_path = Path(args.document)
    spdf_dir = Path(args.spdf_dir)

    if not input_path.exists():
        print(f"Error: Document not found: {input_path}")
        return 1

    if not spdf_dir.exists():
        print(f"Error: SPDF directory not found: {spdf_dir}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_cited{input_path.suffix}"

    print("=" * 60)
    print("Scholaris Auto-Citation")
    print("=" * 60)

    # Load SPDF bibliography using the key approach: add_directory()
    print(f"\nLoading SPDF bibliography from: {spdf_dir}")
    index = CitationIndex(gemini_api_key=api_key)
    count = index.add_directory(spdf_dir)
    print(f"Loaded {count} sources ({index.total_chunks} chunks)")

    if count == 0:
        print("\nWarning: No SPDF files found. Process PDFs first with 'scholaris process'")
        return 1

    # Show sources
    if args.verbose:
        print("\nSources:")
        for source in index.sources():
            print(f"  - {source['citation_key']}: {source['title'][:50]}...")

    # Read document
    print(f"\nReading: {input_path.name}")
    document_text = input_path.read_text(encoding='utf-8')
    print(f"  {len(document_text)} chars, {document_text.count(chr(10))} lines")

    # Select citation style
    style = CitationStyle.CHICAGO17 if args.style == "chicago" else CitationStyle.APA7

    # Generate citations
    print(f"\nGenerating citations ({style.value})...")
    result = index.cite_document(
        document_text=document_text,
        style=style,
        min_confidence=args.confidence,
        max_citations_per_claim=args.max_citations,
        include_bibliography=not args.no_bibliography,
    )

    # Save output
    output_path.write_text(result.modified_document, encoding='utf-8')

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"  Citations inserted: {result.metadata['total_citations']}")
    print(f"  Unique sources: {result.metadata['unique_sources']}")
    print(f"  Output: {output_path}")

    if result.warnings and args.verbose:
        print(f"\nWarnings:")
        for w in result.warnings[:5]:
            print(f"  - {w}")

    print("=" * 60)
    return 0


def cmd_process(args):
    """Process a PDF to SPDF format."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Use --api-key or set environment variable.")
        return 1

    from scholaris.auto_cite.processed_pdf import ProcessedPDF

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent
        output_path = output_dir / f"{args.key or pdf_path.stem}.spdf"

    # Parse authors
    authors = []
    if args.authors:
        authors = [a.strip() for a in args.authors.split(",")]

    print("=" * 60)
    print("Scholaris PDF Processing")
    print("=" * 60)
    print(f"\nInput: {pdf_path}")
    print(f"Output: {output_path}")
    print(f"Citation key: {args.key or pdf_path.stem}")
    if authors:
        print(f"Authors: {', '.join(authors)}")
    if args.year:
        print(f"Year: {args.year}")
    if args.title:
        print(f"Title: {args.title}")

    print("\nProcessing with Vision OCR...")

    processed = ProcessedPDF.from_pdf(
        pdf_path=pdf_path,
        citation_key=args.key or pdf_path.stem,
        authors=authors,
        year=args.year or 0,
        title=args.title or pdf_path.stem,
        gemini_api_key=api_key,
        include_previews=not args.no_previews,
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(output_path)

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"  Pages: {processed.metadata.total_pages}")
    print(f"  Chunks: {processed.metadata.total_chunks}")
    print(f"  Output: {output_path}")
    print("=" * 60)
    return 0


def cmd_info(args):
    """Show information about SPDF files in a directory."""
    from scholaris.auto_cite.processed_pdf import ProcessedPDF

    spdf_dir = Path(args.spdf_dir)
    if not spdf_dir.exists():
        print(f"Error: Directory not found: {spdf_dir}")
        return 1

    spdf_files = sorted(spdf_dir.glob("*.spdf"))
    if not spdf_files:
        print(f"No SPDF files found in: {spdf_dir}")
        return 1

    print("=" * 60)
    print(f"SPDF Collection: {spdf_dir}")
    print("=" * 60)

    total_pages = 0
    total_chunks = 0
    total_size = 0

    print(f"\n{'Citation Key':<25} {'Year':<6} {'Pages':<7} {'Chunks':<8} {'Size':<10}")
    print("-" * 60)

    for spdf_file in spdf_files:
        try:
            info = ProcessedPDF.info(spdf_file)
            size_mb = info['size_mb']
            total_size += size_mb
            total_pages += info['pages']
            total_chunks += info['chunks']

            print(f"{info['citation_key']:<25} {info['year']:<6} {info['pages']:<7} {info['chunks']:<8} {size_mb:.1f} MB")
        except Exception as e:
            print(f"{spdf_file.stem:<25} ERROR: {str(e)[:30]}")

    print("-" * 60)
    print(f"{'TOTAL':<25} {'':<6} {total_pages:<7} {total_chunks:<8} {total_size:.1f} MB")
    print(f"\n{len(spdf_files)} sources")
    return 0


def cmd_install_skills(args):
    """Install Claude Code skills and commands."""
    # Find the scholaris package directory
    import scholaris
    package_dir = Path(scholaris.__file__).parent.parent
    claude_source = package_dir / ".claude"

    if not claude_source.exists():
        # Try relative to this file
        claude_source = Path(__file__).parent.parent / ".claude"

    if not claude_source.exists():
        print(f"Error: Could not find .claude directory in scholaris package")
        print(f"Searched: {package_dir / '.claude'}")
        return 1

    # Determine target directory
    if args.global_install:
        target_dir = Path.home() / ".claude"
    else:
        target_dir = Path.cwd() / ".claude"

    print("=" * 60)
    print("Installing Scholaris Claude Code Skills")
    print("=" * 60)
    print(f"\nSource: {claude_source}")
    print(f"Target: {target_dir}")

    # Create directories
    skills_target = target_dir / "skills" / "scholaris"
    commands_target = target_dir / "commands"

    skills_target.mkdir(parents=True, exist_ok=True)
    commands_target.mkdir(parents=True, exist_ok=True)

    # Copy skills
    skills_source = claude_source / "skills" / "scholaris"
    if skills_source.exists():
        print("\nCopying skills...")
        for item in skills_source.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(skills_source)
                dest = skills_target / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
                print(f"  {rel_path}")

    # Copy commands
    commands_source = claude_source / "commands"
    if commands_source.exists():
        print("\nCopying commands...")
        for item in commands_source.glob("*.md"):
            dest = commands_target / item.name
            shutil.copy2(item, dest)
            print(f"  {item.name}")

    # Copy INSTALL.md
    install_source = claude_source / "INSTALL.md"
    if install_source.exists():
        shutil.copy2(install_source, target_dir / "INSTALL.md")
        print(f"  INSTALL.md")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"\nInstalled to: {target_dir}")
    print("\nAvailable commands:")
    print("  /cite - Cite a document")
    print("  /process-pdf - Process PDF to SPDF")
    print("  /search-papers - Search for academic papers")
    print("  /batch-process - Batch process PDFs")
    print("\nSkill 'scholaris' will auto-activate for citation tasks.")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="scholaris",
        description="Scholaris - AI-powered academic citation automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scholaris cite paper.md ./spdf -o paper_cited.md
  scholaris process paper.pdf --key smith2024 --authors "John Smith" --year 2024
  scholaris info ./spdf
  scholaris install-skills --global
        """
    )
    parser.add_argument("--version", action="version", version=f"scholaris {get_version()}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # cite command
    cite_parser = subparsers.add_parser(
        "cite",
        help="Auto-cite a document using SPDF bibliography",
        description="Insert in-text citations with verified page numbers into a document."
    )
    cite_parser.add_argument("document", help="Input document (markdown, txt)")
    cite_parser.add_argument("spdf_dir", help="Directory containing SPDF files")
    cite_parser.add_argument("-o", "--output", help="Output file path")
    cite_parser.add_argument("--style", choices=["apa", "chicago"], default="apa",
                           help="Citation style (default: apa)")
    cite_parser.add_argument("--confidence", type=float, default=0.5,
                           help="Minimum confidence threshold (default: 0.5)")
    cite_parser.add_argument("--max-citations", type=int, default=2,
                           help="Max citations per claim (default: 2)")
    cite_parser.add_argument("--no-bibliography", action="store_true",
                           help="Don't append bibliography at end")
    cite_parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY)")
    cite_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a PDF to SPDF format",
        description="Convert a PDF to SPDF format using Vision OCR."
    )
    process_parser.add_argument("pdf", help="Input PDF file")
    process_parser.add_argument("-o", "--output", help="Output SPDF file path")
    process_parser.add_argument("--output-dir", help="Output directory for SPDF file")
    process_parser.add_argument("--key", help="Citation key (default: PDF filename)")
    process_parser.add_argument("--authors", help="Authors (comma-separated)")
    process_parser.add_argument("--year", type=int, help="Publication year")
    process_parser.add_argument("--title", help="Document title")
    process_parser.add_argument("--no-previews", action="store_true",
                              help="Don't include page preview images")
    process_parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY)")

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about SPDF files",
        description="Display statistics about SPDF files in a directory."
    )
    info_parser.add_argument("spdf_dir", help="Directory containing SPDF files")

    # install-skills command
    install_parser = subparsers.add_parser(
        "install-skills",
        help="Install Claude Code skills and commands",
        description="Install scholaris skills and slash commands for Claude Code."
    )
    install_parser.add_argument("--global", dest="global_install", action="store_true",
                               help="Install to ~/.claude (global) instead of ./.claude (project)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "cite": cmd_cite,
        "process": cmd_process,
        "info": cmd_info,
        "install-skills": cmd_install_skills,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
