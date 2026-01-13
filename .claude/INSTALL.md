# Installing Scholaris Claude Code Skill

This directory contains a Claude Code skill and slash commands for using Scholaris directly within Claude Code.

## What's Included

```
.claude/
├── INSTALL.md                    # This file
├── skills/
│   └── scholaris/
│       ├── SKILL.md              # Main skill (auto-discovered)
│       ├── REFERENCE.md          # Complete API documentation
│       ├── WORKFLOWS.md          # Common workflow patterns
│       └── scripts/
│           ├── cite_document.py  # CLI citation script
│           └── process_pdf.py    # CLI PDF processing script
└── commands/
    ├── cite.md                   # /cite command
    ├── process-pdf.md            # /process-pdf command
    ├── search-papers.md          # /search-papers command
    └── batch-process.md          # /batch-process command
```

## Installation

### Option 1: Automatic (Recommended)

Copy the skill and commands to your personal Claude configuration:

```bash
# From the scholaris repository root
cp -r .claude/skills/scholaris ~/.claude/skills/
cp .claude/commands/*.md ~/.claude/commands/
```

### Option 2: Project-Specific

Keep the `.claude` directory in your project. Claude Code will automatically discover skills and commands in the project's `.claude` folder.

### Option 3: Symlink (Development)

If you want to keep the skill synced with the repository:

```bash
ln -s "$(pwd)/.claude/skills/scholaris" ~/.claude/skills/scholaris
ln -s "$(pwd)/.claude/commands/cite.md" ~/.claude/commands/cite.md
ln -s "$(pwd)/.claude/commands/process-pdf.md" ~/.claude/commands/process-pdf.md
ln -s "$(pwd)/.claude/commands/search-papers.md" ~/.claude/commands/search-papers.md
ln -s "$(pwd)/.claude/commands/batch-process.md" ~/.claude/commands/batch-process.md
```

## Verification

After installation, verify the skill is available:

1. Start Claude Code in any project
2. Type `/help` to see available commands - you should see `/cite`, `/process-pdf`, etc.
3. Ask Claude "What skills are available?" - it should mention scholaris

## Usage

### Auto-Discovery (Skill)

Claude will automatically suggest using scholaris when you:
- Mention citations, bibliography, or academic writing
- Work with PDF or SPDF files
- Ask about reference management

### Slash Commands

```bash
# Cite a document
/cite paper.md ./bibliography

# Process a PDF to SPDF
/process-pdf paper.pdf smith2024 "John Smith" 2024 "Paper Title"

# Search for papers
/search-papers "transformer attention" 10

# Batch process PDFs
/batch-process ./pdfs ./spdf
```

## Requirements

- Claude Code CLI installed
- Python 3.9+
- Scholaris library installed: `pip install scholaris`
- `GEMINI_API_KEY` environment variable set

## Updating

To update the skill after pulling new changes:

```bash
# If using Option 1 (copy)
cp -r .claude/skills/scholaris ~/.claude/skills/
cp .claude/commands/*.md ~/.claude/commands/

# If using Option 3 (symlink)
# No action needed - symlinks auto-update
```

## Customization

Feel free to modify the skill files to match your workflow:

- **SKILL.md**: Change the description to improve auto-discovery
- **REFERENCE.md**: Add project-specific API notes
- **WORKFLOWS.md**: Add your own workflow patterns
- **commands/*.md**: Customize default arguments and behavior

## Troubleshooting

### Skill not discovered
- Ensure files are in `~/.claude/skills/scholaris/` or `.claude/skills/scholaris/`
- Check that `SKILL.md` has valid YAML frontmatter

### Commands not showing
- Ensure files are in `~/.claude/commands/` or `.claude/commands/`
- Restart Claude Code after adding commands

### API errors
- Verify `GEMINI_API_KEY` is set in your environment or `.env` file
- Check API quota and rate limits
