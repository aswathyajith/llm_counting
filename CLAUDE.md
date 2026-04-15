# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This project studies counting behavior in LLMs by investigating systematic failure modes using random hash strings that have minimal presence in training corpora. The experiments test statistical significance of counting errors across different settings.

## Installation & Setup
```bash
pip install .
```

## Core Commands

### Data Generation
Generate synthetic data (random hashes) for experiments:
```bash
python src/generate_hashes.py --count 100 --length 32 --save_path data/hashes.c32.l100.jsonl
```

### Code Quality
- **Linting**: `ruff check src/` (configured via .pre-commit-config.yaml)
- **Formatting**: `ruff format src/`
- **Pre-commit hooks**: Only apply to files in `src/` directory, uses ruff for linting and formatting

## Architecture

### Core Modules
- **`generate_hashes.py`**: Creates cryptographically random hex strings using `secrets.token_hex()`. Outputs JSONL format with target_string, length, and count fields.
- **`prompt_builder.py`**: Builds prompts from templates using Python's `string.Template`. Takes template files and data files to generate prompt datasets.
- **`openrouter_calls.py`**: OpenAI-compatible API client for OpenRouter. Supports config files (JSON/TOML), environment variables, and CLI usage.

### Data Flow
1. Generate random hashes → `data/hashes.*.jsonl`
2. Build prompts from templates → `data/prompt_dataset.jsonl`
3. Make API calls via OpenRouter → Results for analysis

### Configuration
- **Templates**: `prompts/templates.json` - Contains system_prompt and user_prompt with `${target_string}` placeholder
- **OpenRouter**: `configs/openrouter.json` - Model configuration for API calls
- **Environment**: Set `OPENROUTER_API_KEY` for API authentication

### Key Design Patterns
- JSONL format for all data files
- Template-based prompt generation with string substitution
- Configurable hash generation (length, count, save path)
- OpenRouter integration with config file support