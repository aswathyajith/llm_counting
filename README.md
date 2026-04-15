# The Signature of Counting
This project investigates and characterizes counting behavior in LLMs across different string types.

## Overview
LLMs are known to have deficits in their string matching and counting abilities. This repo studies their failure modes systematically under different settings, using four distinct types of entities with minimal training corpus presence:

- **Hash Strings**: Random hexadecimal strings (meaningless)
- **Noun Strings**: WordNet common nouns (semantic content)
- **Nonce Words**: Pronounceable non-words generated via Markov chains (phonetic patterns)
- **Cipher Nouns**: Alphabet-substituted versions of real nouns (encrypted semantic content)

This multi-faceted approach allows systematic comparison of counting behavior across semantic, phonetic, and structural dimensions. 

## Hypothesis Testing
To study the nature of counting errors that arise in LLMs, we design our experimental framework to test for statistically significant differences in model counting errors. Each setting tests a different dimension of counting, and we would expect the model to produce errors that are fairly close to each other.

- Counting within the input space (how many) vs output space (Give-N)
- Counting homogeneous vs non-homeogenous lists
- Counting the target string within natural language contexts vs counting within random token contexts: requires access to model's vocabulary

## Project structure: 
```text
    ├── data 
    ├── notebooks
    ├── plots
    ├── README.md
    ├── results
    ├── slurm_jobs
    └── src
```

## Steps to Reproduce Experiments
1. Set up environment 
    ```
        $ pip install .
    ```

2. Generate entity datasets for experiments.

    **Individual entity type generation:**
    ```bash
    $ python src/generate_entities.py --type hash --count 100 --length 15 --seed 999
    $ python src/generate_entities.py --type noun --count 100 --length 15 --seed 999
    $ python src/generate_entities.py --type nonce --count 100 --length 15 --seed 999
    $ python src/generate_entities.py --type cipher --count 100 --length 15 --seed 999
    ```

    **Generate all 4 entity types with matching length distributions:**
    ```bash
    $ python src/generate_entities.py --generate-all --count 100 --length 15 --seed 999
    ```

    This creates standardized datasets:
    - `data/hashes.c100.l15.jsonl` - Random hex strings
    - `data/nouns.c100.l15.jsonl` - WordNet common nouns
    - `data/nonce.c100.l15.jsonl` - Pronounceable non-words
    - `data/cipher.c100.l15.jsonl` - Alphabet-substituted nouns

## Entity Generation System

### Entity Types

The system generates four distinct entity types designed to test different aspects of LLM counting behavior:

1. **Hash Strings** (`--type hash`)
   - Random hexadecimal strings with no semantic meaning
   - Distributed lengths (3-15 characters) for natural variation
   - Minimal training corpus presence

2. **Noun Strings** (`--type noun`)
   - Single-word common nouns from WordNet corpus
   - Filtered to exclude proper nouns, compounds, and multi-word expressions
   - Semantic content familiar to language models

3. **Nonce Words** (`--type nonce`)
   - Pronounceable non-words generated using Markov chain models
   - Trained on WordNet noun corpus for realistic phonetic patterns
   - Sound word-like but don't exist in English dictionaries
   - Comprehensive real-word filtering using full WordNet corpus (77k+ words)

4. **Cipher Nouns** (`--type cipher`)
   - Alphabet-substituted versions of real nouns
   - Each letter mapped to a different letter (e.g., a→q, b→m, c→j)
   - Preserves word structure while removing semantic meaning
   - Uses same source nouns as noun dataset for direct comparison

### Key Features

- **Length Distribution Matching**: All entity types use identical length distributions for fair comparison
- **Reproducible Generation**: Comprehensive seeding ensures consistent results across runs
- **Batch Processing**: `--generate-all` flag creates all 4 types simultaneously
- **Quality Filtering**: Advanced filtering removes real words, proper nouns, and compounds

### Implementation Highlights

**Recent Improvements (2025):**
- ✅ **Code Optimization**: Reduced codebase from 706 to 268 lines (62% reduction)
- ✅ **Cipher Noun System**: Added alphabet substitution cipher for encrypted semantic content
- ✅ **Batch Generation**: Single command generates all 4 entity types with matching distributions
- ✅ **Enhanced Filtering**: Comprehensive real-word detection using full WordNet corpus
- ✅ **Length Normalization**: Ensures comparable length distributions across all entity types

**Example Usage:**
```bash
# Generate 100 entities of each type with matching length distributions
python src/generate_entities.py --generate-all --count 100 --length 15 --seed 999

# Output statistics:
# Noun: min=3, max=15, mean=8.62
# Cipher: min=3, max=15, mean=8.62
# Hash: min=3, max=15, mean=8.62
# Nonce: min=3, max=15, mean=8.62
```

## Results / Examples 

## Citation
```bibtex
@article{countingerrors2026,
  title={...}
}