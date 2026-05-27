#!/usr/bin/env python3
"""Run tokenization analysis with leading space characters."""

from analyze_tokenization import TokenizationAnalyzer

if __name__ == "__main__":
    print("Starting tokenization analysis with leading space characters...")

    analyzer = TokenizationAnalyzer()

    if not analyzer.tokenizers:
        print("No tokenizers available. Install tiktoken and/or transformers:")
        print("pip install tiktoken transformers")
        exit(1)

    # Analyze datasets with leading spaces
    comparison = analyzer.compare_entity_types(sample_size=15, add_leading_space=True)

    # Print report
    analyzer.print_comparison_report(comparison)

    # Save detailed analysis
    analyzer.save_report(
        comparison,
        output_path="data/tokenization_analysis_with_space.json",
        add_leading_space=True,
    )

    print(
        "\nAnalysis complete! Check data/tokenization_analysis_with_space.json for detailed results."
    )
