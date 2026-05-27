"""Analyze tokenization patterns across different LLM tokenizers for counting experiments."""

import json
import os
from typing import List, Dict, Any
from collections import defaultdict
import statistics

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TokenizationAnalyzer:
    """Analyze how different tokenizers split entities for counting experiments."""

    def __init__(self):
        self.tokenizers = {}
        self._load_tokenizers()

    def _load_tokenizers(self):
        """Load available tokenizers."""
        # GPT tokenizers (OpenAI)
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizers["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
                self.tokenizers["gpt-3.5"] = tiktoken.encoding_for_model(
                    "gpt-3.5-turbo"
                )
                print("✅ Loaded OpenAI GPT tokenizers (accurate)")
            except Exception as e:
                print(f"Warning: Could not load GPT tokenizers: {e}")

        # Claude tokenizer (Anthropic - approximated with Claude-specific approach)
        if TRANSFORMERS_AVAILABLE:
            try:
                # Claude uses a similar but different tokenizer - approximate with updated model
                self.tokenizers["claude"] = AutoTokenizer.from_pretrained(
                    "microsoft/DialoGPT-medium"
                )
                print("✅ Loaded Claude approximation tokenizer (DialoGPT-based)")
            except Exception as e:
                print(f"Warning: Could not load Claude approximation: {e}")
                # Fallback to GPT-4 with clear warning
                if TIKTOKEN_AVAILABLE:
                    self.tokenizers["claude"] = tiktoken.encoding_for_model("gpt-4")
                    print("⚠️  Using GPT-4 tokenizer as Claude fallback")

        # Gemini Pro tokenizer (Google - approximation using T5-small)
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use T5-small as approximation for Google's tokenization approach
                self.tokenizers["gemini-pro"] = AutoTokenizer.from_pretrained(
                    "t5-small"
                )
                print("✅ Loaded Gemini Pro approximation tokenizer (T5-small)")
            except Exception as e:
                try:
                    # Fallback to BERT-based tokenizer
                    self.tokenizers["gemini-pro"] = AutoTokenizer.from_pretrained(
                        "bert-base-uncased"
                    )
                    print("✅ Loaded Gemini Pro approximation tokenizer (BERT-based)")
                except Exception as e2:
                    print(f"Warning: Could not load Gemini tokenizers: {e}, {e2}")

        print(f"\nTokenizers loaded: {list(self.tokenizers.keys())}")
        print("Note: For most accurate results, use actual model APIs for tokenization")

    def analyze_entity(
        self, entity: str, add_leading_space: bool = False
    ) -> Dict[str, Any]:
        """Analyze how a single entity tokenizes across models."""
        if add_leading_space:
            entity = " " + entity

        result = {"entity": entity, "char_length": len(entity), "tokenizations": {}}

        for model_name, tokenizer in self.tokenizers.items():
            try:
                if model_name in ["gpt-4", "gpt-3.5"]:
                    # OpenAI/tiktoken style (accurate)
                    tokens = tokenizer.encode(entity)
                    token_strings = [tokenizer.decode([token]) for token in tokens]

                elif model_name == "claude":
                    # Check if it's tiktoken fallback or transformers
                    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
                        if hasattr(tokenizer, "name") and "cl100k" in tokenizer.name:
                            # tiktoken fallback
                            tokens = tokenizer.encode(entity)
                            token_strings = [
                                tokenizer.decode([token]) for token in tokens
                            ]
                        else:
                            # Transformers tokenizer
                            tokens = tokenizer.encode(entity, add_special_tokens=False)
                            token_strings = [
                                tokenizer.decode([token], skip_special_tokens=True)
                                for token in tokens
                            ]
                    else:
                        # tiktoken fallback
                        tokens = tokenizer.encode(entity)
                        token_strings = [tokenizer.decode([token]) for token in tokens]

                elif model_name == "gemini-pro":
                    # Transformers style (T5-based approximation)
                    tokens = tokenizer.encode(entity, add_special_tokens=False)
                    token_strings = [
                        tokenizer.decode([token], skip_special_tokens=True)
                        for token in tokens
                    ]

                else:
                    # Generic transformers approach
                    tokens = tokenizer.encode(entity, add_special_tokens=False)
                    token_strings = [
                        tokenizer.decode([token], skip_special_tokens=True)
                        for token in tokens
                    ]

                # Clean up empty token strings
                token_strings = [t for t in token_strings if t.strip()]

                result["tokenizations"][model_name] = {
                    "token_count": len(token_strings),
                    "tokens": token_strings,
                    "compression_ratio": len(entity) / len(token_strings)
                    if len(token_strings) > 0
                    else 0,
                }
            except Exception as e:
                print(f"Error tokenizing '{entity}' with {model_name}: {e}")
                result["tokenizations"][model_name] = {
                    "token_count": -1,
                    "tokens": [],
                    "compression_ratio": 0,
                    "error": str(e),
                }

        return result

    def analyze_dataset(
        self, dataset_path: str, sample_size: int = 20, add_leading_space: bool = False
    ) -> List[Dict[str, Any]]:
        """Analyze tokenization patterns for a dataset."""
        entities = []

        try:
            with open(dataset_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    if line.strip():
                        data = json.loads(line)
                        entities.append(data["target_string"])
        except FileNotFoundError:
            print(f"Dataset not found: {dataset_path}")
            return []

        results = []
        for entity in entities:
            results.append(
                self.analyze_entity(entity, add_leading_space=add_leading_space)
            )

        return results

    def compare_entity_types(
        self,
        data_dir: str = "data",
        sample_size: int = 20,
        add_leading_space: bool = False,
    ) -> Dict[str, Any]:
        """Compare tokenization across all entity types."""
        entity_types = {
            "hash": f"{data_dir}/hashes.c100.l15.jsonl",
            "noun": f"{data_dir}/nouns.c100.l15.jsonl",
            "nonce": f"{data_dir}/nonce.c100.l15.jsonl",
            "cipher": f"{data_dir}/cipher.c100.l15.jsonl",
        }

        comparison = {}

        for entity_type, dataset_path in entity_types.items():
            print(
                f"\nAnalyzing {entity_type} entities{' with leading space' if add_leading_space else ''}..."
            )
            results = self.analyze_dataset(
                dataset_path, sample_size, add_leading_space=add_leading_space
            )

            if results:
                comparison[entity_type] = {
                    "results": results,
                    "stats": self._compute_stats(results),
                }
            else:
                print(f"No results for {entity_type}")

        return comparison

    def _compute_stats(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute tokenization statistics."""
        stats = defaultdict(lambda: defaultdict(list))

        for result in results:
            for model_name, tokenization in result["tokenizations"].items():
                if "error" not in tokenization:
                    stats[model_name]["token_counts"].append(
                        tokenization["token_count"]
                    )
                    stats[model_name]["compression_ratios"].append(
                        tokenization["compression_ratio"]
                    )

        # Convert to summary statistics
        summary = {}
        for model_name, model_stats in stats.items():
            summary[model_name] = {}
            for metric, values in model_stats.items():
                if values:
                    summary[model_name][metric] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                    }

        return summary

    def print_comparison_report(self, comparison: Dict[str, Any]):
        """Print a detailed comparison report."""
        print("\n" + "=" * 80)
        print("TOKENIZATION ANALYSIS REPORT")
        print("=" * 80)

        # Summary table by entity type
        print("\nTOKEN COUNT SUMMARY (Mean ± Std)")
        print("-" * 60)

        entity_types = list(comparison.keys())
        model_names = list(self.tokenizers.keys())

        # Header
        print(f"{'Entity Type':<12}", end="")
        for model in model_names:
            print(f"{model:<15}", end="")
        print()

        # Data rows
        for entity_type in entity_types:
            if entity_type in comparison:
                print(f"{entity_type:<12}", end="")
                stats = comparison[entity_type]["stats"]

                for model in model_names:
                    if model in stats and "token_counts" in stats[model]:
                        mean = stats[model]["token_counts"]["mean"]
                        std = stats[model]["token_counts"]["std"]
                        print(f"{mean:.1f}±{std:.1f}      ", end="")
                    else:
                        print(f"{'N/A':<15}", end="")
                print()

        # Detailed examples
        print("\nDETAILED EXAMPLES")
        print("-" * 60)

        for entity_type, data in comparison.items():
            print(f"\n{entity_type.upper()} EXAMPLES:")

            # Show first 5 examples
            for i, result in enumerate(data["results"][:5]):
                entity = result["entity"]
                print(f"  {i+1}. '{entity}' (chars: {result['char_length']})")

                for model_name in model_names:
                    if model_name in result["tokenizations"]:
                        tok = result["tokenizations"][model_name]
                        if "error" not in tok:
                            tokens_str = " | ".join(tok["tokens"])
                            print(
                                f"     {model_name:<10}: {tok['token_count']} tokens → {tokens_str}"
                            )
                        else:
                            print(f"     {model_name:<10}: ERROR - {tok['error']}")
                print()

    def save_report(
        self,
        comparison: Dict[str, Any],
        output_path: str = "data/tokenization_analysis.json",
        add_leading_space: bool = False,
    ):
        """Save detailed analysis to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare data for JSON serialization
        json_data = {
            "models_analyzed": list(self.tokenizers.keys()),
            "analysis_type": "with_leading_space" if add_leading_space else "standard",
            "entity_types": {},
            "summary": {},
        }

        for entity_type, data in comparison.items():
            json_data["entity_types"][entity_type] = {
                "examples": data["results"][:10],  # Save first 10 examples
                "statistics": data["stats"],
            }

        # Overall summary
        json_data["summary"]["findings"] = self._generate_findings(comparison)

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"\nDetailed analysis saved to: {output_path}")

    def _generate_findings(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """Generate key findings from the analysis."""
        findings = {}

        # Find entity type with highest/lowest token counts
        avg_tokens = {}
        for entity_type, data in comparison.items():
            if "gpt-4" in data["stats"]:
                avg_tokens[entity_type] = data["stats"]["gpt-4"]["token_counts"]["mean"]

        if avg_tokens:
            most_fragmented = max(avg_tokens, key=avg_tokens.get)
            least_fragmented = min(avg_tokens, key=avg_tokens.get)

            findings["most_fragmented_entity_type"] = most_fragmented
            findings["least_fragmented_entity_type"] = least_fragmented
            findings["fragmentation_insight"] = (
                f"{most_fragmented} entities are most fragmented (avg {avg_tokens[most_fragmented]:.1f} tokens), {least_fragmented} are least fragmented (avg {avg_tokens[least_fragmented]:.1f} tokens)"
            )

        return findings


if __name__ == "__main__":
    print("Starting tokenization analysis...")

    analyzer = TokenizationAnalyzer()

    if not analyzer.tokenizers:
        print("No tokenizers available. Install tiktoken and/or transformers:")
        print("pip install tiktoken transformers")
        exit(1)

    # Analyze datasets
    comparison = analyzer.compare_entity_types(sample_size=15)

    # Print report
    analyzer.print_comparison_report(comparison)

    # Save detailed analysis
    analyzer.save_report(comparison)

    print(
        "\nAnalysis complete! Check data/tokenization_analysis.json for detailed results."
    )
