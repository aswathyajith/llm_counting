### Generate examples for N-shot context counting experiments

import pandas as pd
import os
import argparse


def process_context_ablation_data(
    examples, context_ablation, target_type, cipher_swap_rate=0.0
):
    if context_ablation == "cipher":
        examples["content"] = examples[
            f"cipher_ablation_rate_{cipher_swap_rate}_target_{target_type}"
        ]
        examples["target_string"] = examples["cipher"]
    elif context_ablation == "reverse":
        examples["content"] = examples[
            f"{context_ablation}_ablation_target_{target_type}"
        ]
        examples["target_string"] = examples[context_ablation]
    # else:
    #     examples["content"] = examples[f"{context_ablation}_ablation"]
    #     examples["target_string"] = examples[context_ablation]
    return examples


def generate_examples(
    data_file,
    target_nouns,
    examples_dir,
    context_ablation,
    cipher_swap_rates,
    target_type,
):
    # Load dataset
    dataset = pd.read_json(data_file, lines=True, orient="records")
    examples = dataset[dataset.noun.isin(target_nouns)]
    examples = examples.reset_index(drop=True)
    cols = [
        "target_string",
        "num_matches",
        "disambiguation",
        "title",
        "content",
        "matches",
        "case0_space0_standalone0",
        "case1_space0_standalone0",
        "case0_space0_standalone1",
        "case1_space0_standalone1",
        "case0_space1_standalone0",
        "case1_space1_standalone0",
        "case0_space1_standalone1",
        "case1_space1_standalone1",
    ]

    if context_ablation in ["cipher", "reverse", "shuffle"]:
        examples_subdir = os.path.join(examples_dir, context_ablation)
        if context_ablation == "cipher":
            for cipher_swap_rate in cipher_swap_rates:
                examples = process_context_ablation_data(
                    examples, context_ablation, target_type, cipher_swap_rate
                )

                examples_file = os.path.join(
                    examples_subdir,
                    f"cipher_rate_{cipher_swap_rate}",
                    f"target_{target_type}",
                    "example_contexts.jsonl",
                )
                examples["target_string"] = examples[target_type]
                examples_df = examples[cols]
                os.makedirs(os.path.dirname(examples_file), exist_ok=True)
                examples_df.to_json(examples_file, orient="records", lines=True)

        else:
            examples = process_context_ablation_data(examples, context_ablation)
            examples_file = os.path.join(
                examples_subdir, target_type, "example_contexts.jsonl"
            )
            examples_df = examples[cols]
            os.makedirs(os.path.dirname(examples_file), exist_ok=True)
            examples_df.to_json(examples_file, orient="records", lines=True)
    else:
        examples["content"] = examples["noun_context"]

        for target_type in ["noun", "nonce", "hash", "cipher"]:
            examples_file = os.path.join(
                examples_dir, target_type, "example_wiki_contexts.jsonl"
            )
            examples["target_string"] = examples[target_type]
            examples["content"] = examples[f"{target_type}_context"]

            examples_df = examples[cols]
            os.makedirs(os.path.dirname(examples_file), exist_ok=True)
            examples_df.to_json(examples_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate examples for N-shot context counting experiments"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/contexts/wiki/contexts.jsonl",
        help="Path to the dataset file to generate examples from.",
    )
    parser.add_argument(
        "--target_nouns",
        type=str,
        default=["byway", "scotch", "thymidine"],
        help="Nouns corresponding to the examples in the dataset file.",
        nargs="+",
    )
    parser.add_argument(
        "--examples_dir",
        type=str,
        default="data/prompts/context_counting/examples/target_ablations",
        help="Directory with the n-shot examples for the experiment.",
    )
    parser.add_argument(
        "--context_ablation",
        type=str,
        default="none",
        help="Whether to use ablated context. Options: 'cipher', 'reverse', 'shuffle', 'none' [default]",
    )

    parser.add_argument(
        "--target_type",
        type=str,
        default="noun",
        help="Target entity type (Options: 'noun' [default], 'cipher', 'reverse')",
    )

    parser.add_argument(
        "--cipher_swap_rates",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        help="Swap rates for cipher ablation (space-separated list)",
        nargs="+",
    )
    args = parser.parse_args()

    data_file = args.data_file
    target_nouns = args.target_nouns
    examples_dir = args.examples_dir
    context_ablation = args.context_ablation
    cipher_swap_rates = (
        args.cipher_swap_rates if args.context_ablation == "cipher" else None
    )
    target_type = args.target_type
    generate_examples(
        data_file,
        target_nouns,
        examples_dir,
        context_ablation,
        cipher_swap_rates,
        target_type,
    )
