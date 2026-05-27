### Generate examples for N-shot context counting experiments

import pandas as pd
import os
import argparse


def generate_examples(data_file, target_nouns, examples_dir, context_ablation):
    # Load dataset
    dataset = pd.read_json(data_file, lines=True, orient="records")
    examples = dataset[dataset.noun.isin(target_nouns)]
    examples = examples.reset_index(drop=True)

    if context_ablation in ["cipher"]:
        examples["content"] = examples["ablated_cipher_context"]
        examples["target_string"] = examples["cipher"]
        examples_file = os.path.join(examples_dir, "cipher_ablated_examples.jsonl")
    else:
        examples["content"] = examples["noun_context"]

        for target_type in ["noun", "nonce", "hash", "cipher"]:
            examples_file = os.path.join(
                examples_dir, target_type, "example_wiki_contexts.jsonl"
            )
            examples["target_string"] = examples[target_type]
            examples["content"] = examples[f"{target_type}_context"]
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
        default="data/contexts/wiki/contexts_wiki_cipher_ablated.jsonl",
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
        default="data/prompts/context_counting/examples",
        help="Directory with the n-shot examples for the experiment. The directory also contains the examples for the prompt (under <target_type>/<target_string>.txt).",
    )
    parser.add_argument(
        "--context_ablation",
        type=str,
        default="none",
        help="Whether to use ablated context. Options: 'cipher', 'none'",
    )
    args = parser.parse_args()

    data_file = args.data_file
    target_nouns = args.target_nouns
    examples_dir = args.examples_dir
    context_ablation = args.context_ablation
    generate_examples(data_file, target_nouns, examples_dir, context_ablation)
