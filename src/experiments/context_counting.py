"""Context Counting Experiment"""

import argparse
import pandas as pd

import os
import json

from src.openrouter_calls import chat_completion, load_openrouter_config
from src.prompt_builder import PromptBuilder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Counting Experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-5.2",
        help="OpenRouter model to prompt with",
    )
    parser.add_argument(
        "--template_file",
        type=str,
        default="data/prompts/context_counting/templates/prompt_1.txt",
        help="Template file for the prompt",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data/contexts/wiki/contexts_wiki.jsonl",
        help="Dataset file for the experiment",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default=["all"],
        help="Target entity types (noun, nonce, hash, cipher, all) for testing",
        nargs="+",
    )
    parser.add_argument(
        "--context_ablation",
        type=str,
        default="none",
        help="Whether to use ablated context. Options: 'cipher', 'none'",
    )
    parser.add_argument(
        "--examples_dir",
        type=str,
        default="data/prompts/context_counting/examples/orig_contexts",
        help="Directory with file containing the n-shot examples for the experiment (example_contexts.jsonl). The directory also contains the examples for the prompt (under <target_type>/<target_string>.txt).",
    )
    parser.add_argument(
        "--n_shot", type=int, default=3, help="Number of shots for the experiment"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/context_counting/wiki/orig_contexts/uncased_contexts",
        help="Output dir for the experiment",
    )
    parser.add_argument(
        "--responses_dir",
        type=str,
        default="model_responses/context_counting/wiki/orig_contexts/uncased_contexts",
        help="Dir to save the model responses to",
    )
    parser.add_argument(
        "--cased_contexts",
        action="store_true",
        help="Whether to use cased context strings. Default is False (i.e. context is converted to lowercase).",
        default=False,
    )
    parser.add_argument(
        "--openrouter_config",
        type=str,
        default=None,
        help="Optional JSON/TOML OpenRouter config file (e.g., temperature/reasoning/extra/api_key).",
    )

    args = parser.parse_args()
    target_types = args.target_type

    openrouter_cfg = {}
    if args.openrouter_config:
        openrouter_cfg = load_openrouter_config(args.openrouter_config)

    if ("all" in target_types) and (args.context_ablation == "none"):
        target_types = ["noun", "nonce", "hash", "cipher"]

    if args.context_ablation == "cipher":
        ablation_error = False
        for t in args.target_type:
            if t not in ["cipher", "noun"]:
                ablation_error = True
                break
        if ablation_error:
            raise ValueError(
                "Target type must be either cipher or noun when context ablation is cipher"
            )

    for target_type in target_types:
        results_dir = os.path.join(args.results_dir, args.model_name, target_type)
        os.makedirs(results_dir, exist_ok=True)
        responses_dir = os.path.join(args.responses_dir, args.model_name, target_type)
        os.makedirs(responses_dir, exist_ok=True)
        examples_file = os.path.join(
            args.examples_dir, target_type, "example_contexts.jsonl"
        )

        print(f"Running experiment for {target_type}...", flush=True)

        print(f"Reading dataset from {args.dataset_file}...", flush=True)
        dataset = pd.read_json(args.dataset_file, lines=True, orient="records")
        print(f"Reading examples from {examples_file}...", flush=True)
        examples = pd.read_json(examples_file, lines=True, orient="records")
        print(f"Examples: {examples.head()}", flush=True)

        results_file = os.path.join(results_dir, "results.jsonl")
        if os.path.exists(results_file):
            results_df = pd.read_json(results_file, lines=True, orient="records")
        else:
            results_df = pd.DataFrame(
                columns=[
                    "target_string",
                    "target_type",
                    "counting_context",
                    "case0_space0_standalone0",
                    "case1_space0_standalone0",
                    "case0_space0_standalone1",
                    "case1_space0_standalone1",
                    "case0_space1_standalone0",
                    "case1_space1_standalone0",
                    "case0_space1_standalone1",
                    "case1_space1_standalone1",
                    "model_response_id",
                    "model_response_text",
                ]
            )

        # Remove instances that are already in the examples file
        dataset = dataset[
            ~dataset[target_type].isin(examples["target_string"])
        ].reset_index(drop=True)

        # Remove instances with no context
        dataset = dataset[dataset[f"{target_type}_context"] != ""]

        print(dataset.head(), flush=True)
        print(examples.head(), flush=True)

        # Instantiate PromptBuilder
        prompt_builder = PromptBuilder(
            n_shot=args.n_shot,
            template_file=args.template_file,
            examples=examples,
            examples_dir=os.path.join(args.examples_dir, target_type),
        )

        for index, row in dataset.iterrows():
            target_string = row[target_type]

            def get_context_col(target_type, context_ablation):
                if context_ablation == "none":
                    return f"{target_type}_context"
                elif context_ablation == "cipher":
                    if target_type == "cipher":
                        return f"ablated_{context_ablation}_context"
                    elif target_type == "noun":
                        return f"ablated_{context_ablation}_context_tgt_noun"
                return None

            context_col = get_context_col(target_type, args.context_ablation)
            if context_col is None:
                raise ValueError(
                    f"Invalid context ablation / target type combination: {target_type} {args.context_ablation}"
                )
            text = row[context_col]
            if not args.cased_contexts:
                text = text.lower()
            case0_space0_standalone0 = row["case0_space0_standalone0"]
            case1_space0_standalone0 = row["case1_space0_standalone0"]
            case0_space0_standalone1 = row["case0_space0_standalone1"]
            case1_space0_standalone1 = row["case1_space0_standalone1"]
            case0_space1_standalone0 = row["case0_space1_standalone0"]
            case1_space1_standalone0 = row["case1_space1_standalone0"]
            case0_space1_standalone1 = row["case0_space1_standalone1"]
            case1_space1_standalone1 = row["case1_space1_standalone1"]

            prompt = prompt_builder.build_prompt(text=text, target_string=target_string)
            if index == 0:
                print(prompt, flush=True)

            # Prompt OpenRouter
            try:
                cfg_model = openrouter_cfg.get("model")
                cfg_temperature = openrouter_cfg.get("temperature")
                cfg_max_tokens = openrouter_cfg.get("max_tokens")
                cfg_reasoning = openrouter_cfg.get("reasoning")
                cfg_extra = openrouter_cfg.get("extra")
                cfg_api_key = openrouter_cfg.get("api_key")

                response = chat_completion(
                    model=cfg_model or args.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=cfg_api_key or os.environ.get("OPENROUTER_API_KEY"),
                    temperature=cfg_temperature,
                    max_tokens=cfg_max_tokens,
                    reasoning=cfg_reasoning,
                    extra=cfg_extra,
                )

                # save response to file (append to file if it already exists)
                # jsonl format
                with open(
                    os.path.join(responses_dir, f"{target_string}.jsonl"), "a"
                ) as f:
                    f.write(json.dumps(response) + "\n")

                model_response_id = response["id"]
                model_response_text = response["choices"][0]["message"]["content"]

                new_row = {
                    "target_string": target_string,
                    "target_type": target_type,
                    "counting_context": text,
                    "case0_space0_standalone0": case0_space0_standalone0,
                    "case1_space0_standalone0": case1_space0_standalone0,
                    "case0_space0_standalone1": case0_space0_standalone1,
                    "case1_space0_standalone1": case1_space0_standalone1,
                    "case0_space1_standalone0": case0_space1_standalone0,
                    "case1_space1_standalone0": case1_space1_standalone0,
                    "case0_space1_standalone1": case0_space1_standalone1,
                    "case1_space1_standalone1": case1_space1_standalone1,
                    "model_response_id": model_response_id,
                    "model_response_text": model_response_text,
                }
                results_df = pd.concat(
                    [results_df, pd.DataFrame([new_row])], ignore_index=True
                )
            except Exception as e:
                print(f"Error prompting model: {e}", flush=True)

            # save every 10 instances
            if index % 5 == 0:
                print(f"Saving results to {results_file}...", flush=True)
                results_df.to_json(results_file, orient="records", lines=True)

        print(results_df.head(), flush=True)
