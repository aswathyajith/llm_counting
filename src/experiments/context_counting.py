"""Context Counting Experiment"""

import argparse
import pandas as pd

import os
import json

from src.openrouter_calls import (
    chat_completion as openrouter_chat_completion,
    load_openrouter_config,
)
from src.argo_calls import chat_completion as argo_chat_completion, load_argo_config
from src.prompt_builder import PromptBuilder

DEFAULT_MODEL_NAMES = {
    "openrouter": "openai/gpt-5.2",
    "argo": "argo:gpt-5.2",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Counting Experiment")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openrouter", "argo"],
        default="openrouter",
        help="Which API backend to prompt the model through.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=argparse.SUPPRESS,
        help="Model to prompt with (id format depends on --backend). Overrides 'model' in --backend_config if both are given.",
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
        help="Whether to use ablated context. Options: 'cipher', 'none', 'masked",
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
        "--backend_config",
        type=str,
        default=None,
        help="Optional JSON/TOML config file for the selected --backend (e.g., temperature/extra/api_key; reasoning and base_url are openrouter/argo-specific).",
    )
    parser.add_argument(
        "--cipher_swap_rate",
        type=float,
        default=1.0,
        help="Cipher swap rate for the experiment. Default is 1.0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=argparse.SUPPRESS,
        help="Sampling temperature. Overrides 'temperature' in --openrouter_config if both are given.",
    )

    args = parser.parse_args()
    target_types = args.target_type

    backend_cfg = {}
    if args.backend_config:
        if args.backend == "argo":
            backend_cfg = load_argo_config(args.backend_config)
        else:
            backend_cfg = load_openrouter_config(args.backend_config)

    model_name = (
        getattr(args, "model_name", None)
        or backend_cfg.get("model")
        or DEFAULT_MODEL_NAMES[args.backend]
    )

    cli_temperature = getattr(args, "temperature", None)
    temperature = (
        cli_temperature
        if cli_temperature is not None
        else backend_cfg.get("temperature")
    )

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
        results_dir = os.path.join(
            args.results_dir, model_name, f"target_{target_type}"
        )
        os.makedirs(results_dir, exist_ok=True)
        responses_dir = os.path.join(
            args.responses_dir, model_name, f"target_{target_type}"
        )
        os.makedirs(responses_dir, exist_ok=True)
        examples_file = os.path.join(
            args.examples_dir, f"target_{target_type}", "example_contexts.jsonl"
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
            examples_dir=os.path.dirname(examples_file),
        )

        for index, row in dataset.iterrows():
            target_string = row[target_type]

            def get_context_col(target_type, context_ablation):
                if context_ablation == "none":
                    return f"{target_type}_context"
                elif context_ablation == "cipher":
                    return f"cipher_ablation_rate_{args.cipher_swap_rate}_target_{target_type}"
                elif context_ablation == "masked":
                    return f"masked_context"
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
                print("End-to-end example prompt:\n", flush=True)
                print(prompt, flush=True)

            print(f"Input context for {target_string}: {prompt}", flush=True)
            # Prompt the model
            try:
                cfg_max_tokens = backend_cfg.get("max_tokens")
                cfg_extra = backend_cfg.get("extra")
                cfg_api_key = backend_cfg.get("api_key")
                
                if args.backend == "argo":
                    response = argo_chat_completion(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        base_url=backend_cfg.get("base_url"),
                        api_key=cfg_api_key,
                        temperature=temperature,
                        max_tokens=cfg_max_tokens,
                        reasoning_effort=backend_cfg.get("reasoning_effort"),
                        extra=cfg_extra,
                    )
                else:
                    response = openrouter_chat_completion(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        api_key=cfg_api_key or os.environ.get("OPENROUTER_API_KEY"),
                        temperature=temperature,
                        max_tokens=cfg_max_tokens,
                        reasoning=backend_cfg.get("reasoning"),
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
            if ((index + 1) % 10 == 0) or (index == len(dataset) - 1):
                print(f"Saving results to {results_file}...", flush=True)
                results_df.to_json(results_file, orient="records", lines=True)

        print(results_df.head(), flush=True)
