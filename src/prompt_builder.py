"""Build prompts for OpenRouter calls from templates and data."""

from string import Template
from typing import Any, Dict, List
import argparse
import json
import jsonlines


def build_prompt_from_template(template: str, data: Dict[str, Any]) -> str:
    """Build a prompt from a template and data dictionary."""

    filled_template = template.copy()
    user_prompt = Template(filled_template["user_prompt"]).substitute(
        target_string=data
    )
    print(user_prompt)
    print(filled_template)
    filled_template["user_prompt"] = user_prompt
    return filled_template


def prompt_dataset_generator(
    template_file: str, data_file: str
) -> List[Dict[str, Any]]:
    """Generate a dataset of prompts from a template and dataset."""
    template = json.load(open(template_file, "r"))
    data = []
    with jsonlines.open(data_file) as reader:
        for line in reader:
            print(line)
            target_string = line["target_string"]
            length = line["length"]
            count = line["count"]
            data.append(
                {
                    "target_string": target_string,
                    "length": length,
                    "count": count,
                    "prompt": build_prompt_from_template(template, target_string),
                }
            )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a prompt from a template and data."
    )
    parser.add_argument("--template_file", type=str, default="prompts/templates.json")
    parser.add_argument("--data_file", type=str, default="data/hashes.v1.jsonl")
    parser.add_argument("--output_file", type=str, default="data/prompt_dataset.jsonl")
    args = parser.parse_args()

    data = prompt_dataset_generator(args.template_file, args.data_file)
    with jsonlines.open(args.output_file, "w") as writer:
        for entry in data:
            writer.write(entry)
