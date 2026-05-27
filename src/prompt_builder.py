"""Build prompts for OpenRouter calls from templates and data."""

import os


class PromptBuilder:
    def __init__(
        self,
        n_shot: int,
        template_file: str,
        examples: str = None,
        examples_dir: str = None,
    ):
        self.template = open(template_file, "r").read()
        self.examples = examples
        self.example_texts = []
        self.example_targets = self.examples["target_string"].tolist()[:n_shot]

        for target in self.example_targets:
            self.example_texts.append(
                open(os.path.join(examples_dir, f"{target}.txt"), "r").read()
            )

    def build_prompt(self, text: str, target_string: str) -> str:
        """Build prompt for target string for the given template and examples"""
        if target_string in self.example_targets:
            raise ValueError(
                f"Target string {target_string} is in the examples. Please filter out the examples that are in the dataset."
            )
        prompt = (
            self.template.replace("{EXAMPLES}", "\n\n".join(self.example_texts))
            .replace("{TEXT}", text)
            .replace("{TARGET_WORD}", target_string)
        )

        return prompt
