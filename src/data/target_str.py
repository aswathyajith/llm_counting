import re


class TargetMatcher:
    def __init__(
        self,
        target_str: str,
        text: str,
        case: bool = False,
        space: bool = False,
        standalone: bool = False,
    ):
        self.target_str = target_str
        self.text = text
        self.case = case
        self.space = space
        self.standalone = standalone

    def set_regex(self, case: bool, space: bool, standalone: bool) -> None:
        self.case = case
        self.space = space
        self.standalone = standalone

        target_str = self.target_str.lower() if case == 0 else self.target_str
        prefix = r"(?<= )" if space else (r"(?<!\w)" if standalone else "")
        suffix = r'(?=[\s.,!?;:\\\'"]|$)' if standalone else ""

        self.regex = f"{prefix}{target_str}{suffix}"

    def get_num_matches(self) -> int:
        return len(
            re.findall(self.regex, self.text, re.IGNORECASE if self.case == 0 else 0)
        )

    def replace_matches(self, new_str: str) -> str:
        """Replace all matches of the target string with new_str"""

        # Replace with capitalized new_str if the target string is capitalized
        def match_case_replace(replacement):
            def replace(match):
                word = match.group()
                if word.isupper():
                    return replacement.upper()
                elif word.istitle():
                    return replacement.title()
                elif word.islower():
                    return replacement.lower()
                else:
                    return replacement

            return replace

        return re.sub(
            self.regex,
            match_case_replace(new_str),
            self.text,
            flags=re.IGNORECASE if self.case == 0 else 0,
        )

    def set_text(self, text: str) -> str:
        self.text = text
        self.set_regex(self.case, self.space, self.standalone)

    def get_text(self) -> str:
        return self.text

    def set_target_str(self, target_str: str) -> str:
        self.target_str = target_str
        self.set_regex(self.case, self.space, self.standalone)

    def get_target_str(self) -> str:
        return self.target_str
