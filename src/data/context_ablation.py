import random
import pandas as pd


def context_ablation(
    data: pd.DataFrame,
    original_column: str = "noun_context",
    new_column: str = "ablated_cipher_context",
    type: str = "cipher",
) -> pd.DataFrame:
    if type == "cipher":
        random.seed(999)
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        shuffled = list(alphabet)
        random.shuffle(shuffled)
        mapping = {}

        def get_mapped_string(string):
            return "".join(
                [mapping[char] if char in mapping else char for char in string]
            )

        for i, letter in enumerate(alphabet):
            mapping[letter] = shuffled[i]
            mapping[letter.upper()] = shuffled[i].upper()
        data[new_column] = data[original_column].apply(get_mapped_string)
    return data
