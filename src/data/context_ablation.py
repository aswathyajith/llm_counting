import random
import pandas as pd


def create_cipher_mapping() -> dict:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    shuffled = list(alphabet)
    random.seed(999)
    random.shuffle(shuffled)
    mapping = {}
    for i, letter in enumerate(alphabet):
        mapping[letter] = shuffled[i]
        mapping[letter.upper()] = shuffled[i].upper()
    return mapping


def map_words_in_text_with_prob(text: str, replace_prob: float, mapping: dict) -> str:
    map_word_cipher = False
    new_text = ""
    for i in range(len(text)):
        # if it is the first character or
        # previous character is a whitespace
        # then a new word is starting
        # print(i)
        if (i == 0) or (text[i - 1].isspace()):
            random_toss = random.random()
            if random_toss <= replace_prob:
                map_word_cipher = True
            else:
                map_word_cipher = False
        if text[i] in mapping and map_word_cipher:
            new_text += mapping[text[i]]
        else:
            new_text += text[i]
    return new_text


def context_ablation(
    data: pd.DataFrame,
    original_column: str = "noun_context",
    new_column: str = "ablated_cipher_context",
    type: str = "cipher",
    cipher_swap_rate: float = 1.0,
) -> pd.DataFrame:
    if type == "cipher":
        mapping = create_cipher_mapping()
        data[new_column] = data[original_column].apply(
            lambda x: map_words_in_text_with_prob(x, cipher_swap_rate, mapping)
        )
    elif type == "reverse":
        data[new_column] = data[original_column].apply(lambda x: x[::-1])
    return data
