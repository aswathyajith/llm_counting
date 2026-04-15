"""Generate random entities (hex strings, WordNet nouns, nonce words, cipher nouns) with configurable length and count."""

import secrets
import os
import json
import argparse
import random
from typing import List

try:
    from nltk.corpus import wordnet

    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
try:
    from markov_word_generator import MarkovWordGenerator, WordType

    MARKOV_AVAILABLE = True
except ImportError:
    MARKOV_AVAILABLE = False


class EntityGenerator:
    """Generate random entities including hex strings and WordNet common nouns."""

    def __init__(self, seed: int = None):
        self._target_strings = []
        self._real_words_cache = None
        if seed:
            random.seed(seed)

    def generate(
        self, length: int, count: int, entity_type: str = "hash", seed: int = None
    ) -> List[str]:
        """Generate entities of specified type and max length."""
        if seed:
            random.seed(seed)

        generators = {
            "hash": self._gen_hashes,
            "noun": self._gen_nouns,
            "nonce": self._gen_nonce,
            "cipher": self._gen_cipher,
        }

        if entity_type not in generators:
            raise ValueError(
                f"Unsupported entity_type: {entity_type}. Use 'hash', 'noun', 'nonce', or 'cipher'."
            )

        self._target_strings = generators[entity_type](count, length, seed)
        return self._target_strings

    def generate_all_types(
        self, count: int, max_length: int = 15, seed: int = None, save_dir: str = "data"
    ) -> dict:
        """Generate all 4 entity types with matching length distributions."""
        if seed:
            random.seed(seed)

        print(f"Generating {count} entities of each type (max length {max_length})")

        # Generate nouns to establish length distribution
        print("Generating nouns...")
        nouns = self._gen_nouns(count, max_length, seed)
        lengths = [len(n) for n in nouns]
        self._save_dataset(nouns, f"{save_dir}/nouns.c{count}.l{max_length}.jsonl")

        # Generate other types with matching lengths
        print("Generating cipher nouns...")
        cipher = self._gen_cipher_from_nouns(nouns, seed)
        self._save_dataset(cipher, f"{save_dir}/cipher.c{count}.l{max_length}.jsonl")

        print("Generating hashes...")
        hashes = self._gen_hashes_with_lengths(lengths, seed)
        self._save_dataset(hashes, f"{save_dir}/hashes.c{count}.l{max_length}.jsonl")

        print("Generating nonce words...")
        nonce = self._gen_nonce_with_lengths(lengths, seed)
        self._save_dataset(nonce, f"{save_dir}/nonce.c{count}.l{max_length}.jsonl")

        return {"noun": nouns, "cipher": cipher, "hash": hashes, "nonce": nonce}

    def _gen_hashes(self, count: int, max_length: int, seed: int = None) -> List[str]:
        """Generate hex strings with distributed lengths."""
        return [
            self._gen_single_hash(random.randint(3, max_length), seed)
            for _ in range(count)
        ]

    def _gen_single_hash(self, length: int, seed: int = None) -> str:
        """Generate single hex string of specific length."""
        if seed:
            return "".join(random.choice("0123456789abcdef") for _ in range(length))
        return secrets.token_hex((length + 1) // 2)[:length]

    def _gen_hashes_with_lengths(
        self, lengths: List[int], seed: int = None
    ) -> List[str]:
        """Generate hashes with specific lengths."""
        return [self._gen_single_hash(length, seed) for length in lengths]

    def _gen_nouns(
        self, count: int, max_length: int = None, seed: int = None
    ) -> List[str]:
        """Generate WordNet common nouns."""
        if not WORDNET_AVAILABLE:
            raise ValueError(
                "NLTK WordNet not available. Install with: pip install nltk"
            )

        nouns = []
        for synset in wordnet.all_synsets("n"):
            for lemma in synset.lemmas():
                noun = lemma.name().replace("_", " ")
                if self._is_valid_noun(noun) and (
                    not max_length or len(noun) <= max_length
                ):
                    nouns.append(noun)

        nouns = sorted(set(nouns))
        return random.sample(nouns, min(count, len(nouns)))

    def _gen_cipher(
        self, count: int, max_length: int = None, seed: int = None
    ) -> List[str]:
        """Generate cipher nouns from existing noun dataset."""
        if not os.path.exists("data/nouns.c100.l15.jsonl"):
            raise ValueError("Noun dataset not found. Generate nouns first.")

        with open("data/nouns.c100.l15.jsonl", "r") as f:
            nouns = [json.loads(line)["target_string"] for line in f if line.strip()]

        if max_length:
            nouns = [n for n in nouns if len(n) <= max_length]

        return self._gen_cipher_from_nouns(nouns[:count], seed)

    def _gen_cipher_from_nouns(self, nouns: List[str], seed: int = None) -> List[str]:
        """Apply cipher transformation to nouns."""
        mapping = self._create_cipher_mapping(seed)
        return [self._apply_cipher(noun, mapping) for noun in nouns]

    def _create_cipher_mapping(self, seed: int = None) -> dict:
        """Create substitution cipher mapping."""
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        shuffled = list(alphabet)
        if seed:
            random.seed(seed)
        random.shuffle(shuffled)

        mapping = {}
        for i, letter in enumerate(alphabet):
            mapping[letter] = shuffled[i]
            mapping[letter.upper()] = shuffled[i].upper()
        return mapping

    def _apply_cipher(self, text: str, mapping: dict) -> str:
        """Apply cipher mapping to text."""
        return "".join(mapping.get(char, char) for char in text)

    def _gen_nonce(
        self, count: int, max_length: int = None, seed: int = None
    ) -> List[str]:
        """Generate nonce words with distributed lengths."""
        lengths = [random.randint(3, max_length or 12) for _ in range(count)]
        return self._gen_nonce_with_lengths(lengths, seed)

    def _gen_nonce_with_lengths(
        self, lengths: List[int], seed: int = None
    ) -> List[str]:
        """Generate nonce words with specific lengths."""
        if not MARKOV_AVAILABLE:
            raise ValueError("markov-word-generator not available")

        self._ensure_dict_exists()
        generator = MarkovWordGenerator(
            markov_length=4, language="EN", word_type=WordType.WORD
        )

        nonce_words = []
        for length in lengths:
            # Try Markov generation
            for _ in range(100):
                word = generator.generate_word()
                if (
                    word
                    and len(word) == length
                    and word.isalpha()
                    and word.islower()
                    and not self._is_real_word(word)
                ):
                    nonce_words.append(word)
                    break
            else:
                # Fallback to phonetic generation
                nonce_words.append(self._gen_phonetic(length))

        return nonce_words

    def _gen_phonetic(self, length: int) -> str:
        """Generate phonetic fallback word."""
        vowels, consonants = "aeiou", "bcdfghjklmnpqrstvwxyz"
        word = []
        for i in range(length):
            if i == 0 or (word and word[-1] in vowels):
                word.append(random.choice(consonants))
            else:
                word.append(
                    random.choice(vowels if random.random() < 0.6 else consonants)
                )
        return "".join(word)

    def _is_valid_noun(self, noun: str) -> bool:
        """Check if noun is valid single word."""
        return (
            noun.islower()
            and noun.isalpha()
            and " " not in noun
            and "-" not in noun
            and not noun[0].isupper()
        )

    def _is_real_word(self, word: str) -> bool:
        """Check if word is real using cached WordNet lookup."""
        if self._real_words_cache is None:
            self._real_words_cache = self._build_real_words()
        return word.lower() in self._real_words_cache

    def _build_real_words(self) -> set:
        """Build comprehensive real words set."""
        words = set()
        if WORDNET_AVAILABLE:
            try:
                for synset in wordnet.all_synsets():
                    for lemma in synset.lemmas():
                        word = lemma.name().replace("_", " ").lower()
                        if " " not in word and word.isalpha() and len(word) >= 2:
                            words.add(word)
            except Exception as e:
                raise ValueError(f"Failed to build real words set: {e}") from e

        # Add fallback words
        words.update(
            [
                "apple",
                "table",
                "house",
                "water",
                "light",
                "stone",
                "tent",
                "ball",
                "plastic",
            ]
        )
        return words

    def _ensure_dict_exists(self):
        """Create Markov dictionary if needed."""
        dict_path = "dictionaries/EN-words.dic"
        if not os.path.exists(dict_path):
            os.makedirs("dictionaries", exist_ok=True)

            # Use WordNet nouns if available
            words = []
            if WORDNET_AVAILABLE:
                try:
                    for synset in wordnet.all_synsets("n"):
                        for lemma in synset.lemmas():
                            noun = lemma.name().replace("_", " ")
                            if self._is_valid_noun(noun):
                                words.append(noun)
                    words = sorted(set(words))
                except Exception as e:
                    raise ValueError(f"Failed to build dictionary: {e}") from e

            # Fallback
            if not words:
                words = [
                    "apple",
                    "table",
                    "house",
                    "water",
                    "light",
                    "garden",
                    "mountain",
                    "forest",
                ]

            with open(dict_path, "w") as f:
                for word in words:
                    f.write(word + "\n")

    def _save_dataset(self, entities: List[str], path: str):
        """Save entities to JSONL file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for entity in entities:
                f.write(
                    json.dumps(
                        {
                            "target_string": entity,
                            "length": len(entity),
                            "count": len(entities),
                        }
                    )
                    + "\n"
                )
        print(f"Saved {len(entities)} entities to {path}")

    def save_to_disk(self, save_path: str = None):
        """Save current entities to disk."""
        if not save_path:
            raise ValueError("save_path required")
        self._save_dataset(self._target_strings, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate entities with configurable length and count"
    )
    parser.add_argument(
        "--length", type=int, default=16, help="Max length for generated entities"
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of entities to generate"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/entities.v1.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--type",
        choices=["hash", "noun", "nonce", "cipher"],
        default="hash",
        help="Entity type: hash, noun, nonce, or cipher",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all 4 entity types with matching length distributions",
    )
    parser.add_argument(
        "--save_dir", type=str, default="data", help="Directory for --generate-all"
    )
    args = parser.parse_args()

    generator = EntityGenerator(seed=args.seed)

    if args.generate_all:
        all_entities = generator.generate_all_types(
            args.count, args.length, args.seed, args.save_dir
        )
        print(
            f"\nGenerated {args.count} entities of each type with matching length distributions:"
        )
        for entity_type, entities in all_entities.items():
            lengths = [len(e) for e in entities]
            print(
                f"  {entity_type.capitalize()}: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.2f}"
            )
    else:
        entities = generator.generate(args.length, args.count, args.type, args.seed)
        generator.save_to_disk(args.save_path)
        print(f"Generated {len(entities)} {args.type}s and saved to {args.save_path}")
