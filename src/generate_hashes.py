"""Generate random hex strings with configurable length and count."""

from __future__ import annotations

import secrets
from typing import Iterator, List
import os
import json
import argparse


class RandomHashGenerator:
    """Build cryptographically random hex strings (hash-like appearance)."""

    def __init__(self, length: int, count: int, save_path: str) -> None:
        pass

    def generate(self, length: int, count: int) -> List[str]:
        """Return `count` independent random hex strings, each of length `length`."""
        self._length = length
        self._count = count
        nbytes = (length + 1) // 2
        try:
            print(f"Generating {count} hashes of length {length}")
            hashes = [secrets.token_hex(nbytes)[:length] for _ in range(count)]
        except Exception as e:
            raise ValueError(f"Failed to generate hashes: {e}") from e
        self._hashes = hashes
        return self._hashes

    def save_to_disk(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            # save as jsonl
            with open(self._save_path, "w") as f:
                for hash in self._hashes:
                    f.write(
                        json.dumps(
                            {
                                "target_string": hash,
                                "length": self._length,
                                "count": self._count,
                            }
                        )
                        + "\n"
                    )
            print(f"Saved hashes to {self._save_path}")
        except Exception as e:
            raise ValueError(f"Failed to save hashes: {e}") from e

    def __iter__(self) -> Iterator[str]:
        nbytes = (self._length + 1) // 2
        for _ in range(self._count):
            yield secrets.token_hex(nbytes)[: self._length]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate random hex strings with configurable length and count."
    )
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="data/hashes.v1.jsonl")
    args = parser.parse_args()
    generator = RandomHashGenerator()
    generator.generate(args.length, args.count)
    generator.save_to_disk(args.save_path)
    print(
        f"Generated {args.count} hashes of length {args.length} and saved to {args.save_path}"
    )
