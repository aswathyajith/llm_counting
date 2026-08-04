### See notebooks/context_extraction.ipynb for initial development of context generation.

"""

We collect Wikipedia articles for the target noun words in the dataset (extracted from WordNet) and present them as context for the target word to be counted.

"""

import os
import random
import argparse
import pandas as pd
import wikipediaapi
from src.data.target_str import TargetMatcher
import requests
from src.data.context_ablation import context_ablation


def get_mentions(row):
    pattern, text = row["noun"], row["noun_context"]
    tgtmatcher = TargetMatcher(pattern, text)
    case_space_standalone = [
        {"case": 0, "space": 0, "standalone": 0},
        {"case": 0, "space": 0, "standalone": 1},
        {"case": 0, "space": 1, "standalone": 0},
        {"case": 0, "space": 1, "standalone": 1},
        {"case": 1, "space": 0, "standalone": 0},
        {"case": 1, "space": 0, "standalone": 1},
        {"case": 1, "space": 1, "standalone": 0},
        {"case": 1, "space": 1, "standalone": 1},
    ]

    for case_space_standalone in case_space_standalone:
        case, space, standalone = (
            case_space_standalone["case"],
            case_space_standalone["space"],
            case_space_standalone["standalone"],
        )
        tgtmatcher.set_regex(case=case, space=space, standalone=standalone)
        row[f"case{case}_space{space}_standalone{standalone}"] = (
            tgtmatcher.get_num_matches()
        )

        # replace target string with nonce, hash, cipher, etc.
        if case == space == standalone == 0:
            nonce = row["nonce"]
            hash = row["hash"]
            cipher = row["cipher"]
            nonce_context = tgtmatcher.replace_matches(nonce)
            hash_context = tgtmatcher.replace_matches(hash)
            cipher_context = tgtmatcher.replace_matches(cipher)

            row["nonce_context"] = nonce_context
            row["hash_context"] = hash_context
            row["cipher_context"] = cipher_context

    return row


def fetch_disambiguation_options(page_title):
    """Query MediaWiki API to check if a title is a disambiguation page and return its link options."""
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "pageprops|links",
            "titles": page_title,
            "pllimit": "max",
            "format": "json",
            "redirects": 1,
        },
        headers={"User-Agent": "llm_counting_experiment/0.1 (your_email@uchicago.edu)"},
        timeout=30,
    )
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for _, page_data in pages.items():
        is_disambig = "disambiguation" in page_data.get("pageprops", {})
        if is_disambig:
            return [link["title"] for link in page_data.get("links", [])]
    return None


def get_page_from_options(wiki, wiki_matches, page_title):
    wiki_matches = [
        m.replace('"', "").replace("'", "")
        for m in wiki_matches
        if page_title in m.lower() and "all pages with" not in m.lower()
    ]
    num_matches = len(wiki_matches)

    random.seed(42)
    if num_matches == 0:
        return 0, None
    chosen = random.choice(wiki_matches)
    page = wiki.page(chosen)
    return num_matches, page if page.exists() else None


def get_wiki_contexts(nouns_df: pd.DataFrame) -> pd.DataFrame:
    wiki = wikipediaapi.Wikipedia(
        user_agent="llm_counting_experiment/0.1 (your_email@uchicago.edu)",
        language="en",
    )

    wiki_nouns_df = nouns_df.copy()

    wiki_nouns_df["num_matches"] = 0
    wiki_nouns_df["disambiguation"] = 0
    wiki_nouns_df["title"] = ""
    wiki_nouns_df["noun_context"] = ""
    wiki_nouns_df["matches"] = ""

    for index, row in nouns_df.iterrows():
        disambiguation = 0
        num_matches = 0
        page_title = row["noun"]
        wiki_matches = []
        page = None

        disambig_options = fetch_disambiguation_options(page_title)

        if disambig_options is not None:
            disambiguation = 1
            wiki_matches = disambig_options
            num_matches, page = get_page_from_options(wiki, wiki_matches, page_title)
        else:
            candidate = wiki.page(page_title)
            if candidate.exists() and page_title.lower() in candidate.title.lower():
                page = candidate
                num_matches = 1
                wiki_matches = [page_title]
            else:
                print(f"{page_title} is not a valid wikipedia page", flush=True)
                continue

        wiki_nouns_df.at[index, "num_matches"] = num_matches
        if num_matches > 0 and page is not None:
            print("Found Wikipedia page for", page_title, ":", page.title, flush=True)
            wiki_nouns_df.at[index, "disambiguation"] = int(disambiguation)
            wiki_nouns_df.at[index, "title"] = page.title
            wiki_nouns_df.at[index, "noun_context"] = page.text
            wiki_nouns_df.at[index, "matches"] = wiki_matches
        else:
            print("No Wikipedia page found for", page_title, flush=True)

    return wiki_nouns_df


def replace_target(context, target_string, replacement_string):
    tgtmatcher = TargetMatcher(target_string, context)
    tgtmatcher.set_regex(case=0, space=0, standalone=0)
    return tgtmatcher.replace_matches(replacement_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Generation")
    parser.add_argument(
        "--target_words_file",
        type=str,
        default="data/matched_entities.c100.l15.jsonl",
        help="File containing the target words for the experiment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/contexts",
        help="Save path for the processed dataset",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="wiki",
        help="Source of the context (wiki, news, social media, etc.)",
    )
    parser.add_argument(
        "--cipher_swap_rates",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        help="Swap rates for cipher ablation (space-separated list)",
    )

    args = parser.parse_args()

    dataset = pd.read_json(args.target_words_file, lines=True)
    output_dir = os.path.join(args.output_dir, args.source)

    wiki_nouns_df = get_wiki_contexts(dataset)
    wiki_nouns_df = wiki_nouns_df.apply(lambda row: get_mentions(row), axis=1)

    # Perform cipher ablation on the context with different swap rates
    for cipher_swap_rate in args.cipher_swap_rates:
        wiki_nouns_df = context_ablation(
            wiki_nouns_df,
            "noun_context",
            f"cipher_ablation_rate_{cipher_swap_rate}",
            "cipher",
            cipher_swap_rate,
        )

    wiki_nouns_df["reverse_noun"] = wiki_nouns_df["noun"].apply(lambda x: x[::-1])
    wiki_nouns_df = context_ablation(
        wiki_nouns_df, "noun_context", "reverse_ablation", "reverse"
    )

    # Change all target instances to ciphered version
    for cipher_swap_rate in args.cipher_swap_rates:
        wiki_nouns_df[f"cipher_ablation_rate_{cipher_swap_rate}_target_cipher"] = (
            wiki_nouns_df.apply(
                lambda row: replace_target(
                    row[f"cipher_ablation_rate_{cipher_swap_rate}"],
                    row["noun"],
                    row["cipher"],
                ),
                axis=1,
            )
        )

    # Change all target instances to original version

    for cipher_swap_rate in args.cipher_swap_rates:
        wiki_nouns_df[f"cipher_ablation_rate_{cipher_swap_rate}_target_noun"] = (
            wiki_nouns_df.apply(
                lambda row: replace_target(
                    row[f"cipher_ablation_rate_{cipher_swap_rate}"],
                    row["cipher"],
                    row["noun"],
                ),
                axis=1,
            )
        )

    # Replacing reverse target string with original target string in the reverse-ablated contexts
    wiki_nouns_df["reverse_ablation"] = wiki_nouns_df.apply(
        lambda row: replace_target(
            row["reverse_ablation"], row["reverse_noun"], row["noun"]
        ),
        axis=1,
    )

    # [TODO] Swap word order ablation
    # [TODO] Dist-shifted ablation to test coherence / consistency: Replace alphabetical cipher mapping to a numeric mapping

    os.makedirs(output_dir, exist_ok=True)
    wiki_nouns_df.to_json(
        os.path.join(output_dir, "contexts.jsonl"), orient="records", lines=True
    )
