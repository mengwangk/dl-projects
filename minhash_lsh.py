# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
from snapy import LSH, MinHash

SEED = 3


def load_content(sentence_file):
    """Load input file with sentences to build LSH.

    Args:
        sentence_file (str): Path to input with txt file with sentences to Build LSH.

    Returns:
        dict: Dict with strings and version of string in lower case and without comma.

    """
    sentences = {}
    with open(sentence_file) as content:
        for line in content:
            line = line.strip()
            line_clean = line.replace(",", "")
            line_clean = line_clean.lower()
            sentences[line_clean] = line

    return sentences


def create_lsh(content, no_of_bands, n_permutations, n_gram):
    """Create Minhash and Locality Sensitive Hashing (LSH) to detect near duplicate texts.

    Args:
        content (list):       List with string to build LSH.
        no_of_bands (int):    Number of bands to break minhash signature into before hashing into buckets.
        n_permutations (int): Number of permutations used to create minhash signatures used in LSH model.
        n_gram (int):         Size of each overlapping text shingle to break text into prior to hashing.
        no_of_bands(int):     Number of bands to break minhash signature into before hashing into buckets.

    Returns:
        class 'snapy.lsh.LSH':  Snapy LSH object.

    """
    labels = range(len(content))

    # Create MinHash object.
    minhash = MinHash(
        content, n_gram=n_gram, permutations=n_permutations, hash_bits=64, seed=SEED
    )

    # Create LSH model.
    lsh = LSH(minhash, labels, no_of_bands=no_of_bands)

    return lsh


def find_near_duplicate(
    query_sentences,
    sentences,
    min_jaccard_value=0.25,
    no_of_bands=50,
    n_permutations=100,
    n_gram=9,
):
    """Using LSH object finds the near duplicate strings.

    Args:
        query_sentences (dict): Dict with query strings and version of string in lower case and without comma.
        sentences (dict): Dict with target strings and version of string in lower case and without comma.
        min_jaccard_value (float): Minimum value for the Jaccard Distance.
        no_of_bands (int): Number of bands to break minhash signature into before hashing into buckets.
        n_permutations (int): Number of permutations used to create minhash signatures used in LSH model.
        n_gram (int): Size of each overlapping text shingle to break text into prior to hashing.

    """
    content = list(query_sentences.keys()) + list(sentences.keys())
    lsh = create_lsh(content, no_of_bands, n_permutations, n_gram)

    # Query to find near duplicates the string in `search`
    closest_results = lsh.query(0, min_jaccard=min_jaccard_value)

    for index_query, search_string in enumerate(query_sentences):
        print("{} QUERY: {}".format(index_query + 1, query_sentences[search_string]))
        for content_index in closest_results:
            result = content[content_index]
            print(sentences[result])
        print()


def parse_args():
    """Parse args entered by the user.

    Returns:
        argparse.Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description="Detect near duplicate texts using Minhash and Locality Sensitive Hashing.",
        epilog="example > python3 find_near_duplicate.py  -q INPUT -t TARGERS",
    )
    parser.add_argument(
        "-q", "--query", help="Path to file with sentences to query", required=True
    )
    parser.add_argument(
        "-t",
        "--targets",
        help="Path to file with sentences be matched against",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--n_gram",
        help="Size of each overlapping text shingle to break text into "
        "prior to hashing",
        default=9,
    )
    parser.add_argument(
        "-p",
        "--n_permutations",
        help="Number of permutations used to create minhash signatures used "
        "in LSH model.",
        default=100,
    )
    parser.add_argument(
        "-j",
        "--min_jaccard",
        help="Jaccard similarity threshold texts have to exceed to be "
        "returned as similar.",
        default=0.25,
    )
    parser.add_argument(
        "-b",
        "--no_of_bands",
        help="Number of bands to break minhash signature into "
        "before hashing into buckets..",
        default=50,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    query = args.query
    targets = args.targets
    min_jaccard_value = float(args.min_jaccard)
    n_gram = int(args.n_gram)
    n_permutations = int(args.n_permutations)
    no_of_bands = int(args.no_of_bands)

    # load sentences from file
    query_sentences = load_content(query)
    targets_sentences = load_content(targets)

    # find near duplicate sequences to `search_string`
    find_near_duplicate(
        query_sentences,
        targets_sentences,
        min_jaccard_value,
        no_of_bands,
        n_permutations,
        n_gram,
    )


def load_dataset():
    """Load and return databaset in dictionary."""
    dataset_name = "4D.zip"
    dataset_path = Path("datasets")
    dataset = dataset_path / dataset_name
    df = pd.read_csv(
        dataset,
        header=0,
        sep=",",
        quotechar='"',
        parse_dates=["DrawDate"],
        dtype={"PrizeType": str},
    )
    draws = {}
    for i in range(len(df)):
        draw_no = df.iloc[i, 0]
        nos = []
        for j in range(2, 25):
            nos.append(df.iloc[i, j])
        nos = sorted(nos)
        draws[draw_no] = nos
        # print(draws)
        # break
    return draws


def find_adjacency(draws):
    draws_idx = list(draws)
    draws_nos = list(draws.values())

    # labels = [draws_idx[0]]
    # content = [" ".join(map(str, draws_nos[0]))]
    # print(labels)
    # print(content)

    # minhash = MinHash(content, n_gram=9, permutations=100, hash_bits=64, seed=3)
    # lsh = LSH(minhash, labels, no_of_bands=50)
    # print(lsh.query(1, min_jaccard=0.5))

    new_labels = []
    new_content = []
    for i in range(0, len(draws)):
        new_labels.append(draws_idx[i])
        new_content.append(" ".join(map(str, draws_nos[i])))
    # print(new_labels)
    # print(new_content)
    new_labels.append(99999)
    new_content.append(" ".join(map(str, draws_nos[0])))

    minhash = MinHash(new_content, n_gram=9, permutations=500, hash_bits=64, seed=3)
    lsh = LSH(minhash, new_labels, no_of_bands=100)

    adjacency_list = lsh.adjacency_list(min_jaccard=0.5)
    for key, value in adjacency_list.items():
        if len(value) > 0:
            print(key, value)


if __name__ == "__main__":
    # query_sentences = load_content("datasets/query.txt")
    # target_sentences = load_content("datasets/targets.txt")
    # find_near_duplicate(query_sentences, target_sentences)
    draws = load_dataset()
    find_adjacency(draws)
    # for key, value in draws.items():
    #     print(value)
