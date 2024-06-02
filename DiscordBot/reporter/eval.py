from misinformation_detector import (
    MisinformationDetector,
    ValidatorResponseObject,
)
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import numpy as np
from typing import List, Dict, Tuple
import tqdm


executor = ThreadPoolExecutor(max_workers=10)


def load_data(filename: str):
    assert os.path.exists(filename), f"File {filename} does not exist"
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def evaluate_dataset(
    dataset: List[Dict],
    misinformation_detector: MisinformationDetector,
    is_negative: bool,
) -> float:

    num_correct = 0

    for tweet in tqdm.tqdm(dataset):
        tweet_text = tweet["tweet_text"]

        misinformation_payload: ValidatorResponseObject = misinformation_detector(
            tweet_text
        ).get_payload()
        if misinformation_payload["flagged"] == "YES" and not is_negative:
            num_correct += 1
        elif not misinformation_payload["flagged"] == "NO" and is_negative:
            num_correct += 1

    return num_correct / len(dataset)


def write_outputs(
    positive_examples_selected, negative_examples_selected, acc_positive, acc_negative
):

    if args.write_outputs:
        if not os.path.isdir("eval_outputs"):
            os.makedirs("eval_outputs")
    import csv
    from datetime import datetime

    # Prepare metadata
    metadata = {
        "positive_example_path": args.positive_example_path,
        "negative_example_path": args.negative_example_path,
        "num_positive_examples": args.num_positive_examples,
        "num_negative_examples": args.num_negative_examples,
    }

    # Get current date
    current_date = datetime.now().strftime("%Y%m%d")

    # Define output filenames
    output_filename_positive = (
        f"eval_outputs_{args.num_positive_examples}_positive_{current_date}.csv"
    )
    output_filename_negative = (
        f"eval_outputs_{args.num_negative_examples}_negative_{current_date}.csv"
    )

    # Write positive examples and accuracy to CSV
    with open(output_filename_positive, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        for key, value in metadata.items():
            writer.writerow([key, value])
        writer.writerow(["Accuracy", acc_positive])
        writer.writerow(["Examples"])
        for example in positive_examples_selected:
            writer.writerow([example])

    # Write negative examples and accuracy to CSV
    with open(output_filename_negative, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        for key, value in metadata.items():
            writer.writerow([key, value])
        writer.writerow(["Accuracy", acc_negative])
        writer.writerow(["Examples"])
        for example in negative_examples_selected:
            writer.writerow([example])


def evaluate():
    positive_examples = load_data(args.positive_example_path)
    negative_examples = load_data(args.negative_example_path)

    # Shuffle examples
    np.random.shuffle(positive_examples)
    np.random.shuffle(negative_examples)

    positive_examples_selected = positive_examples[: args.num_positive_examples]
    negative_examples_selected = negative_examples[: args.num_negative_examples]

    misinformation_detector = MisinformationDetector()

    acc_positive = evaluate_dataset(
        positive_examples_selected, misinformation_detector, is_negative=False
    )
    acc_negative = evaluate_dataset(
        negative_examples_selected, misinformation_detector, is_negative=True
    )

    write_outputs(
        positive_examples_selected,
        negative_examples_selected,
        acc_positive,
        acc_negative,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive_example_path",
        type=str,
        default="data/positive.json",
        help="Path to positive example",
    )
    parser.add_argument(
        "--negative_example_path",
        type=str,
        default="data/negative.json",
        help="Path to negative example",
    )
    parser.add_argument(
        "--num_positive_examples",
        type=int,
        default=10,
        help="Number of positive examples to evaluate on",
    )
    parser.add_argument(
        "--num_negative_examples",
        type=int,
        default=10,
        help="Number of negative examples to evaluate on",
    )
    parser.add_argument("--write_outputs", action="store_true")
    args = parser.parse_args()
    evaluate()
