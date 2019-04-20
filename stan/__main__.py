import logging
import argparse

from stan.annotate import annotate


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    required=True,
    type=str,
    help="path containing the dataset to be converted",
)
parser.add_argument(
    "--output-dir",
    required=True,
    type=str,
    help="path to store the converted dataset",
)
parser.add_argument(
    "--dataset",
    required=True,
    type=str,
    choices=["semeval2010task8"],
    help="name of the dataset",
)
parser.add_argument(
    "--corenlp",
    required=True,
    type=str,
    help="path or host of Stanford CoreNLP",
)
parser.add_argument(
    "--output-format",
    type=str,
    default="json",
    choices=["json", "jsonl", "tacred"],
    help="name of the dataset",
)
parser.add_argument(
    "--validation-size",
    type=float,
    default=0,
    help="if > 0, split this fraction from train for validation",
)
parser.add_argument(
    "--shuffle",
    action="store_true",
    default=False,
    help="whether to shuffle train",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1111,
    help="initial seed for the RNG",
)
parser.add_argument(
    "--n-jobs",
    type=int,
    default=1,
    help="the number of request jobs to run in parallel",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="enable debug logging",
)


def main():
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=log_level
    )

    annotate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        corenlp=args.corenlp,
        dataset=args.dataset,
        output_format=args.output_format,
        shuffle=args.shuffle,
        validation_size=args.validation_size,
        seed=args.seed,
        n_jobs=args.n_jobs,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
