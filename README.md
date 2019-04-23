[![CircleCI](https://circleci.com/gh/ChristophAlt/StAn.svg?style=svg)](https://circleci.com/gh/ChristophAlt/StAn)

# StAn - Quickly annotate your dataset with Stanford CoreNLP

In natural language processing, algorithms often require additional linguistic features (syntactic and semantic), such as part-of-speech, named entity, and dependency tags; information that is not readily available in most datasets. StAn provides a convenient way to quickly annotate an existing dataset with additional linguistic features computed by [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/).

## Getting Started

### Prerequisites

StAn either uses a local CoreNLP installation or an exisiting [CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html). To use a local installation, download and unpack the latest version from the [Stanford CoreNLP website](https://stanfordnlp.github.io/CoreNLP/).

### Installing

#### With pip

```bash
TBD
```

#### From Source

Clone the repository and run:

```bash
pip install [--editable] .
```

## Usage

For example, the following command annotates the [SemEval 2010 Task 8](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk) relation extraction dataset with POS, NER, and dependency information and saves it in [JSONL](http://jsonlines.org) format.

```bash
stan \
    --input-dir $INPUT_PATH/SemEval2010_task8_all_data/ \
    --output-dir $OUTPUT_PATH/ \
    --corenlp $PATH_TO_CORENLP_JAR_OR_SERVER_URL \
    --input-format semeval2010task8 \
    --output-format jsonl \
    --shuffle \
    --validation-size 0.1 \
    --n-jobs 4
```

### Parameters:

- `input-dir`: the directory containing the dataset or dataset files. StAn expects a specific structure for common datasets (e.g. SemEval 2010 Task 8). The format of the input is specified by `input-format`.
- `output-dir`: the directory to store the annotated dataset. The format in which to save the dataset is specified by `output-format`.
- `corenlp`: the path to the directory containing the CoreNLP jar file or a url pointing to an exisiting CoreNLP server.
- `input-format`: the format of the input dataset, can be one of "semeval2010task8", "json" or "jsonl".
- `output-format`: the format of the output dataset, can be one of "tacred", "json", "jsonl".
- `shuffle`: whether to shuffle the training dataset before splitting into train and validation (only if validation size > 0).
- `validation-size`: if > 0, use a `validation-size` fraction of the training dataset for validation.
- `n-jobs`: the number of threads to use for concurrent requests to CoreNLP.


## Running the tests

Explain how to run the automated tests for this system

### Unittests

```bash
pytest -v tests/
```

### Typechecker and coding style tests

```bash
mypy stan --ignore-missing-imports
```

## Built With

* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
* [stanford-corenlp](https://github.com/Lynten/stanford-corenlp) - Python wrapper for Stanford CoreNLP

## Authors

* **Christoph Alt**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
