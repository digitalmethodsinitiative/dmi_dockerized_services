import itertools
import argparse
import torch
import yaml
import json
import sys

from pathlib import Path
from stormtrooper import Text2TextZeroShotClassifier, Text2TextFewShotClassifier, GenerativeZeroShotClassifier, \
    GenerativeFewShotClassifier

have_cuda = torch.cuda.is_available()
gpu_or_cpu = "cuda:0" if have_cuda else "cpu"

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", "-m", help="Model ID: HuggingFace ID or `openai/model_id` for OpenAI models",
                     required=True)
    cli.add_argument("--prompt", "-p", help="Text prompt")
    # cli.add_argument("--parameters", help="Extra model parameters, as a JSON-encoded object")
    cli.add_argument("--apikey", "-a", help="OpenAI API key (only needed when using OpenAI models)")
    cli.add_argument("--inputfile", "-i", help="NDJSON file containing items to classify, item id -> text",
                     required=True)
    cli.add_argument("--labelfile", "-l", help="Labels to use, JSON file with an object, label -> [list of examples]",
                     required=True)
    cli.add_argument("--output-dir", "-o", help="Output directory where image will be saved", default="data",
                     required=True)

    args = cli.parse_args()

    model_map = {}
    with open("local-models.yml") as infile:
        available_models = yaml.full_load(infile)
        for model_type, models in available_models.items():
            model_map.update({model: model_type for model in models})

    if args.model not in model_map:
        print(f"Model {args.model} is not available or enabled.", file=sys.stderr)
        exit(1)

    model_type = model_map[args.model]

    labelpath = Path(args.labelfile)
    inputpath = Path(args.inputfile)
    outputpath = Path(args.output_dir).joinpath("results.json")

    if not labelpath.exists():
        print(f"Label file not available at {labelpath}.", file=sys.stderr)
        exit(1)

    if not inputpath.exists():
        print(f"Input data not available at {inputpath}.", file=sys.stderr)
        exit(1)

    with labelpath.open() as infile:
        try:
            labels = json.load(infile)
        except json.JSONDecodeError:
            print(f"Read error while loading labels from {labelpath}. Make sure the file is valid JSON.",
                  file=sys.stderr)
            exit(1)

    if model_type in ("text2text", "textgen"):
        have_examples = any(labels.values())
        model_name = args.model.split("/").pop()
        if have_examples:
            # fit() expects a list of examples as the first arg, and a list of
            # corresponding labels as the second arg
            examples = itertools.chain(*[v for v in labels.values()])
            chained_labels = itertools.chain(*[[l] * len(labels[l]) for l in labels.keys()])
            predictor = {"text2text": Text2TextFewShotClassifier, "textgen": GenerativeFewShotClassifier}[model_type](
                model_name=model_name, device=gpu_or_cpu, progress_bar=False)
            predictor.fit(examples, chained_labels)
        else:
            predictor = {"text2text": Text2TextZeroShotClassifier, "textgen": GenerativeZeroShotClassifier}[model_type](
                model_name=model_name, device=gpu_or_cpu, progress_bar=False)
            predictor.fit(None, labels.keys())

        # we *could* just load all data into a list and pass it to the predictor in
        # its entirety
        # but, we don't know how large it is - it could be millions of items
        # so instead work in batches, and also write the result in batches, so that
        # we don't use a theoretically infinite amount of memory
        batch_size = 100
        input_exhausted = False
        looping = True
        batch = {}

        # clear output file
        with outputpath.open("w") as outfile:
            pass

        # loop through items to label
        line = 0
        with inputpath.open() as infile:
            while looping:
                if len(batch) >= batch_size or input_exhausted:
                    predicted_labels = predictor.predict(batch.values())

                    if (predicted_labels is None):
                        print(
                            f"Got no predictions for batch. Saving results so far and halting. Batch was {batch}.",
                            file=sys.stderr)
                        input_exhausted = True
                    else:
                        for item_id in batch.keys():
                            with outputpath.open("a") as outfile:
                                written_bytes = outputpath.stat().st_size
                                if written_bytes == 0:
                                    outfile.write("{\n")
                                    outfile.flush()
                                elif written_bytes > 2:
                                    outfile.write(",\n")
                                outfile.write(f"  {json.dumps(item_id)}: {json.dumps(predicted_labels.pop(0))}")

                    batch = {}
                    if input_exhausted:
                        with outputpath.open("a") as outfile:
                            outfile.write("\n}")
                        looping = False

                try:
                    line += 1
                    try:
                        item = json.loads(next(infile).strip())
                        batch.update(
                            item)  # theoretically, could be any number of items per line (but one is recommended)
                    except json.JSONDecodeError:
                        print(
                            f"Error parsing line {line:,} from {inputpath} as JSON. Saving results so far and halting.",
                            file=sys.stderr)
                        input_exhausted = True

                except StopIteration:
                    input_exhausted = True

    else:
        print(f"OpenAI models are currently not supported.", file=sys.stderr)
        exit(1)
