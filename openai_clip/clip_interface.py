import argparse
import importlib
import os
import torch
import torchvision
import clip
from PIL import Image
from pathlib import Path
import json
from urllib.parse import quote_plus
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """
    Parse command line arguments
    """
    cli = argparse.ArgumentParser()
    cli.add_argument("--available_models", "-a", default=False, help="Get available models.", action="store_true")
    cli.add_argument("--dataset", "-d", default="", help="Use existing torchvision.dataset for image categories.")
    cli.add_argument("--model", "-m", default="", help="CLIP model.")
    cli.add_argument("--output_dir", "-o", default="", help="Directory to store JSON results.")
    cli.add_argument("--categories", "-c", default="", help="Categories to classify image (comma seperated list).")
    cli.add_argument("--images", "-i", nargs="+", type=str, help="Image(s) to classify.")
    # These arguments are added by the DMI Service Manager in order for the service to, if desired, provide status updates which will be logged in the DMI Service Manager database.
    cli.add_argument("--database_key", "-k", default="",
                     help="DMI Service Manager database key to provide status updates.")
    cli.add_argument("--dmi_sm_server", "-s", default="",
                     help="DMI Service Manager server address to provide status updates.")

    return cli.parse_args()

def log(message, server=None, db_key=None, num_records=None):
    print(message)
    if server and db_key:
        try:
            requests.post(f"{server}/status_update/?key={db_key}&status=running&message={quote_plus(message)}{'&num_records=' + str(num_records) if num_records else ''}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to log status update: {e}")


def collect_image_categories(dataset_name):
    if dataset_name not in torchvision.datasets.__all__:
        raise ValueError(f"Invalid dataset type: {dataset_name}")

    torch_dataset = getattr(importlib.import_module("torchvision.datasets"), dataset_name)
    try:
        dataset = torch_dataset(root=os.path.expanduser("~/.cache"), download=True)
    except TypeError as e:
        print(e)
        raise ValueError(f"Unable to load dataset type: {dataset_name}")

    return dataset.classes


def get_available_models():
    return clip.available_models()


def load_model(model_name):
    if model_name not in get_available_models():
        raise ValueError(f"Invalid model name: {model_name}. Available models: {get_available_models()}")

    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess


def predict_image_category_probabilities(model, preprocess, image_path, text_list):
    """
    Possibly helpful, but the `top_labels` function is probably more useful.
    :param model:
    :param preprocess:
    :param image_path:
    :param text_list:
    :return:
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(text_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs = sorted([(i, j) for i, j in zip(text_list, probs[0])], key=lambda x: x[1], reverse=True)

    print("Label probs:")
    for prob in probs[:5]:
        print(f"{prob[0]}: {100 * prob[1]:.2f}%")

    return probs


def top_labels(model, preprocess, classes, image_path):
    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(len(classes))

    # Print the result
    predictions = []
    print(f"\nTop predictions for {Path(image_path).name}:")
    for value, index in zip(values, indices):
        predictions.append((classes[index], value.item()))

        if len(predictions) < 6:
            print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

    return predictions


if __name__ == "__main__":
    args = parse_args()
    if args.available_models:
        print(get_available_models())
        exit(0)

    if args.dataset:
        try:
            classes = collect_image_categories(args.dataset)
        except ValueError as e:
            print(e)
            exit(1)
        if args.categories:
            print("Cannot specify both --dataset and --categories.")
            exit(1)
    elif args.categories:
        classes = args.categories.split(",")
    else:
        print("Must specify either --dataset or --categories.")
        exit(1)

    if not args.model:
        print("Must specify --model.")
        exit(1)

    if not args.images:
        print("Must specify at least one image.")
        exit(1)

    try:
        model, preprocess = load_model(args.model)
    except ValueError as e:
        print(e)
        exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(".")

    for i, image in enumerate(args.images):
        if Path(image_path).is_file():
            prediction = top_labels(model, preprocess, classes, image)
            results = {"filename": Path(image).name,
                       "predictions": prediction}
        else:
            error = f"Invalid image path {image}"
            print(error)
            results = {"filename": Path(image).name,
                       "error": error}

        with open(output_dir.joinpath(Path(image).with_suffix(".json").name), "w") as out_file:
            out_file.write(json.dumps(results))

        log(f"Processed {i + 1} images", args.dmi_sm_server, args.database_key, num_records=i + 1)
