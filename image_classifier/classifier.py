import argparse
import numpy as np
import json
import torch
import PIL
from urllib.parse import quote_plus
import requests

from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path

have_cuda = torch.cuda.is_available()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifiers = {
    "features": {
        "model_name": "google/vit-base-patch16-224",
        "model": None,
        "preprocessor": None
    },
    "celebrities": {
        "model_name": "tonyassi/celebrity-classifier",
        "model": None,
        "preprocessor": None
    },
    "nsfw": {
        "model_name": "Falconsai/nsfw_image_detection",
        "model": None,
        "preprocessor": None
    }
}


# Copied from transformers.pipelines.text_classification.softmax
def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def parse_args():
    """
    Parse command line arguments
    """
    cli = argparse.ArgumentParser()
    cli.add_argument("--image-folder", "-i", help="Path to folder containing images", required=True)
    cli.add_argument("--with-features", "-f", help=f"Annotate with ImageNet-based classes ({classifiers['features']['model_name']})", action="store_true", default=False)
    cli.add_argument("--with-celebrities", "-c", help=f"Annotate with celebrity detection ({classifiers['celebrities']['model_name']})", action="store_true", default=False)
    cli.add_argument("--with-nsfw", "-n", help=f"Annotate with NSFW detection ({classifiers['nsfw']['model_name']})", action="store_true", default=False)
    cli.add_argument("--output-dir", "-o", help="Output directory where annotations will be saved", default="data",
                     required=True)
    cli.add_argument("--dataset-name", "-d", help="Dataset name (to use for output file)", required=True)
    cli.add_argument("--label-threshold", "-t", help="Threshold for confidence in labels to be included in output (default 0.5, or 50%%)", default=0.5, type=float)
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

if __name__ == "__main__":
    args = parse_args()

    output_folder = Path(args.output_dir)
    if not output_folder.exists():
        print(f"Output folder {args.output_dir} not found.")
        exit(1)

    for classifier, settings in classifiers.items():
        if not getattr(args, f"with_{classifier}"):
            continue

        classifiers[classifier]["preprocessor"] = AutoImageProcessor.from_pretrained(settings["model_name"])
        classifiers[classifier]["model"] = AutoModelForImageClassification.from_pretrained(settings["model_name"]).to(
            device)

    done = 0
    with output_folder.joinpath(args.dataset_name + ".ndjson").open("w") as outfile:
        images = Path(args.image_folder)
        for image in images.glob("*"):
            metadata = {}

            try:
                image_obj = PIL.Image.open(image)
            except PIL.UnidentifiedImageError:
                continue

            if image_obj.size == (1, 1):
                continue

            if image_obj.mode != "RGB":
                image_obj = image_obj.convert("RGB")

            for classifier, settings in classifiers.items():
                if not settings["preprocessor"]:
                    continue

                inputs = settings["preprocessor"](image_obj, return_tensors="pt").to(device)

                with torch.no_grad():
                    output = settings["model"](**inputs).logits
                    scores = softmax(output[0].cpu().numpy())
                    metadata.update({classifier: {
                        settings["model"].config.id2label[i]: score.item() for i, score in enumerate(scores) if score > args.label_threshold}
                    })

            previous = True
            done += 1

            outfile.write(json.dumps({image.name: metadata}) + "\n")
            log(f"Processed {done} images", args.dmi_sm_server, args.database_key, num_records=done)
            