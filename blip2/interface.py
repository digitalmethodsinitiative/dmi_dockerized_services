import argparse
import requests
import json
import PIL
from urllib.parse import quote_plus
from pathlib import Path
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# Add model types to download
model_types = ["Salesforce/blip2-opt-2.7b"]

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """
    Parse command line arguments
    """
    cli = argparse.ArgumentParser()
    cli.add_argument("--image-folder", "-i", help="Path to folder containing images", required=True)
    cli.add_argument("--model", "-m", help=f"Model for (options: {', '.join(model_types)})", default="Salesforce/blip2-opt-2.7b")
    cli.add_argument("--max_new_tokens", "-t", help=f"Maximum number of tokens to be generated for model caption/response.", type=int, default=20)
    cli.add_argument("--prompt", "-p", help="Output directory where annotations will be saved", default=None)
    cli.add_argument("--output-dir", "-o", help="Output directory where annotations will be saved", default="data", required=True)
    cli.add_argument("--dataset-name", "-d", help="Dataset name (to use for output file)", required=True)
    # These arguments are added by the DMI Service Manager in order for the service to, if desired, provide status updates which will be logged in the DMI Service Manager database.
    cli.add_argument("--database_key", "-k", default="",
                     help="DMI Service Manager database key to provide status updates.")
    cli.add_argument("--dmi_sm_server", "-s", default="",
                     help="DMI Service Manager server address to provide status updates.")
    return cli.parse_args()

def log(message, server=None, db_key=None):
    print(message)
    if server and db_key:
        try:
            requests.post(f"{server}/status_update/?key={db_key}&status=running&message={quote_plus(message)}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to log status update: {e}")

if __name__ == "__main__":
    args = parse_args()

    output_folder = Path(args.output_dir)
    if not output_folder.exists():
        log(f"Output folder {args.output_dir} not found.", args.dmi_sm_server, args.database_key)
        exit(1)

    # Setup models
    log("Setting up model...", args.dmi_sm_server, args.database_key)
    processor = AutoProcessor.from_pretrained(args.model)
    # TODO: check torch_dtype usage
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(device)

    # Prompt if provided
    prompt = {"text": args.prompt} if args.prompt else {}

    done = 0
    log("Processing images...", args.dmi_sm_server, args.database_key)
    with output_folder.joinpath(args.dataset_name + ".ndjson").open("w") as outfile:
        images = Path(args.image_folder)
        for image in images.glob("*"):
            metadata = {}

            try:
                image_obj = PIL.Image.open(image)
            except PIL.UnidentifiedImageError:
                print(f"Unable to open image {image.name}")
                continue

            if image_obj.mode != "RGB":
                image_obj = image_obj.convert("RGB")

            # Process image
            inputs = processor(image_obj, return_tensors="pt", **prompt).to(device, torch.float16)
            # Generate text
            # max_new_tokens is in BLIP examples; there is also min_length and max_length, but they seem to behave oddly with the prompt parameter
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            # skip_special_tokens is in BLIP examples; unsure what tokens they have marked as special to be removed
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            metadata["text"] = generated_text

            done += 1
            outfile.write(json.dumps({image.name: metadata}) + "\n")
            log(f"Processed {done} images", args.dmi_sm_server, args.database_key)
