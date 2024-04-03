import argparse
import json
import PIL
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
    cli.add_argument("--prompt", "-p", help="Output directory where annotations will be saved", default=None)
    cli.add_argument("--output-dir", "-o", help="Output directory where annotations will be saved", default="data", required=True)
    cli.add_argument("--dataset-name", "-d", help="Dataset name (to use for output file)", required=True)
    return cli.parse_args()

if __name__ == "__main__":
    args = parse_args()

    output_folder = Path(args.output_dir)
    if not output_folder.exists():
        print(f"Output folder {args.output_dir} not found.")
        exit(1)

    # Setup models
    processor = AutoProcessor.from_pretrained(args.model)
    # TODO: check torch_dtype usage
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(device)

    # Prompt if provided
    prompt = {"text": args.prompt} if args.prompt else {}

    done = 0
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
            # TODO: check max_new_tokens usage
            # TODO: check skip_special_tokens usage
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            metadata["text"] = generated_text

            done += 1
            outfile.write(json.dumps({image.name: metadata}) + "\n")
            print(f"Processed {done} images", end="\r")
    print(f"Finished w/ {done} images")