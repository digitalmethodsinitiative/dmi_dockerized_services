import argparse
import requests
import json
import torch
import sys
import re
from urllib.parse import quote_plus
import requests

from diffusers import DiffusionPipeline
from pathlib import Path

have_cuda = torch.cuda.is_available()


def parse_args():
    """
    Parse command line arguments
    """
    cli = argparse.ArgumentParser()
    cli.add_argument("--prompts-file", "-f", help="Path to prompt file, one prompt per line")
    cli.add_argument("--prompt", "-p", help="Text prompt")
    cli.add_argument("--negative-prompt", "-n", help="Negative prompt", default="")
    cli.add_argument("--steps", "-s", help="Number of steps (default 40)", default=40, type=int)
    cli.add_argument("--output-dir", "-o", help="Output directory where image will be saved", default="data", required=True)
    # These arguments are added by the DMI Service Manager in order for the service to, if desired, provide status updates which will be logged in the DMI Service Manager database.
    cli.add_argument("--database_key", "-k", default="",
                     help="DMI Service Manager database key to provide status updates.")
    cli.add_argument("--dmi_sm_server", "-m", default="",
                     help="DMI Service Manager server address to provide status updates.")

    return cli.parse_args()

def log(message, server=None, db_key=None, num_records=None):
    print(message)
    if server and db_key:
        try:
            requests.post(f"{server}/status_update/?key={db_key}&status=running&message={quote_plus(message)}{'&num_records=' + str(num_records) if num_records else ''}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to log status update: {e}")

def make_filename(prompt_id, prompt):
    """
    Generate filename for generated image

    Should mirror the make_filename method in generate_images.py in 4CAT.

    :param prompt_id:  Unique identifier, eg `54`
    :param str prompt:  Text prompt, will be sanitised, e.g. `Rasta Bill Gates`
    :return str:  For example, `54-rasta-bill-gates.jpeg`
    """
    safe_prompt = re.sub(r"[^a-zA-Z0-9 _-]", "", prompt).replace(" ", "-").lower()[:90]
    return f"{prompt_id}-{safe_prompt}.jpeg"


def use_sdxl1(args, prompts):
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        "stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    if have_cuda:
        base.to("cuda")
    else:
        base.enable_model_cpu_offload()

    refiner = DiffusionPipeline.from_pretrained(
        "stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    if have_cuda:
        refiner.to("cuda")
    else:
        refiner.enable_model_cpu_offload()

    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = args.steps
    high_noise_frac = 0.8

    if args.prompts_file:
        with open(args.prompts_file) as infile:
            prompts = json.load(infile)
    else:
        prompts = {1: {"prompt": args.prompt, "negative": args.negative_prompt}}

    done = 0
    for prompt_id, prompt in prompts.items():
        if not prompt["prompt"]:
            continue

        print(repr(prompt), file=sys.stderr)

        # run both experts
        image = base(
            prompt=prompt["prompt"],
            negative_prompt=prompt["negative"],
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        image = refiner(
            prompt=prompt["prompt"],
            negative_prompt=prompt["negative"],
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        filename = make_filename(prompt_id, prompt["prompt"])
        image.save(Path(args.output_dir).joinpath(filename))
        done += 1

        log(f"Generated {done} image(s)", args.dmi_sm_server, args.database_key, num_records=done)
        
