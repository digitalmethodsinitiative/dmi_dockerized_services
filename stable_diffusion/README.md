# Stable Diffusion Docker image

Stable Diffusion is an image model that can generate images for a given text prompt. There are multiple versions and
variants, this image is based on [Stable Diffusion XL 
1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

## DMI Docker image
The DMI Docker image is set up to allow easy interfacing with the model to automatically generate images for a prompt or
list of prompts.

### Build the Docker image
1. Navigate to this folder
2. Install the models:
   - `git clone --depth 1 https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0`
   - `git clone --depth 1 https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0`
   These are pre-loaded this way because `git clone` is not cached by Docker builds, so any subsequent builds would need
   to download the full models and weights again, which takes super long. Once you are happy with the image (probably 
   immediately after building) you can delete these repositories from the folder. You may need to install `git-lfs` (for
   example via `apt install` first).
3. Build the image: `docker build -t stable_diffusion .`
4. `rm -rf stable-diffusion-xl-*` if you're not going to rebuild the images and want to free up some disk space. 

The build process pre-loads the models from Hugging Face, these are huge (20+ gigabytes) so building will take quite a 
while. The resulting image will be enormous (~200 GB) because the uncompressed models and weights are. Welcome to the
age of foundation models!

### Run the container persistently 
Run container as daemon from image

`docker run -v $(pwd)/data/:/app/data/ --name stable_diffusion --gpus all -d stable_diffusion`
-  `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
- `--name stable_diffusion` names the container "stable_diffusion"
- `--gpus all` is needed for the container to use the host GPUs; remove and it will run without GPUs albeit MUCH more 
  slowly (unusably, basically)
- `-d` runs the container and disconnects

### Run the container for a single use
This Docker run command combines the above to create a one time use Docker container to run the prompt via the interface
script discussed below.

Assuming you have a folder `data` in your current working directory with audio files in it, run:
`docker run --rm -v $(pwd)/data/:/app/data/ --gpus all stable_diffusion bash -c "python3 interface.py --prompt 'rasta bill gates'"`

### Command line tool
We developed a simple command line tool to do things with the model:c
1. Connect to container to run commands
  `docker exec -it stable_diffusion bash`
2. Generate a single image
  `python3 interface.py --prompt 'rasta bill gates'`
3. Or run a list of prompts stored as JSON (for 4CAT compatbility): 
  `python3 interface.py --prompts-file prompts.json`
4. See the full range of options
  `python3 interface.py --help`

Image files will appear in the `data` folder by default. File names reflect the prompt, and include a unique ID if 
provided  via the JSON file.