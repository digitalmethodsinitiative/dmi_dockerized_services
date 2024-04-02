# Image classification Docker image

This Docker image runs a simple script to run a number of image classification models from HuggingFace on a given image
dataset. It reports all labels an image is annotated with by any of the models with at least a certain confidence level.

### Build the Docker image
1. Navigate to this folder
2. Install the models:
   - `git clone --depth 1 https://huggingface.co/google/vit-base-patch16-224`
   - `git clone --depth 1 https://huggingface.co/tonyassi/celebrity-classifier`
   - `git clone --depth 1 https://huggingface.co/Falconsai/nsfw_image_detection`
   These are pre-loaded this way because `git clone` is not cached by Docker builds, so any subsequent builds would need
   to download the full models and weights again, which can take a long time. Once you are happy with the image 
   (probably immediately after building) you can delete these repositories from the folder. You may need to install 
   `git-lfs` (for example via `apt install` first).
3. Build the image: `docker build -t image_classifier .` 

The build process pre-loads the models from Hugging Face. If you want to add other models or change the ones used, 
simply edit the references in `classifier.py` and adjust the build accordingly.

### Run the container persistently 
Run container as daemon from image

`docker run -v $(pwd)/data/:/app/data/ --name image_classifier --gpus all -d image_classifier`
-  `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
- `--name image_classifier` names the container "image_classifier"
- `--gpus all` is needed for the container to use the host GPUs; remove and it will run without GPUs albeit MUCH more 
  slowly (unusably, basically)
- `-d` runs the container and disconnects

### Run the container for a single use
This Docker run command combines the above to create a one time use Docker container to run the prompt via the interface
script discussed below.

Assuming you have a folder `data` in your current working directory with files in it, run:
`docker run --rm -v $(pwd)/data/:/app/data/ --gpus all image_classifier bash -c "python3 classifier.py --help"`

### Output
Annotations are saved as an .ndjson file, with one line per classified image, formatted as follows:

```json
{
  "image-filename.jpg": {
    "features": {
      "web site, website, internet site, site": 0.9856060147285461,
      "comic book": 0.6124848127365112
    },
    "celebrities": {
      "Johnny Depp": 0.99977010238953,
      "Keira Knightley": 0.7143849120365015
    },
    "nsfw": {
      "normal": 0.9997770190238953
    }
  }
}
```