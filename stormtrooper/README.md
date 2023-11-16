# Stormtrooper
Stormtrooper is a Python library that makes it easy to use large language models (LLMs) for data classification. It can
load a variety of models to run locally, or connect to remote APIs (e.g. OpenAI's).

Read more in the [library documentation](https://centre-for-humanities-computing.github.io/stormtrooper/).

## DMI Docker image
The DMI Docker image is set up to allow easy interfacing with the library to classify data, taking advantage of a GPU if
it is available.

### Build the Docker image
1. Navigate to this folder
2. Install the models you want to use ahead of time, so they don't need to be downloaded when the library is first used.
   You can download models by using `git clone`, if the model is available on Hugging Face. Once the Docker image has
   been built, the downloaded model can be removed. For large models, `git-lfs` may need to be installed (e.g. via apt).
   - `git clone --depth 1 https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0`
3. Build the image: `docker build -t stormtrooper .`
4. Remove the downloaded models if you're not going to rebuild the images and want to free up some disk space. 

Note that language models can be huge, so building the image can take a while and the resulting image can require quite
a bit of disk space, depending on the models you decide to pre-load.

### Run the container persistently 
Run container as daemon from image

`docker run -v $(pwd)/data/:/app/data/ --name stormtrooper --gpus all -d stormtrooper`
-  `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
- `--name stormtrooper` names the container "stormtrooper"
- `--gpus all` is needed for the container to use the host GPUs; remove and it will run without GPUs albeit MUCH more 
  slowly (unusably, basically)
- `-d` runs the container and disconnects