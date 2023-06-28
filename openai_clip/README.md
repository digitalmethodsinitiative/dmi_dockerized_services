# DMI implementation of OpenAI CLIP image categorization tool
Build a Docker image based on `nvidia/cuda` to quickly deploy containers with 
[CLIP](https://github.com/openai/CLIP) and its models ready to use.

## Basic Docker setup
### Build the image
Build the image
`docker build -t clip .`
### Run the container persistently 
Run container as daemon from image

`docker run -v $(pwd)/data/:/app/data/ --name clip --gpus all -d clip`
-  `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
- `--name clip` names the container "clip"
- `--gpus all` is needed for the container to use the host GPUs; remove and Whisper will run without GPUs albeit MUCH more slowly
- `-d` runs the container and disconnects

#### DMI CLIP command line tool
We developed a simple command line tool to allow you to run CLIP on a folder of images and output the results as a .json file.
1. Connect to container to run commands
`docker exec -it clip bash`
2. Check available models
`python3 clip_interface.py --available_models`
3. Run CLIP on a folder of images and categorize them as "cats", "dogs", or "other"
`python3 clip_interface.py --model ViT-B/16 --output_dir data/output/ --categories cats,dogs,other --images data/*`
4. Or run CLIP on a folder of images and categorize them using an existing [Torchvision](https://pytorch.org/vision/stable/datasets.html) dataset
Note: not all datasets are available for download
`python3 clip_interface.py --model ViT-B/32 --dataset CIFAR100 --output_dir data/output/ --images data/*`

will output .json files with transcripts into your local `data` directory for each audio file in the same `data` directory.

### Run the container for a single use
This docker run command combines the above to create a one time use Docker container to run this `whisper` command

Assuming you have a folder `data` in your current working directory with audio files in it, run:
`docker run --rm -v $(pwd)/data/:/app/data/ --gpus all clip bash -c "python3 clip_interface.py --model ViT-B/16 --output_dir data/output/ --categories cats,dogs,other --images data/*"`
The results will be in your local `data` directory as .json files.
