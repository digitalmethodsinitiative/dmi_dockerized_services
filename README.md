# DMI Dockerized Services
These are a collection of Dockerized services allowing us to deploy them as single use or readily accessible.

## Services

| Service Name | Source                            | Notes |
|--------------|-----------------------------------|-------|
| Whisper | https://github.com/openai/whisper ||
| CLIP | https://github.com/openai/CLIP/ ||

# Basic Docker setup
### Build the image
Build the image
`docker build -t image_name .`
### Run the container persistently 
Run container as daemon from image

`docker run -v $(pwd)/data/:/app/data/ --name container_name --gpus all -d image_name`
-  `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
- `--name whisper` names the container "container_name"
- `--gpus all` is needed for the container to use the host GPUs
- `-d` runs the container and disconnects

Connect to container to run `container_name` commands

`docker exec -it container_name bash`

### Run the container for a single use
This docker run command combines the above to create a one time use Docker container to run commands

Assuming you have a folder `data` in your current working directory with audio files in it, run:
`docker run --rm -v $(pwd)/data/:/app/data/ --gpus all container_name command_of_choice`
- `--rm` removes the container after it has completed