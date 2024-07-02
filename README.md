# DMI Dockerized Services
This repo contains a collection of Dockerized services allowing them to be run via a command line inside a controlled environment. 
Each container should contain a `Dockerfile` to build the proper environment and, possibly, a `docker-entrypoint.sh` file running a 
default commnads. It also may contain a command line interface file allowing different arguments to be passed to the service.

These Services were designed primarily to work with our [DMI Service Manager](https://github.com/digitalmethodsinitiative/dmi_service_manager/blob/main/readme.md) 
which allows these services to be run via an API and can be integrated with our tools such as [4CAT](https://4cat.nl). The 
services can also be run independently on local files and some contain their own APIs. Explore their individual README.md files
for more information.

## Services

| Service Name | Folder | Source                            | Notes |
|--------------|---------|-----------------------------------|-------|
| Whisper | openai_whipser | https://github.com/openai/whisper ||
| CLIP | openai_clip | https://github.com/openai/CLIP/ ||
| BLIP2 | blip2 | https://huggingface.co/Salesforce/blip2-opt-2.7b ||

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
