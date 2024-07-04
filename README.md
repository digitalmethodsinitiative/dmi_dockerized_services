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
| Stable Diffusion | stable_diffusion | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 ||

# Installation
1. Install Docker itself
  -  You can find [information here](https://docs.docker.com/engine/install/) to install on Windows, Mac, or Linux.
2. Build the Docker images for your desired services

## Building a Docker image
1. Clone this repository
  - `git clone https://github.com/digitalmethodsinitiative/dmi_dockerized_services.git`
2. Navigate to the desired image
  - e.g., `cd dmi_dockerized_services/openai_whipser` for our Whisper image
  - Individual services may have `README.md` files similar to this which additional information on running the service and types of commands that you can send
3. Build the image
  - `docker build -t image_name .`
  - `image_name` can be whatever you wish, but you will need it to run the image and set up the DMI Service Manager (e.g., `docker build -t whisper .` for Whisper)
4. You can now run the image individually or set up the DMI Service Manager to run it for you and enable it in 4CAT

## Enable a service in the DMI Service Manager
Complete instructions are available in the [DMI Service Manager wiki](https://github.com/digitalmethodsinitiative/dmi_service_manager?tab=readme-ov-file#docker-images-setup)
1. Enable the service in DMI Service Manager's `config.yml` under the `DOCKER_ENDPOINTS` section. E.g.,:
```
whisper: 
 image_name: whisper
 local: True  # Set to True if 4CAT is running locally
 remote: False  # Set to True if 4CAT is running remotely
 command: whisper
 data_path: /app/data/
```

## Enable a service in 4CAT
1. First enable the service in the DMI Service Manager (see above)
2. On your 4CAT server, navigate to Control Panel -> Settings -> DMI Service Manager (you must be an administrator).
  - Set the DMI Service Manager server/URL to the server where the DMI Service Manager is running (e.g. http://localhost:4000) 
  - Set DMI Services Local or Remote to either local or remote depending on whether 4CAT and the DMI Service Manager are on the same server
  - Find and enable the server you have just created and adjust any relevant settings
  - Run the analysis on an appropriate dataset!

## Run the container persistently 
Most services are set up to run persistantly allowing you to send multiple commands as desired.

### Run container as daemon from image
`docker run -v $(pwd)/data/:/app/data/ --name container_name --gpus all -d image_name`
-  `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
- `--name container_name` names the container "container_name"
- `--gpus all` is needed for the container to use the host GPUs
- `-d` runs the container and disconnects
- the final argument is the tag assigned to your image when built above

Connect to container to run `container_name` commands

`docker exec -it container_name bash`

### Run the container for a single use
This docker run command combines the above to create a one time use Docker container to run commands

Assuming you have a folder `data` in your current working directory with audio files in it, run:
`docker run --rm -v $(pwd)/data/:/app/data/ --gpus all container_name command_of_choice`
- `--rm` removes the container after it has completed
