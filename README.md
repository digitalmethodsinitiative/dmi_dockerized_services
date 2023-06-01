# DMI implementation of Whisper audio transcription tool
Build a Docker image based on `nvidia/cuda` to quickly deploy containers with 
[Whisper](https://github.com/openai/whisper) and its models ready to use.

## Basic Docker setup
Build the image
`sudo docker build -t whisper .`
Run container as daemon from image
`sudo docker run -v $(pwd)/data/:/app/data/ --name whisper --gpus all -d whisper`
    - `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
    - `--name whisper` names the container "whisper"
    - `--gpus all` is needed for the container to use the host GPUs
    - `-d` runs the container and disconnects
Connect to container to run `whisper` commands
`sudo docker exec -it whisper bash`
Once connected...
`whisper --output_dir data/ --output_format json --model medium data/*`
will output .json files with transcripts into your local `data` directory for each audio file in the same `data` directory.