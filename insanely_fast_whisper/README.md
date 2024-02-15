# DMI implementation of Whisper audio transcription tool
Build a Docker image based on `nvidia/cuda` to quickly deploy containers with 
[Whisper](https://github.com/openai/whisper) and its models ready to use.

## Basic Docker setup
### Build the image
Build the image
`docker build -t fast_whisper .`
### Run the container persistently 
Run container as daemon from image

`docker run -v $(pwd)/data/:/whisper/data/ --name fast_whisper --gpus all -d fast_whisper`
-  `-v $(pwd)/data/:/app/data/` mounts the `data` directory in your current working directory to the container
- `--name whisper` names the container "whisper"
- `--gpus all` is needed for the container to use the host GPUs; remove and Whisper will run without GPUs albeit MUCH more slowly
- `-d` runs the container and disconnects

Connect to container to run `` commands

`docker exec -it fast_whisper bash`

#TODO.... HARNESS ITS POWER
```
# Faster pipeline
results = []

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from transformers.pipelines.pt_utils import KeyDataset
import os
import time
data_dir = "data/audio-extractor-eee3c59a900423f83827051f9c364ca2/"
audio_files = [file for file in os.listdir(data_dir) if file[-4:] == ".wav"]
dataset = KeyDataset([{'path':data_dir + file} for file in audio_files], "path")
transcripts = {}
models = ["distil-whisper/distil-large-v2", "openai/whisper-large-v3"]
for model in models:
    start_time = time.time()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device="cuda:0", # or mps for Mac devices
        model_kwargs={"use_flash_attention_2": is_flash_attn_2_available()},
    )
    after_load_time = time.time()
    mypipe = pipe(
                    dataset,
                    chunk_length_s=30,
                    batch_size=24,
                    return_timestamps=True,
                    return_language=True,
                )
    for i, audio_file in enumerate(mypipe):
        filename = dataset[i]
        if filename not in transcripts:
            transcripts[filename] = {}
        transcripts[filename][model] = audio_file
        print(filename, ":\n", audio_file.get("text"), "\n")
    end_time = time.time()
    results.append(f"{model} - transcript time: {end_time-after_load_time}; total time: {end_time-start_time}")

# View results
for file, transcript in transcripts.items():
     print(file)
     for model, text in transcript.items():
         print(model, text)
     print()
```
Once connected...

`whisper --output_dir data/ --output_format json --model medium data/*`

will output .json files with transcripts into your local `data` directory for each audio file in the same `data` directory.

### Run the container for a single use
This docker run command combines the above to create a one time use Docker container to run this `whisper` command

Assuming you have a folder `data` in your current working directory with audio files in it, run:
`docker run --rm -v $(pwd)/data/:/app/data/ --gpus all whisper whisper --output_dir data/ --output_format json --model tiny data/*`
The results will be in your local `data` directory as .json files.
