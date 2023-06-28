FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set working directory
WORKDIR /app/

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3 \
    pip \
    git

# Install Whisper package
RUN python3 -m pip install -U ftfy regex tqdm && \
    python3 -m pip install -U git+https://github.com/openai/CLIP.git

# Copy project
COPY . /app/
RUN mkdir /app/data/

RUN chmod +x docker-entrypoint.sh download_models.py

# Download CLIP models
RUN python3 download_models.py

CMD ["./docker-entrypoint.sh"]
