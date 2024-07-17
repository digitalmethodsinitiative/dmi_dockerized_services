FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set working directory
WORKDIR /app/

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3 \
    pip \
    git

# Python to run unbuffered and print directly (to Docker logs)
ENV PYTHONUNBUFFERED=1

# Install Clip packages and dependencies
RUN python3 -m pip install -U ftfy regex tqdm requests && \
    python3 -m pip install -U git+https://github.com/openai/CLIP.git

# Copy and download models first (helps speed builds)
COPY download_models.py /app/
RUN chmod +x download_models.py
RUN python3 download_models.py

# Copy project
COPY . /app/
RUN mkdir /app/data/

RUN chmod +x docker-entrypoint.sh

CMD ["./docker-entrypoint.sh"]
