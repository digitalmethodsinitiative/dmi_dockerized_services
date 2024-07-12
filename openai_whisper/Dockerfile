FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set working directory
WORKDIR /app/

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3 \
    pip \
    ffmpeg

# Python to run unbuffered and print directly (to Docker logs)
ENV PYTHONUNBUFFERED=1

# Install Whisper package
RUN python3 -m pip install -U openai-whisper

# Copy project
COPY docker-entrypoint.sh /app/
COPY whisper_download_models.py /app/
RUN mkdir /app/data/

RUN chmod +x docker-entrypoint.sh whisper_download_models.py

# Download Whisper models
RUN python3 whisper_download_models.py

CMD ["./docker-entrypoint.sh"]
