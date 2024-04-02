FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set working directory
WORKDIR /whisper/

# Install dependencies
RUN apt-get update && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3 \
    pip \
    libsndfile1-dev \
    ffmpeg \
    git

# Install Whisper package
RUN python3 -m pip install -U whisperplus flash-attn transformers optimum accelerate

# Copy project
COPY docker-entrypoint.sh /whisper/
# COPY whisper_download_models.py /app/
RUN mkdir /whisper/data/

RUN chmod +x docker-entrypoint.sh
#whisper_download_models.py

# Download Whisper models
#RUN python3 whisper_download_models.py

CMD ["./docker-entrypoint.sh"]
