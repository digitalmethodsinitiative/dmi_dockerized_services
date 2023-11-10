FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set working directory
WORKDIR /app/

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3 \
    pip \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0

# Copy project
COPY . /app/
RUN mkdir /app/data/

# pre-cache model
RUN git lfs install
RUN rm -rf stable-diffusion-xl-base-1.0
RUN git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
RUN git clone https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

# Install pip dependencies
RUN pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu117

# Start
RUN chmod +x docker-entrypoint.sh
CMD ["./docker-entrypoint.sh"]
