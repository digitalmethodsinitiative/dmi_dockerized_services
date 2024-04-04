FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

RUN apt-get update && apt install -y \
    # General requirements:
    libpq-dev \
    python3-dev \
    python3-pip \
    git \
    wget

# Set working directory
WORKDIR /app/
COPY requirements.txt /app/
# Install python packages
RUN python3 -m pip install -U -r requirements.txt

# Copy project
COPY . /app/
RUN mkdir /app/data/
RUN chmod +x docker-entrypoint.sh download_models.py

# Download CLIP models
RUN python3 download_models.py

CMD ["./docker-entrypoint.sh"]