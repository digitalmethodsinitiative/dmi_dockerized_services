FROM nvidia/cuda:12.1.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt install -y \
    # General requirements:
    libpq-dev \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    git \
    wget
# Set Python 3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip for Python 3.9
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app/
COPY requirements.txt /app/
# Install python packages
RUN python3 -m pip install -U -r requirements.txt
# Python to run unbuffered and print directly (to Docker logs)
ENV PYTHONUNBUFFERED=1

# Copy project
COPY . /app/
RUN mkdir /app/data/
RUN chmod +x docker-entrypoint.sh download_models.py

# Download CLIP models
RUN python3 download_models.py

CMD ["./docker-entrypoint.sh"]