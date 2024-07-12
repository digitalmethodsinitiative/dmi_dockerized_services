FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app/

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3 \
    pip

# Copy interface files and pre-loaded models
COPY . /app/

# Python to run unbuffered and print directly (to Docker logs)
ENV PYTHONUNBUFFERED=1

# Install pip dependencies
RUN pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118

# make default output folder
RUN mkdir /app/data/

# Start
RUN chmod +x docker-entrypoint.sh
CMD ["./docker-entrypoint.sh"]
