# Base image
FROM python:3.11-slim

# Install Python and dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy the required files into the image
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Set the working directory
WORKDIR /

# Ensure all dependencies are installed
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Set the entry point to the evaluation script
ENTRYPOINT ["python", "-u", "src/cnn_mnist/evaluate.py"]
