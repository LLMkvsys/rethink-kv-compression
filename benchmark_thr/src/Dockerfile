# Base image
# FROM continuumio/miniconda3:latest
# FROM nvidia/cuda:12.4.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Working dir
WORKDIR /mlsys

# Install Miniconda, since nv base image doesn't have conda installed
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
# Update PATH to include Conda
ENV PATH="/opt/conda/bin:$PATH"
# Verify Conda installation
RUN conda --version

# Copy environment files
COPY mlsys_environment.yml mlsys_requirements.txt ./

# Copy code
COPY ./src-docker .

# Create Conda environment and install pip dependencies
RUN conda env create -f mlsys_environment.yml
RUN conda clean -a
RUN pip install --no-cache-dir -r mlsys_requirements.txt
RUN pip install -e .

# Activate environment and run default code
CMD ["conda", "run", "-n", "lmdeploy", "python", "0_test.py"]