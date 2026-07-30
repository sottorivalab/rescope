FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV HDF5_USE_FILE_LOCKING FALSE
ENV NUMBA_CACHE_DIR /tmp

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge3 with Python 3.11
ENV CONDA_DIR=/opt/conda
RUN wget --quiet "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p $CONDA_DIR && \
    rm ~/miniforge.sh

# Add conda to path and initialize for shell interaction
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash

RUN conda install -n base -c conda-forge mamba

# Copy the conda environment YAML file
COPY environment.yml /tmp/environment.yml

RUN CONDA_OVERRIDE_CUDA="12.4" mamba env create -f /tmp/environment.yml && \
    mamba clean -afy && \
    rm -rf /opt/conda/pkgs /tmp/environment.yml

CMD ["/bin/bash"]
