FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Install other dependencies (e.g., GCC, make)
RUN apt-get update && apt-get install -y build-essential

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Copy your CUDA code into the container
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Install additional tools for development
RUN apt-get install -y vim git wget

# Expose port for VS Code to attach
EXPOSE 22

#runs properly (and interactively) with the following command:
    #docker run -it --gpus all -p 2222:22 -v $(pwd):/workspace --name cuda-dev-env cuda-dev-env /bin/bash
# to build the image:
    #docker build -t cuda-dev-env .