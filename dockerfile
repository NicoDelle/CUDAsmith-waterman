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

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb \
    && apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub \
    && dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb \
    && apt-get update \
    && apt-get install -y nsight-systems-2024.4.2

# Compile with NVCC
RUN nvcc -arch=sm_70 -o sw smith-waterman.cu

# Expose port for VS Code to attach
EXPOSE 22

#runs properly (and interactively) with the following command:
    #docker run -it --gpus all -p 2222:22 -v $(pwd):/workspace --name cuda-dev-env cuda-dev-env /bin/bash
# to build the image:
    #docker build -t cuda-dev-env .