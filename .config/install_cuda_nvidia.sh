#!/bin/bash

# Install nvidia repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo dpkg --install cuda-repo-ubuntu1804_10.1.243-1_amd64.deb

# Install CUDA GPG key
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install nvidia docker repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install cuda and nvidia-toolkit with standard commands
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo apt-get install -y cuda

# Update path variable
cat >> ~/.bashrc << EOT
export PATH="/usr/local/cuda-9.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH"
EOT