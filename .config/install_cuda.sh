#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
sudo dpkg --install cuda-repo-ubuntu1604_9.2.88-1_amd64.deb

cat >> ~/.bashrc << EOT
export PATH="/usr/local/cuda-9.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH"
EOT