# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

USER root

USER $NB_UID

USER root
WORKDIR $HOME

# Install spatialindex
RUN apt-get update
RUN apt-get install -y libspatialindex-c4v5

# Install CUDA dependencies
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y build-essential
RUN apt-get install -y gnupg
RUN apt-get install -y apt-transport-https

# Add local files as late as possible to avoid cache busting
COPY .docker/ /usr/local/bin/
COPY .config/ /usr/local/bin/
RUN chmod +x /usr/local/bin/*
RUN fix_permissions.sh
RUN install_cuda_nvidia.sh

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

COPY ./requirements.txt /home/$NB_USER

RUN pip install -r requirements.txt

CMD ["start_notebook.sh"]