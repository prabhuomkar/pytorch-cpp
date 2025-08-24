# Based on and inspired by Dockerfiles and resources at:
#   https://github.com/pytorch/pytorch/blob/master/Dockerfile
#   https://github.com/anibali/docker-pytorch
#   https://jtreminio.com/blog/running-docker-containers-as-current-host-user/

ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} AS dev-base
# Install basic packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

FROM dev-base AS conda
# Install conda.
ARG PYTHON_VERSION
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN curl --silent --show-error --location --output ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

FROM conda AS conda-installs
# Install pytorch for CPU and torchvision.
ARG PYTORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ENV NO_CUDA=1
RUN conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cpuonly -y -c pytorch && conda clean -ya

FROM conda AS build
# Build tutorials.
ARG PYTHON_VERSION
ARG GROUP_ID=1000
ARG USER_ID=1000
ENV PYTHON_VERSION=${PYTHON_VERSION}
WORKDIR /pytorch-cpp
RUN pip install --upgrade --no-cache-dir cmake && \
    groupadd --gid ${GROUP_ID} pytorch && \
    useradd --uid ${USER_ID} --gid pytorch  --create-home --no-log-init --shell /bin/bash pytorch && \
    chown --changes --silent --no-dereference --recursive ${USER_ID}:${GROUP_ID} /home/pytorch
USER pytorch
ENV HOME=/home/pytorch
COPY --from=conda-installs /opt/conda /opt/conda
COPY --chown=pytorch:pytorch ./docker/docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT [ "/docker-entrypoint.sh" ]

LABEL maintainer="prabhuomkar@pm.me,markus.fleischhacker28@gmail.com"
