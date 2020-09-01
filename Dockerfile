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
ARG PYTHON_VERSION
# Install conda.
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN curl --silent --show-error --location --output ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

FROM conda AS conda-installs
ARG PYTORCH_VERSION=1.6.0
ARG TORCHVISION_VERSION=0.7.0
# Install pytorch for cpu and torchvision.
ENV NO_CUDA=1
RUN conda install pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} cpuonly -y -c pytorch && conda clean -ya

FROM conda AS build
ARG PYTHON_VERSION
ARG GROUP_ID=1000
ARG USER_ID=1000
ENV PYTHON_VERSION=${PYTHON_VERSION}
# Build the tutorials.
WORKDIR /pytorch-cpp
# INstall
RUN pip install --upgrade --no-cache-dir cmake && \
    groupadd --gid ${GROUP_ID} user && \
    useradd --uid ${USER_ID} --gid user  --create-home --no-log-init --shell /bin/bash user && \
    chown --changes --silent --no-dereference --recursive ${USER_ID}:${GROUP_ID} /home/user
USER user
ENV HOME=/home/user
COPY --from=conda-installs /opt/conda /opt/conda
COPY --chown=user:user ./docker/docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT [ "/docker-entrypoint.sh" ]
CMD [ "bash" ]
